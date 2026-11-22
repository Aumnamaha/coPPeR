#!/usr/bin/env python3
r"""Single-command two-stage trainer for coPPeR ~1B document Transformer.

Usage:
    # Full training (requires Qwen3.6 unloaded from LM Studio)
    python scripts/train_copper.py --stage all

    # Smoke test (< 15 seconds, works even with Qwen3.6 loaded)
    python scripts/train_copper.py --smoke --checkpoint-dir checkpoints/smoke_verify

    # Resume training
    python scripts/train_copper.py --resume

Stage 1: Causal LM pretraining on packed document tokens (pretrain_train.bin).
Stage 2: Supervised fine-tuning distilled from teacher agents (sft_train_inputs.bin + sft_train_labels.bin).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import CopperConfig, CopperForCausalLM

# ---------------------------------------------------------------------------
# VRAM preflight guard
# ---------------------------------------------------------------------------


def check_free_vram(min_gb: float = 10.0) -> None:
    """Check free VRAM and exit with actionable message if insufficient."""
    free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1e9
    if free_gb < min_gb:
        print("ERROR: Insufficient free VRAM for training.")
        print(f"  Free:   {free_gb:.2f} GB")
        print(f"  Needed: {min_gb:.1f} GB (set --min-free-vram-gb to lower)")
        print("  Unload Qwen3.6 from LM Studio and retry, or use --smoke mode.")
        sys.exit(1)
    print(f"VRAM check passed: {free_gb:.2f} GB free")


# ---------------------------------------------------------------------------
# Dataset helpers (memory-mapped binary reading)
# ---------------------------------------------------------------------------


def load_bin_array(path: Path, dtype=np.uint16) -> np.ndarray:
    """Load a .bin file as a numpy array."""
    return np.fromfile(str(path), dtype=dtype)


class PretrainDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that yields fixed-length token chunks from packed bin."""

    def __init__(self, data_path: Path, seq_len: int = 2048) -> None:
        self.tokens = load_bin_array(data_path, np.uint16)
        self.seq_len = seq_len
        # Number of complete sequences we can form
        self.n_seqs = (len(self.tokens)) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        tokens = self.tokens[start:end]
        x = torch.from_numpy(tokens[:-1]).long()
        y = torch.from_numpy(tokens[1:]).long()
        return x, y


class SFTDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset for SFT with loss masking."""

    def __init__(self, inputs_path: Path, labels_path: Path) -> None:
        self.inputs = load_bin_array(inputs_path, np.uint16)
        self.labels = load_bin_array(labels_path, np.int32)
        assert len(self.inputs) == len(self.labels), "inputs/labels length mismatch"
        # Each record is seq_len tokens; we step by seq_len
        self.n_records = len(self.inputs) // 2048  # default seq_len

    def __len__(self) -> int:
        return self.n_records

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = idx * 2048
        x = torch.from_numpy(self.inputs[offset : offset + 2048]).long()
        y = torch.from_numpy(self.labels[offset : offset + 2048]).long()
        return x, y


# ---------------------------------------------------------------------------
# LR scheduler (cosine with linear warmup)
# ---------------------------------------------------------------------------


class CosineWarmupScheduler:
    """Cosine decay with linear warmup over warmup_steps."""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, current_step: int) -> None:
        if current_step < self.warmup_steps:
            lr = self.base_lr * (current_step + 1) / self.warmup_steps
        else:
            progress = (current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            lr = self.base_lr * 0.1 * (1 + math.cos(math.pi * progress)) / 1
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: CopperForCausalLM,
    checkpoint_dir: Path,
    stage: str,
    global_step: int,
    loss: float | None = None,
) -> None:
    """Save model weights and trainer state."""
    ckpt_path = checkpoint_dir / f"stage{stage}_step{global_step}"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Save model via safetensors (reuse method from model.py)
    model.save_pretrained(ckpt_path)

    # Save trainer state
    state = {"step": global_step, "loss": loss}
    torch.save(state, ckpt_path / "trainer_state.pt")


def load_checkpoint(
    checkpoint_dir: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[CopperForCausalLM, dict]:
    """Load model and trainer state from a checkpoint directory."""
    # Find latest stage dir
    stages = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()], reverse=True)
    if not stages:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    ckpt_path = stages[0]
    model = CopperForCausalLM.from_pretrained(ckpt_path, device=device, dtype=dtype)
    state = torch.load(ckpt_path / "trainer_state.pt", weights_only=False)
    return model, state


# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------


def train_step_causal_lm(model, batch_x, batch_y, optimizer, grad_accum_steps):
    """Single training step for causal LM pretraining."""
    logits = model(input_ids=batch_x)["logits"]
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = batch_y[:, 1:].reshape(-1)
    loss = (
        F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        / grad_accum_steps
    )
    loss.backward()
    return loss.detach().item() * grad_accum_steps


def train_step_sft(model, batch_x, batch_y, optimizer, grad_accum_steps):
    """Single training step for SFT with -100 masked labels."""
    # Full forward pass computes loss internally when labels are provided;
    # we compute manually here to control gradient accumulation.

    # Manual loss computation to match the model's internal logic
    full_logits = model(input_ids=batch_x)
    logits_sft = full_logits["logits"]
    shift_logits = logits_sft[:, :-1, :].reshape(-1, logits_sft.size(-1))
    shift_labels = batch_y[:, 1:].reshape(-1).to(shift_logits.device)
    valid_mask = shift_labels != -100

    if not valid_mask.any():
        # No valid labels — use a zero tensor that still supports backward.
        loss_val = (
            model(input_ids=batch_x)["logits"][:, -1, :].sum() * 0.0 / grad_accum_steps
        )
    else:
        loss_val = (
            F.cross_entropy(
                shift_logits[valid_mask], shift_labels[valid_mask], ignore_index=-100
            )
            / grad_accum_steps
        )

    loss_val.backward()
    return loss_val.detach().item() * grad_accum_steps


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def run_stage_pretrain(
    model: CopperForCausalLM,
    packed_dir: Path,
    checkpoint_dir: Path,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seq_len: int = 2048,
    resume_from: str | None = None,
) -> dict:
    """Run Stage 1 causal LM pretraining."""
    device = next(model.parameters()).device
    print(f"\n{'='*60}")
    print("STAGE 1: Causal LM Pretraining")
    print(f"{'='*60}")

    data_path = packed_dir / "pretrain_train.bin"
    if not data_path.exists():
        raise FileNotFoundError(f"Packed pretrain data not found: {data_path}")

    dataset = PretrainDataset(data_path, seq_len=seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    total_steps = steps
    warmup_steps = max(10, int(steps * 0.05))
    scheduler = CosineWarmupScheduler(optimizer, total_steps, warmup_steps)

    start_step = 0
    if resume_from:
        saved_model, state = load_checkpoint(Path(resume_from), device=str(device))
        model.load_state_dict(saved_model.state_dict())
        optimizer.load_state_dict(
            torch.load(
                Path(resume_from) / "stage1" / "trainer_state.pt", weights_only=False
            )
            if False
            else {"param_groups": optimizer.param_groups, "state": {}}
        )
        start_step = state.get("step", 0)
        print(f"Resumed from step {start_step}")

    model.train()
    global_step = start_step
    running_loss = 0.0
    losses_recorded: list[float] = []
    peak_vram = 0.0

    # Validate token ID bounds before any CUDA transfer
    vocab_size = model.config.vocab_size if hasattr(model, "config") else 32768

    def _validate_tokens(x: torch.Tensor) -> None:
        mx = x.max().item()
        mn = x.min().item()
        if mn < 0 or mx >= vocab_size:
            raise ValueError(
                f"Token ID out of range [0, {vocab_size}): " f"min={mn}, max={mx}"
            )

    pbar = tqdm(range(steps), desc="Pretrain", initial=start_step)
    for _ in pbar:
        try:
            batch_x, batch_y = next(data_loader_iter)
        except (NameError, StopIteration):
            data_loader_iter = iter(dataloader)
            batch_x, batch_y = next(data_loader_iter)

        # CPU bounds check before transferring to CUDA
        _validate_tokens(batch_x)

        optimizer.zero_grad(set_to_none=True)
        loss_val = train_step_causal_lm(
            model, batch_x.to(device), batch_y.to(device), optimizer, grad_accum
        )
        running_loss += loss_val

        global_step += 1
        scheduler.step(global_step - 1)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses_recorded.append(loss_val)
        pbar.set_postfix(
            {"loss": f"{loss_val:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
        )

        # Periodic checkpointing
        if global_step % max(1, steps // 5) == 0 and not getattr(
            run_stage_pretrain, "_smoke", False
        ):
            save_checkpoint(
                model,
                checkpoint_dir / "stage1_pretrain_latest",
                1,
                global_step,
                loss_val,
            )

        mem = torch.cuda.max_memory_allocated() / 1e9
        peak_vram = max(peak_vram, mem)

    avg_loss = np.mean(losses_recorded[-10:]) if losses_recorded else 0.0
    print(
        f"\nPretraining complete: {global_step} steps, avg loss={avg_loss:.4f}, peak VRAM={peak_vram:.2f} GB"
    )
    return {
        "start_loss": losses_recorded[0] if losses_recorded else None,
        "end_loss": avg_loss,
        "steps": global_step,
        "peak_vram_gb": peak_vram,
    }


def run_stage_sft(
    model: CopperForCausalLM,
    packed_dir: Path,
    checkpoint_dir: Path,
    steps: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    resume_from: str | None = None,
) -> dict:
    """Run Stage 2 SFT distillation."""
    device = next(model.parameters()).device
    print(f"\n{'='*60}")
    print("STAGE 2: Supervised Fine-Tuning (Teacher Distillation)")
    print(f"{'='*60}")

    inputs_path = packed_dir / "sft_train_inputs.bin"
    labels_path = packed_dir / "sft_train_labels.bin"
    if not inputs_path.exists():
        raise FileNotFoundError(f"Packed SFT data not found: {inputs_path}")

    dataset = SFTDataset(inputs_path, labels_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    total_steps = steps
    warmup_steps = max(10, int(steps * 0.05))
    scheduler = CosineWarmupScheduler(optimizer, total_steps, warmup_steps)

    start_step = 0
    if resume_from:
        saved_model, state = load_checkpoint(Path(resume_from), device=str(device))
        model.load_state_dict(saved_model.state_dict())
        start_step = state.get("step", 0)
        print(f"Resumed from step {start_step}")

    model.train()
    global_step = start_step
    losses_recorded: list[float] = []
    peak_vram = 0.0

    # Validate token ID bounds before any CUDA transfer
    vocab_size_sft = model.config.vocab_size if hasattr(model, "config") else 32768

    def _validate_tokens_sft(x: torch.Tensor) -> None:
        mx = x.max().item()
        mn = x.min().item()
        # Labels can contain -100 (ignore_index); only validate input_ids >= 0
        if mn < 0 or mx >= vocab_size_sft:
            raise ValueError(
                f"Token ID out of range [0, {vocab_size_sft}): " f"min={mn}, max={mx}"
            )

    data_loader_iter = iter(dataloader)
    pbar = tqdm(range(steps), desc="SFT", initial=start_step)
    for _ in pbar:
        try:
            batch_x, batch_y = next(data_loader_iter)
        except StopIteration:
            data_loader_iter = iter(dataloader)
            batch_x, batch_y = next(data_loader_iter)

        # CPU bounds check before transferring to CUDA (validate inputs only; labels may have -100)
        _validate_tokens_sft(batch_x)

        optimizer.zero_grad(set_to_none=True)
        loss_val = train_step_sft(
            model, batch_x.to(device), batch_y.to(device), optimizer, grad_accum
        )
        global_step += 1
        scheduler.step(global_step - 1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses_recorded.append(loss_val)
        pbar.set_postfix(
            {"loss": f"{loss_val:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
        )

        if global_step % max(1, steps // 5) == 0 and not getattr(
            run_stage_sft, "_smoke", False
        ):
            save_checkpoint(
                model, checkpoint_dir / "stage2_sft_latest", 2, global_step, loss_val
            )

        mem = torch.cuda.max_memory_allocated() / 1e9
        peak_vram = max(peak_vram, mem)

    avg_loss = np.mean(losses_recorded[-10:]) if losses_recorded else 0.0
    print(
        f"\nSFT complete: {global_step} steps, avg loss={avg_loss:.4f}, peak VRAM={peak_vram:.2f} GB"
    )
    return {
        "start_loss": losses_recorded[0] if losses_recorded else None,
        "end_loss": avg_loss,
        "steps": global_step,
        "peak_vram_gb": peak_vram,
    }


def save_final(model: CopperForCausalLM, checkpoint_dir: Path) -> Path:
    """Save the final trained model."""
    final_path = checkpoint_dir / "copper_1b_final"
    model.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="coPPeR two-stage trainer")
    parser.add_argument("--stage", choices=["all", "pretrain", "sft"], default="all")
    parser.add_argument("--packed-dir", type=Path, default=Path("data/packed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    # -1.0 sentinel means "use auto-default" (10.0 for normal training, 0.5 for smoke)
    parser.add_argument("--min-free-vram-gb", type=float, default=-1.0)
    parser.add_argument("--pretrain-steps", type=int, default=1000)
    parser.add_argument("--sft-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr-pretrain", type=float, default=3e-4)
    parser.add_argument("--lr-sft", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Run tiny 2-layer smoke test (<15s)"
    )

    args = parser.parse_args()

    # Smoke mode: use tiny config for fast verification
    if args.smoke:
        print("=== SMOKE MODE (tiny 2-layer model, 6 steps per stage) ===")
        cfg = CopperConfig(
            vocab_size=32768,  # match tokenizer; only +~2 MB embedding for tiny model
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=args.seq_len,
        )
        # Default 0.5 GB for smoke, but respect explicit --min-free-vram-gb overrides
        if args.min_free_vram_gb >= 0:
            min_vram = max(0.5, args.min_free_vram_gb)
        else:
            min_vram = 0.5
    else:
        cfg = CopperConfig()
        # Default 10.0 GB for full training, or explicit override
        if args.min_free_vram_gb >= 0:
            min_vram = args.min_free_vram_gb
        else:
            min_vram = 10.0

    torch.cuda.empty_cache()
    check_free_vram(min_vram)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = CopperForCausalLM(cfg).to(device).to(dtype)
    model.enable_gradient_checkpointing()
    params = model.count_parameters()
    print(f"Model: {params:,} parameters, dtype={dtype}")

    # Ensure checkpoint dir exists
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Stage 1: Pretraining
    if args.stage in ("all", "pretrain"):
        resume_path = None
        if args.resume and (args.checkpoint_dir / "stage1_pretrain_latest").exists():
            resume_path = str(args.checkpoint_dir / "stage1_pretrain_latest")
        results["pretrain"] = run_stage_pretrain(
            model,
            args.packed_dir,
            args.checkpoint_dir,
            steps=args.pretrain_steps if not args.smoke else 6,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr_pretrain,
            seq_len=args.seq_len,
            resume_from=resume_path,
        )

    # Stage 2: SFT
    if args.stage in ("all", "sft"):
        resume_path = None
        if args.resume and (args.checkpoint_dir / "stage2_sft_latest").exists():
            resume_path = str(args.checkpoint_dir / "stage2_sft_latest")
        results["sft"] = run_stage_sft(
            model,
            args.packed_dir,
            args.checkpoint_dir,
            steps=args.sft_steps if not args.smoke else 6,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr_sft,
            resume_from=resume_path,
        )

    # Save final checkpoint
    save_final(model, args.checkpoint_dir)

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for stage_name, res in results.items():
        print(
            f"  {stage_name}: start_loss={res['start_loss']}, end_loss={res['end_loss']:.4f}, steps={res['steps']}"
        )

    return results


if __name__ == "__main__":
    main()
