#!/usr/bin/env python3
r"""Evaluate a trained coPPeR checkpoint on perplexity, loss, and task accuracy.

Usage:
    # Full evaluation (default)
    python scripts/evaluate_copper.py --checkpoint checkpoints/copper_1b_final

    # Evaluate with custom packed data directory
    python scripts/evaluate_copper.py --packed-dir data/packed --out-report exports/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_schemas import validate_teacher_output
from model import CopperForCausalLM


def load_bin_array(path: Path, dtype=np.uint16) -> np.ndarray:
    """Load a .bin file as a numpy array."""
    return np.fromfile(str(path), dtype=dtype)


class PretrainDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that yields fixed-length token chunks from packed bin."""

    def __init__(self, data_path: Path, seq_len: int = 2048) -> None:
        self.tokens = load_bin_array(data_path, np.uint16)
        self.seq_len = seq_len
        self.n_seqs = len(self.tokens) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int):
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
        self.n_records = len(self.inputs) // 2048

    def __len__(self) -> int:
        return self.n_records

    def __getitem__(self, idx: int):
        offset = idx * 2048
        x = torch.from_numpy(self.inputs[offset : offset + 2048]).long()
        y = torch.from_numpy(self.labels[offset : offset + 2048]).long()
        return x, y


def compute_perplexity(model, data_path: Path, seq_len: int = 2048) -> float | None:
    """Compute validation perplexity on packed pretrain eval bin."""
    if not data_path.exists():
        return None

    dataset = PretrainDataset(data_path, seq_len=seq_len)
    if len(dataset) == 0:
        return None

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(input_ids=x)["logits"]
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = y[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100).item()
            valid = (shift_labels != -100).sum().item()
            total_loss += loss * valid
            total_tokens += valid

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"  Pretrain eval perplexity: {perplexity:.4f} (loss={avg_loss:.4f})")
    return perplexity


def compute_sft_loss(model, inputs_path: Path, labels_path: Path) -> float | None:
    """Compute validation SFT cross-entropy loss."""
    if not inputs_path.exists() or not labels_path.exists():
        return None

    dataset = SFTDataset(inputs_path, labels_path)
    if len(dataset) == 0:
        return None

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(input_ids=x)["logits"]
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = y[:, 1:].reshape(-1).to(shift_logits.device)
            valid_mask = shift_labels != -100

            if not valid_mask.any():
                continue

            loss = F.cross_entropy(
                shift_logits[valid_mask], shift_labels[valid_mask], ignore_index=-100
            ).item()
            total_loss += loss * valid_mask.sum().item()
            total_tokens += valid_mask.sum().item()

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    print(f"  SFT eval cross-entropy loss: {avg_loss:.4f}")
    return avg_loss


def run_task_probe(model, tokenizer_cfg_path: Path) -> dict[str, bool]:
    """Run deterministic task probes using model.generate()."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_cfg_path))
    device = next(model.parameters()).device
    model.eval()

    eos_id = 3
    if hasattr(tok, "token_to_id"):
        try:
            eos_id = tok.token_to_id("<|eos|>")
        except (KeyError, AttributeError):
            pass

    # Build a short document for probing
    sample_doc = (
        "# Invoice #INV-2024-0341\n"
        "Date: 2024-03-15\n"
        "Bill To: Acme Corporation, 123 Main St\n"
        "| Item | Qty | Price |\n"
        "|------|-----|-------|\n"
        "| Widget A | 5 | $12.00 |\n"
    )

    probes = {}

    # Probe 1: JSON extraction
    input_ids_1 = tok.encode(sample_doc[:200], add_special_tokens=True).ids
    input_tensor_1 = torch.tensor([input_ids_1], device=device, dtype=torch.long)

    with torch.no_grad():
        out_1 = model.generate(
            input_tensor_1, max_new_tokens=96, temperature=0.0, eos_token_id=eos_id
        )
    generated_1 = tok.decode(
        out_1[0].tolist()[len(input_ids_1) :], skip_special_tokens=True
    )
    ok1, _ = validate_teacher_output("json_extraction", generated_1.strip())
    probes["json_extraction"] = bool(ok1)

    # Probe 2: cursor_span_edit
    doc_with_sel = (
        f"{sample_doc[:150]}\n<|sel_start|>Widget A<|sel_end|> needs reordering."
    )
    input_ids_2 = tok.encode(doc_with_sel[:250], add_special_tokens=True).ids
    input_tensor_2 = torch.tensor([input_ids_2], device=device, dtype=torch.long)

    with torch.no_grad():
        out_2 = model.generate(
            input_tensor_2, max_new_tokens=96, temperature=0.0, eos_token_id=eos_id
        )
    generated_2 = tok.decode(
        out_2[0].tolist()[len(input_ids_2) :], skip_special_tokens=True
    )
    ok2, _ = validate_teacher_output("cursor_span_edit", generated_2.strip())
    probes["cursor_span_edit"] = bool(ok2)

    # Probe 3: editor_tool_call
    prompt3 = (
        "Given the selection <|sel_start|>Widget A<|sel_end|>, output a JSON tool call "
        'to replace it with "Widget Alpha" using the replace_selection tool.'
    )
    input_ids_3 = tok.encode(prompt3[:250], add_special_tokens=True).ids
    input_tensor_3 = torch.tensor([input_ids_3], device=device, dtype=torch.long)

    with torch.no_grad():
        out_3 = model.generate(
            input_tensor_3, max_new_tokens=96, temperature=0.0, eos_token_id=eos_id
        )
    generated_3 = tok.decode(
        out_3[0].tolist()[len(input_ids_3) :], skip_special_tokens=True
    )
    ok3, _ = validate_teacher_output("editor_tool_call", generated_3.strip())
    probes["editor_tool_call"] = bool(ok3)

    # Probe 4: grounded Q&A with NOT_FOUND
    prompt4 = f"Document:\n{sample_doc}\nQuestion: What is the shipping address for this order?"
    input_ids_4 = tok.encode(prompt4[:250], add_special_tokens=True).ids
    input_tensor_4 = torch.tensor([input_ids_4], device=device, dtype=torch.long)

    with torch.no_grad():
        out_4 = model.generate(
            input_tensor_4, max_new_tokens=96, temperature=0.0, eos_token_id=eos_id
        )
    generated_4 = tok.decode(
        out_4[0].tolist()[len(input_ids_4) :], skip_special_tokens=True
    ).strip()
    ok4, _ = validate_teacher_output("grounded_qa_with_not_found", generated_4)
    probes["grounded_qa_not_found"] = bool(ok4)

    return probes


def main() -> None:
    parser = argparse.ArgumentParser(description="coPPeR model evaluation")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/copper_1b_final")
    )
    parser.add_argument("--packed-dir", type=Path, default=Path("data/packed"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/tokenizer/copper_tokenizer.json")
    )
    parser.add_argument("--out-report", type=str, default="exports/eval_report.json")
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 60)
    print("coPPeR Evaluation Report")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {device}")
    print("=" * 60)

    # Load model
    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint}")
        sys.exit(1)

    model = CopperForCausalLM.from_pretrained(
        args.checkpoint, device=str(device), dtype=dtype
    )
    params = model.count_parameters()
    print(f"Model parameters: {params:,}")

    # 1. Perplexity on pretrain eval
    perplexity = compute_perplexity(
        model, args.packed_dir / "pretrain_eval.bin", seq_len=args.seq_len
    )

    # 2. SFT loss on sft eval
    sft_loss = compute_sft_loss(
        model,
        args.packed_dir / "sft_eval_inputs.bin",
        args.packed_dir / "sft_eval_labels.bin",
    )

    # 3. Task probes
    if args.tokenizer.exists():
        try:
            print("\nTask probe results:")
            probes = run_task_probe(model, args.tokenizer)
            for task_name, passed in probes.items():
                status = (
                    "PASS" if passed else "FAIL (model not yet trained on this task)"
                )
                print(f"  {task_name}: {status}")
        except (RuntimeError, OSError) as exc:
            print(f"\nWarning: Task probes failed ({exc}), skipping")
            probes = {}
    else:
        print(
            f"\nWarning: Tokenizer not found at {args.tokenizer}, skipping task probes"
        )
        probes = {}

    # Save report
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "model_parameters": params,
        "pretrain_eval_perplexity": perplexity,
        "sft_eval_loss": sft_loss,
        "task_probes": probes,
    }

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to {out_path}")
    return report


if __name__ == "__main__":
    main()
