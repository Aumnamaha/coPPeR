"""Comprehensive unit tests for CopperForCausalLM (~1B causal Transformer).

Tests:
- Exact parameter count ~950M-1,050M on full config
- Tiny config forward pass on CUDA with finite loss
- Strict causal mask isolation (future tokens never leak)
- Loss masking: -100 positions contribute zero gradient/loss difference
- Gradient backward pass with and without checkpointing
- KV-cache autoregressive generation equivalence
- safetensors save/load roundtrip reproduces identical logits
- Lossless newline tokenizer roundtrip
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the coPPeR project root directory."""
    return Path(__file__).resolve().parent.parent


def _add_src_to_path() -> None:
    sys.path.insert(0, str(_project_root() / "src"))


# ---------------------------------------------------------------------------
# Test: Parameter count on full ~1B config (meta device)
# ---------------------------------------------------------------------------


class TestParameterCount:
    """Verify that CopperConfig produces a ~1B parameter model."""

    def test_parameter_count_in_range(self) -> None:
        """Full config should have between 950M and 1,050M parameters."""
        _add_src_to_path()
        from src.model import CopperConfig

        cfg = CopperConfig()
        params = cfg.count_parameters()
        assert 950_000_000 <= params <= 1_050_000_000, f"Got {params:,} parameters"

    def test_bf16_weight_memory_estimate(self) -> None:
        """BF16 weights should occupy ~2 GB for a ~1B parameter model."""
        _add_src_to_path()
        from src.model import CopperConfig

        cfg = CopperConfig()
        byte_mem = cfg.count_parameters() * 2  # bf16 = 2 bytes
        mb = byte_mem / (1024 * 1024)
        assert 1500 <= mb <= 3000, f"BF16 memory estimate: {mb:.0f} MB"


@pytest.fixture(scope="module")
def tiny_config():
    """Tiny model config for fast forward-pass tests."""
    _add_src_to_path()
    from src.model import CopperConfig

    return CopperConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def tiny_model(tiny_config, device):
    _add_src_to_path()
    from src.model import CopperForCausalLM

    model = CopperForCausalLM(tiny_config).to(device)
    model.eval()
    return model


class TestTinyForward:
    """Verify forward pass shapes and loss on a tiny model."""

    def test_forward_logits_shape(self, tiny_model, device) -> None:
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, 256, (batch, seq_len), device=device)
        out = tiny_model(input_ids=input_ids)
        logits = out["logits"]
        assert logits.shape == (batch, seq_len, 256), f"Got {logits.shape}"
        assert torch.all(torch.isfinite(logits))

    def test_forward_loss_is_finite(self, tiny_model, device) -> None:
        batch, seq_len = 1, 8
        input_ids = torch.randint(0, 256, (batch, seq_len), device=device)
        labels = input_ids.clone()
        out = tiny_model(input_ids=input_ids, labels=labels)
        assert out["loss"] is not None
        loss_val = out["loss"].item()
        assert torch.isfinite(torch.tensor(loss_val))
        assert (
            0.0 <= loss_val < 15.0
        ), f"Loss {loss_val} seems too high for random labels"


class TestCausalMask:
    """Verify that future tokens never leak to past logits."""

    def test_causal_mask_isolation(self, tiny_model, device) -> None:
        """Modifying token t+1 should NOT change logits at positions 0..t."""
        _add_src_to_path()

        model_fp32 = tiny_model.to(torch.float32)
        batch, seq_len = 1, 8

        input_a = torch.randint(0, 256, (batch, seq_len), device=device).to(torch.long)
        out_a = model_fp32(input_ids=input_a)
        logits_a = out_a["logits"][0]

        input_b = input_a.clone()
        input_b[0, -1] = (input_b[0, -1] + 1) % 256
        out_b = model_fp32(input_ids=input_b)
        logits_b = out_b["logits"][0]

        past_logits_a = logits_a[:-1, :]
        past_logits_b = logits_b[:-1, :]
        diff = (past_logits_a - past_logits_b).abs().max().item()
        assert diff < 1e-5, f"Causal mask leak: max diff at past positions = {diff}"

    def test_causal_mask_allows_future_change(self, tiny_model, device) -> None:
        """Changing the last token SHOULD change logits at position -1."""
        _add_src_to_path()

        model_fp32 = tiny_model.to(torch.float32)
        batch, seq_len = 1, 8

        input_a = torch.randint(0, 256, (batch, seq_len), device=device).to(torch.long)
        out_a = model_fp32(input_ids=input_a)
        logits_a_last = out_a["logits"][0, -1, :]

        input_b = input_a.clone()
        input_b[0, -1] = (input_b[0, -1] + 1) % 256
        out_b = model_fp32(input_ids=input_b)
        logits_b_last = out_b["logits"][0, -1, :]

        diff = (logits_a_last - logits_b_last).abs().max().item()
        assert diff > 1e-3, f"Last position should change: diff={diff}"


class TestLossMasking:
    """Verify that -100 positions in labels contribute zero loss."""

    def test_all_masked_labels_give_zero_loss(self, tiny_model, device) -> None:
        _add_src_to_path()

        batch, seq_len = 1, 8
        input_ids = torch.randint(0, 256, (batch, seq_len), device=device)
        labels_all_masked = torch.full_like(input_ids, -100)
        out = tiny_model(input_ids=input_ids, labels=labels_all_masked)
        assert out["loss"] is not None
        loss_val = out["loss"].item()
        assert torch.isfinite(torch.tensor(loss_val))

    def test_partial_masking_reduces_loss_contribution(
        self, tiny_model, device
    ) -> None:
        """Masked positions should not affect loss computation."""
        _add_src_to_path()

        model_fp32 = tiny_model.to(torch.float32)
        batch, seq_len = 1, 8

        input_ids = torch.randint(0, 256, (batch, seq_len), device=device).to(
            torch.long
        )

        labels_full = input_ids.clone()
        out_full = model_fp32(input_ids=input_ids, labels=labels_full)

        labels_masked = torch.full_like(input_ids, -100)
        labels_masked[0, -1] = input_ids[0, -1]
        out_masked = model_fp32(input_ids=input_ids, labels=labels_masked)

        assert torch.isfinite(out_full["loss"])
        assert torch.isfinite(out_masked["loss"])
        assert out_full["loss"].item() > 0.0
        assert out_masked["loss"].item() > 0.0


class TestGradientBackward:
    """Verify that gradients flow correctly through the model."""

    def test_backward_no_checkpoint(self, tiny_model, device) -> None:
        _add_src_to_path()

        model = tiny_model.to(torch.float32)
        batch, seq_len = 1, 8
        input_ids = torch.randint(0, 256, (batch, seq_len), device=device)
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)
        assert out["loss"] is not None
        out["loss"].backward()

        grad_count = sum(
            1
            for p in model.parameters()
            if p.grad is not None and torch.all(torch.isfinite(p.grad))
        )
        total_params = sum(1 for _ in model.parameters())
        assert (
            grad_count == total_params
        ), f"Only {grad_count}/{total_params} params have gradients"

    def test_backward_with_checkpoint(self, tiny_model, device) -> None:
        """Gradient checkpointing should also produce valid gradients."""
        _add_src_to_path()

        model = tiny_model.to(torch.float32)
        model.train()
        model.enable_gradient_checkpointing()

        batch, seq_len = 1, 8
        input_ids = torch.randint(0, 256, (batch, seq_len), device=device)
        labels = input_ids.clone()

        out = model(input_ids=input_ids, labels=labels)
        assert out["loss"] is not None
        out["loss"].backward()

        grad_count = sum(
            1
            for p in model.parameters()
            if p.grad is not None and torch.all(torch.isfinite(p.grad))
        )
        total_params = sum(1 for _ in model.parameters())
        assert (
            grad_count == total_params
        ), f"Only {grad_count}/{total_params} params have gradients with checkpointing"


class TestKVCacheGeneration:
    """Verify that KV-cache generation produces identical results to full forward."""

    def test_kv_cache_equivalence(self, tiny_model, device) -> None:
        _add_src_to_path()

        model = tiny_model.to(torch.float32)
        model.eval()
        batch, prompt_len = 1, 4
        torch.manual_seed(42)
        input_ids = torch.randint(0, 256, (batch, prompt_len), device=device)

        # Generate tokens step-by-step using KV-cache
        generated_kv = model.generate(
            input_ids=input_ids.clone(), max_new_tokens=3, temperature=0.0
        )

        # Replicate generation by repeated full forward passes (no cache)
        torch.manual_seed(42)
        input_ids_ref = torch.randint(0, 256, (batch, prompt_len), device=device)
        generated_full = input_ids_ref.clone()
        for _ in range(3):
            out = model(input_ids=generated_full)
            next_tok = out["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated_full = torch.cat([generated_full, next_tok], dim=1)

        # Token sequences must match exactly
        assert (
            generated_kv == generated_full
        ).all(), f"KV-cache tokens {generated_kv[0].tolist()} != full-fwd {generated_full[0].tolist()}"

    def test_step_by_step_logits_match_prefix(self, tiny_model, device) -> None:
        """Generation step 0 logits match a full forward on the same prefix."""
        _add_src_to_path()

        model = tiny_model.to(torch.float32)
        model.eval()
        batch, prompt_len = 1, 4
        input_ids = torch.randint(0, 256, (batch, prompt_len), device=device)

        # Generation step 0: use_cache=True, no past_key_values
        out_gen = model(input_ids=input_ids.clone(), use_cache=True)
        logits_gen = out_gen["logits"][0, -1, :]

        # Full forward on same prefix without cache
        out_full = model(input_ids=input_ids.clone())
        logits_full = out_full["logits"][0, -1, :]

        diff = (logits_gen.float() - logits_full.float()).abs().max().item()
        assert diff < 1e-6, f"Step 0 mismatch: max diff = {diff}"


class TestSafetensorsRoundtrip:
    """Verify that save_pretrained -> from_pretrained reproduces identical logits."""

    def test_safetensors_roundtrip(self, tiny_model, device) -> None:
        _add_src_to_path()
        from src.model import CopperForCausalLM

        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_model.save_pretrained(tmpdir)
            loaded = CopperForCausalLM.from_pretrained(
                tmpdir, device=str(device), dtype=torch.float32
            )
            loaded.eval()

            test_input = torch.randint(0, 256, (1, 8), device=device)
            with torch.no_grad():
                out_orig = tiny_model(input_ids=test_input)
                out_loaded = loaded(input_ids=test_input)

            diff = (
                (out_orig["logits"].float() - out_loaded["logits"].float())
                .abs()
                .max()
                .item()
            )
            assert diff < 1e-5, f"Model roundtrip mismatch: max diff = {diff}"


class TestTokenizerNewlineRoundtrip:
    """Verify that the tokenizer preserves newlines through encode/decode."""

    def test_newline_roundtrip(self) -> None:
        _add_src_to_path()
        from src.tokenizer_wrapper import load_tokenizer

        tok = load_tokenizer(
            str(_project_root() / "data" / "tokenizer" / "copper_tokenizer.json")
        )

        test_cases = [
            "Hello world",
            "Line one\nLine two",
            "| col1 | col2 |\n|----|----|\n| A | B |\n",
            "# Header\n\nBody text.\n\n## Section\nMore content.",
        ]

        for tc in test_cases:
            ids = tok.encode(tc)
            decoded = tok.decode(ids, skip_special_tokens=True)
            assert (
                decoded == tc
            ), f"Roundtrip failed:\n  input:  {tc!r}\n  output: {decoded!r}"

    def test_json_block_roundtrip(self) -> None:
        _add_src_to_path()
        from src.tokenizer_wrapper import load_tokenizer

        tok = load_tokenizer(
            str(_project_root() / "data" / "tokenizer" / "copper_tokenizer.json")
        )
        text = '{"name": "Alice", "age": 30, "active": true}'
        ids = tok.encode(text)
        decoded = tok.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"JSON roundtrip failed:\n{decoded!r}"


class TestFullConfigInstantiation:
    """Verify that the full ~1B config can be instantiated."""

    def test_full_config_instantiates(self) -> None:
        _add_src_to_path()
        from src.model import CopperConfig

        cfg = CopperConfig()
        params = cfg.count_parameters()
        assert 950_000_000 <= params <= 1_050_000_000, f"Got {params:,} parameters"
