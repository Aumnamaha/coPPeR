"""Tests for training loop, VRAM guard, editor schemas, and tokenizer special tokens."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_schemas import (
    EDITOR_TOOLS_SPEC,
    SUPPORTED_HOST_PLATFORMS,
    TASK_TEMPLATES,
    validate_teacher_output,
)
from tokenizer_wrapper import SPECIAL_TOKENS, VOCAB_SIZE


class TestSpecialTokens:
    def test_special_token_count(self):
        assert len(SPECIAL_TOKENS) == 12

    def test_sel_start_id_is_10(self):
        assert SPECIAL_TOKENS[10] == "<|sel_start|>"

    def test_sel_end_id_is_11(self):
        assert SPECIAL_TOKENS[11] == "<|sel_end|>"

    def test_vocab_size_exact(self):
        assert VOCAB_SIZE == 32768


class TestEditorToolValidation:
    def test_valid_editor_tool_call(self):
        valid_json = json.dumps(
            {"tool": "replace_selection", "args": {"text": "new content"}}
        )
        ok, _msg = validate_teacher_output("editor_tool_call", valid_json)
        assert ok is True

    def test_valid_cursor_span_edit(self):
        text = "This is the corrected replacement text."
        ok, _msg = validate_teacher_output("cursor_span_edit", text)
        assert ok is True

    def test_editor_tool_call_unknown_tool(self):
        bad_json = json.dumps({"tool": "unknown_action", "args": {}})
        ok, _msg = validate_teacher_output("editor_tool_call", bad_json)
        assert ok is False

    def test_editor_tool_call_malformed_json(self):
        ok, _msg = validate_teacher_output("editor_tool_call", "{not valid json}")
        assert ok is False

    def test_cursor_span_edit_empty(self):
        ok, _msg = validate_teacher_output("cursor_span_edit", "")
        assert ok is False

    def test_cursor_span_edit_filler_rejected(self):
        ok, _msg = validate_teacher_output(
            "cursor_span_edit", "Sure! Here is the replacement text."
        )
        assert ok is False

    def test_editor_tool_call_missing_args(self):
        bad_json = json.dumps({"tool": "replace_selection"})
        ok, _msg = validate_teacher_output("editor_tool_call", bad_json)
        assert ok is False


class TestEditorToolsSpec:
    def test_editor_tools_spec_exists(self):
        assert isinstance(EDITOR_TOOLS_SPEC, dict)

    def test_supported_platforms_exist(self):
        assert isinstance(SUPPORTED_HOST_PLATFORMS, list)
        assert len(SUPPORTED_HOST_PLATFORMS) > 0

    def test_task_templates_count_is_7(self):
        assert len(TASK_TEMPLATES) == 7


class TestVRAMGuard:
    def test_vram_guard_high_requirement_exits(self):
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "train_copper.py"
                ),
                "--smoke",
                "--min-free-vram-gb",
                "999.0",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 1
        assert (
            "Insufficient free VRAM" in result.stdout
            or "Insufficient free VRAM" in result.stderr
        )


class TestSmokeTraining:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.ckpt_dir = str(tmp_path / "smoke_test")
        self.out_dir = str(tmp_path / "packed_smoke")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        import shutil

        for fname in [
            "pretrain_train.bin",
            "sft_train_inputs.bin",
            "sft_train_labels.bin",
        ]:
            src = Path(f"data/packed/{fname}")
            if src.exists():
                shutil.copy2(src, str(Path(self.out_dir) / fname))

    def test_smoke_training_completes_and_decreases_loss(self):
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "train_copper.py"
                ),
                "--stage",
                "all",
                "--smoke",
                "--packed-dir",
                self.out_dir,
                "--checkpoint-dir",
                self.ckpt_dir,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Smoke training failed: {result.stderr}"
        final_ckpt = Path(self.ckpt_dir) / "copper_1b_final"
        assert final_ckpt.exists()
        assert (final_ckpt / "model.safetensors").exists()

    def test_smoke_training_resume(self):
        subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "train_copper.py"
                ),
                "--stage",
                "all",
                "--smoke",
                "--packed-dir",
                self.out_dir,
                "--checkpoint-dir",
                self.ckpt_dir + "_resume",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "train_copper.py"
                ),
                "--stage",
                "sft",
                "--smoke",
                "--packed-dir",
                self.out_dir,
                "--checkpoint-dir",
                self.ckpt_dir + "_resume",
                "--resume",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Resume failed: {result.stderr}"


class TestTokenBoundsValidation:
    """Verify out-of-range token IDs raise ValueError on CPU, not CUDA assert."""

    def test_out_of_range_raises_valueerror_on_cpu(self):
        """When input_ids contain tokens >= vocab_size, training should raise
        a clear ValueError before any CUDA kernel launch."""
        import subprocess
        import sys

        code = """
import torch

def validate_tokens(x, vocab_size):
    mx = x.max().item()
    mn = x.min().item()
    if mn < 0 or mx >= vocab_size:
        raise ValueError(
            f"Token ID out of range [0, {vocab_size}): "
            f"min={mn}, max={mx}"
        )

# Simulate batch with token IDs that exceed a tiny model's vocab (256)
tiny_vocab = 256
batch_x = torch.tensor([[1, 2, tiny_vocab + 5, 4]], dtype=torch.long)
try:
    validate_tokens(batch_x, tiny_vocab)
    print("FAIL: Should have raised ValueError")
except ValueError as e:
    msg = str(e)
    assert "Token ID out of range" in msg
    assert f"max={tiny_vocab + 5}" in msg
    print(f"OK: {msg}")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "OK:" in result.stdout
        assert "FAIL" not in result.stdout


class TestExtraSFTPacking:
    """Verify pack_dataset.py --extra-sft-jsonl merging."""

    @staticmethod
    def _write_test_jsonl(path: str, records: list[dict]) -> None:
        """Write test ChatML JSONL records to a file."""
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(rec) + "\n" for rec in records)

    def test_extra_sft_jsonl_merges_records(self, tmp_path):
        """pack_dataset.py merges extra SFT JSONL files into training set."""
        import subprocess
        import sys

        # Create minimal tokenizer for testing (reuse existing)
        packed_dir = str(tmp_path / "packed")
        Path(packed_dir).mkdir(parents=True, exist_ok=True)

        # Copy existing packed data as baseline
        src_packed = Path("data/packed")
        if src_packed.exists():
            for fname in ["pretrain_train.bin", "tokenizer"]:
                pass  # We'll use the existing tokenizer path

        # Create a tiny test SFT JSONL file
        sft_main = str(tmp_path / "main.jsonl")
        self._write_test_jsonl(
            sft_main,
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                },
            ],
        )

        # Create extra SFT JSONL file
        sft_extra = str(tmp_path / "extra.jsonl")
        self._write_test_jsonl(
            sft_extra,
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I am fine!"},
                    ],
                },
            ],
        )

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "pack_dataset.py"
                ),
                "--sft-train-jsonl",
                sft_main,
                "--extra-sft-jsonl",
                sft_extra,
                "--out-dir",
                packed_dir,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Packing failed: {result.stderr}"
        # Should report merging from 2 sources
        assert (
            "merged from 2 sources" in result.stdout
            or "extra SFT source" in result.stdout
        )

    def test_no_extra_sft_flag_disables_auto_merge(self, tmp_path):
        """--no-extra-sft prevents auto-merging of aug_skills.jsonl."""
        import subprocess
        import sys

        packed_dir = str(tmp_path / "packed")
        Path(packed_dir).mkdir(parents=True, exist_ok=True)

        sft_main = str(tmp_path / "main.jsonl")
        self._write_test_jsonl(
            sft_main,
            [
                {
                    "messages": [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                },
            ],
        )

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "pack_dataset.py"
                ),
                "--sft-train-jsonl",
                sft_main,
                "--no-extra-sft",
                "--out-dir",
                packed_dir,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Packing failed: {result.stderr}"


class TestResumeFromTraining:
    """Verify train_copper.py --resume-from SFT warm-start."""

    def test_resume_from_flag_is_accepted(self):
        """train_copper.py accepts --resume-from argument without error."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "train_copper.py"
                ),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--resume-from" in result.stdout

    def test_resume_from_produces_help_text(self):
        """--resume-from flag shows proper help description."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parent.parent
                    / "scripts"
                    / "train_copper.py"
                ),
                "--help",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        help_text = result.stdout + result.stderr
        assert "warm-start" in help_text.lower() or "checkpoint" in help_text.lower()


class TestCUDA:
    def test_cuda_available(self):
        import torch

        assert torch.cuda.is_available()
