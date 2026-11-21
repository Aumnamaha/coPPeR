"""Tests for training loop, VRAM guard, editor schemas, and tokenizer special tokens."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

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


class TestCUDA:
    def test_cuda_available(self):
        assert torch.cuda.is_available()
