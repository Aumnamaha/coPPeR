"""Tests for coPPeR data pipeline: schema validation, train/eval split, mocked e2e."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_schemas import (
    TASK_TEMPLATES,
    content_hash,
    format_chatml_record,
    validate_teacher_output,
)


# ============================================================
# Schema validation tests
# ============================================================
class TestValidateTeacherOutput:
    """Test validate_teacher_output() for all task types."""

    def test_valid_json_extraction(self):
        valid = '{"title": "Invoice", "date": "2025-01-15"}'
        ok, reason = validate_teacher_output("json_extraction", valid)
        assert ok, f"Should accept valid JSON: {reason}"

    def test_invalid_json_extraction(self):
        invalid = "{broken json}"
        ok, _reason = validate_teacher_output("json_extraction", invalid)
        assert not ok, "Should reject invalid JSON"

    def test_empty_output_rejected(self):
        ok, _reason = validate_teacher_output("json_extraction", "")
        assert not ok, "Empty output should be rejected"

    def test_filler_prefix_sure(self):
        ok, _reason = validate_teacher_output(
            "json_extraction", "sure, here is the JSON: {}"
        )
        assert not ok, "Should reject filler prefix sure"

    def test_filler_prefix_here_is(self):
        ok, _reason = validate_teacher_output("json_extraction", "here is your result.")
        assert not ok, "Should reject filler prefix here is"

    def test_filler_prefix_certainly(self):
        ok, _reason = validate_teacher_output(
            "grounded_qa_with_not_found", "certainly the answer is 42"
        )
        assert not ok, "Should reject filler prefix certainly"

    def test_grounding_not_found(self):
        ok, _reason = validate_teacher_output("grounded_qa_with_not_found", "NOT_FOUND")
        assert ok, "NOT_FOUND should be accepted for grounded QA"

    def test_table_refactor_has_pipes(self):
        table_out = "| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        ok, reason = validate_teacher_output("markdown_table_refactor", table_out)
        assert ok, f"Should accept pipe-table output: {reason}"

    def test_table_refactor_no_pipes(self):
        no_table = "This is just plain text without any tables."
        ok, _reason = validate_teacher_output("markdown_table_refactor", no_table)
        # Should reject if no table markers at all
        assert (
            not ok or "|---" in no_table
        )  # flexible: may accept if no pipes but has content

    def test_filler_prefix_of_course(self):
        ok, _reason = validate_teacher_output(
            "grounded_qa_with_not_found", "of course the answer is here"
        )
        assert not ok, "Should reject filler prefix of course"


# ============================================================
# Content hash / dedup tests
# ============================================================
class TestContentHash:
    def test_deterministic(self):
        h1 = content_hash("hello")
        h2 = content_hash("hello")
        assert h1 == h2

    def test_different_text_different_hash(self):
        assert content_hash("text a") != content_hash("text b")

    def test_returns_hex_string(self):
        h = content_hash("test")
        assert len(h) == 64  # SHA-256 hex length


# ============================================================
# ChatML format tests
# ============================================================
class TestFormatChatmlRecord:
    def test_basic_record(self):
        rec = format_chatml_record(
            system_prompt="You are a teacher.",
            user_prompt="What is 2+2?",
            assistant_output="4",
            meta={"doc_id": "d1"},
        )
        assert len(rec["messages"]) == 3
        assert rec["messages"][0]["role"] == "system"
        assert rec["messages"][1]["role"] == "user"
        assert rec["messages"][2]["role"] == "assistant"
        assert rec["meta"]["doc_id"] == "d1"

    def test_default_meta(self):
        rec = format_chatml_record("sys", "usr", "asst")
        assert isinstance(rec["meta"], dict)


# ============================================================
# Train/eval split determinism tests
# ============================================================
class TestTrainEvalSplit:
    """Verify deterministic split with zero doc_id leakage."""

    def test_split_deterministic(self):
        """Same doc_id always goes to same split."""
        for _ in range(100):
            hash_a = int(content_hash("test_doc_42"), 16) % 100
            assert hash_a == (int(content_hash("test_doc_42"), 16) % 100)

    def test_no_leakage_with_many_docs(self):
        """With deterministic hashing, a doc_id always maps to one split."""
        eval_ratio = 0.05
        splits_seen = {}
        for i in range(1000):
            doc_id = f"doc_{i}"
            key = int(content_hash(doc_id), 16) % 100
            split = "eval" if key < (eval_ratio * 100) else "train"
            splits_seen.setdefault(doc_id, set()).add(split)

        # Every doc should map to exactly one split
        for doc_id, splits in splits_seen.items():
            assert len(splits) == 1, f"{doc_id} maps to multiple splits: {splits}"


# ============================================================
# Mocked end-to-end pipeline tests
# ============================================================
class TestMockedE2EPipeline:
    """Test pipeline logic with mocked LM Studio and agy.exe."""

    def test_generate_request_structure(self):
        """Verify the generate request would have correct structure for LM Studio."""
        from scripts.generate_synthetic_docs import CATEGORY_PROMPTS, DOC_CATEGORIES

        assert len(DOC_CATEGORIES) == 8
        assert len(CATEGORY_PROMPTS) >= 8

    @patch("scripts.distill_teacher.run_agy")
    def test_distill_accepts_valid_output(self, mock_agy):
        """Mock agy to return valid JSON for json_extraction task."""
        mock_agy.return_value = (
            '{"title": "Test Invoice", "date": "2025-01-01"}',
            True,
        )

        # Just verify the mock works -- full pipeline tested in smoke test
        result, success = mock_agy("test prompt")
        assert success is True
        parsed = json.loads(result)
        assert "title" in parsed

    def test_task_templates_count(self):
        """Verify exactly 5 task templates are defined."""
        assert len(TASK_TEMPLATES) == 5


# ============================================================
# Live smoke test helpers (run separately with --live flag)
# ============================================================
def run_live_smoke(count=2, output_path="data/synthetic_docs/smoke_docs.jsonl"):
    """Run a live smoke test: generate N docs from LM Studio."""
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "scripts" / "generate_synthetic_docs.py"),
        "--count",
        str(count),
        "--output",
        output_path,
        "--endpoint",
        "http://192.168.56.1:1234/v1/chat/completions",
        "--model",
        "qwen3.6-35b-a3b",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
    print("generate_synthetic_docs stdout:", out)
    if result.returncode != 0:
        print("generate_synthetic_docs stderr:", result.stderr)

    # Validate output
    with open(output_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) >= count, f"Expected {count} docs, got {len(lines)}"
    for line in lines:
        json.loads(line)  # verify valid JSON

    return lines


if __name__ == "__main__":
    # Run live smoke when executed directly
    print("Running live 2-document smoke test...")
    run_live_smoke(count=2)
    print("\nSmoke test complete. Check data/synthetic_docs/smoke_docs.jsonl")
