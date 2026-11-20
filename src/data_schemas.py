"""Strict JSON/ChatML validation, deduplication, and quality filters for coPPeR data pipeline."""

import hashlib
import json

DOC_CATEGORIES = [
    "invoice_receipt",
    "legal_contract",
    "technical_spec",
    "executive_memo",
    "financial_table_report",
    "academic_abstract_notes",
    "sop_policy_doc",
    "messy_meeting_notes",
]

TASK_TEMPLATES = {
    "json_extraction": (
        "Extract the following fields as JSON from the document below. "
        'Return ONLY valid JSON with keys: "{fields}". '
        "Use null for any field not present in the document."
    ),
    "executive_rewrite": (
        "Rewrite the document into a concise executive summary (max 300 words). "
        "Output only the revised Markdown text -- no preamble."
    ),
    "markdown_table_refactor": (
        "Refactor any tables in this document to proper Markdown table format. "
        'Use pipe separators: "| col1 | col2 |". '
        "Output only the full revised Markdown."
    ),
    "grounded_qa_with_not_found": (
        "Answer the question strictly from the given document. "
        "If the answer is not present, output exactly: NOT_FOUND"
    ),
    "targeted_section_edit": (
        "Edit only the section titled '{section}' according to these instructions: {instructions}. "
        "Output a unified diff or the exact revised Markdown text."
    ),
}

_FILLER_PREFIXES = (
    "sure",
    "here is",
    "certainly",
    "of course",
    "here's",
    "ok,",
)


def validate_teacher_output(task_name: str, output_text: str) -> tuple:
    """Validate teacher agent output for a given task.

    Returns (is_valid: bool, reason: str).
    """
    if not output_text or not output_text.strip():
        return False, "empty output"

    stripped = output_text.strip()

    # Reject conversational filler prefixes
    for prefix in _FILLER_PREFIXES:
        if stripped.lower().startswith(prefix):
            return False, f'conversational filler: "{stripped[:len(prefix)]}"'

    # Task-specific validation
    if task_name == "json_extraction":
        try:
            parsed = json.loads(stripped)
            assert isinstance(parsed, (dict, list)), "not dict/list"
            return True, "valid JSON"
        except (json.JSONDecodeError, AssertionError) as exc:
            return False, f"invalid JSON: {exc}"

    if task_name == "markdown_table_refactor":
        # Must contain at least one pipe character for table formatting
        if "|" not in stripped and "---" not in stripped:
            return False, "no table markers found (| or ---)"
        return True, "has table markers"

    if task_name == "grounded_qa_with_not_found":
        # NOT_FOUND is a valid answer; otherwise any non-empty text > 1 char is OK
        if stripped == "NOT_FOUND":
            return True, "NOT_FOUND returned"
        if len(stripped) < 2:
            return False, "answer too short"
        return True, "valid grounded answer"

    # Default: accept non-empty, non-filler output
    return True, f"accepted ({task_name})"


def format_chatml_record(
    system_prompt: str,
    user_prompt: str,
    assistant_output: str,
    meta: dict | None = None,
) -> dict:
    """Format a single ChatML training record."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ],
        "meta": meta or {},
    }


def content_hash(text: str) -> str:
    """Return SHA-256 hex digest of text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
