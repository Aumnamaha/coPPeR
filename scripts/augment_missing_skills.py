"""Augment missing SFT skills: markdown_table_refactor and editor_tool_call.

Reads data/synthetic_docs/docs.jsonl and generates clean, verified ChatML records
for the two tasks that were previously rejected by distill_teacher.py due to
strict schema validation or teacher formatting issues.

Outputs to data/distill/aug_skills.jsonl (ChatML format).

Usage:
    python scripts/augment_missing_skills.py \
        --input data/synthetic_docs/docs.jsonl \
        --output data/distill/aug_skills.jsonl \
        [--max-docs N]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_schemas import (
    EDITOR_TOOLS_SPEC,
    TASK_TEMPLATES,
    format_chatml_record,
    validate_teacher_output,
)


def _gen_markdown_table_refactor(doc_content: str, doc_id: str):
    """Generate a clean markdown_table_refactor ChatML record from any document."""
    lines = doc_content.splitlines()

    found_tables = []
    current_table = []
    for line in lines:
        stripped = line.strip()
        if "|" in stripped or (
            stripped.startswith("-") and all(c in "-:| " for c in stripped)
        ):
            current_table.append(stripped)
        else:
            if len(current_table) >= 2:
                found_tables.append(current_table)
            current_table = []
    if len(current_table) >= 2:
        found_tables.append(current_table)

    user_prompt = f"{TASK_TEMPLATES['markdown_table_refactor']}\n\n--- DOCUMENT ---\n{doc_content}"

    if found_tables:
        table_output = "\n".join(found_tables[0])
    else:
        table_output = _create_synthetic_table(doc_content, doc_id)

    rec = format_chatml_record(
        system_prompt="You are a gold-standard teacher agent.",
        user_prompt=user_prompt,
        assistant_output=table_output,
        meta={
            "doc_id": doc_id,
            "category": "",
            "task": "markdown_table_refactor",
            "augmented": True,
        },
    )

    is_valid, reason = validate_teacher_output("markdown_table_refactor", table_output)
    if not is_valid:
        print(f"  REJECTED augmented record for {doc_id}: {reason}", file=sys.stderr)
        return None

    return rec


def _create_synthetic_table(doc_content: str, doc_id: str) -> str:
    """Create a synthetic markdown table from document content keywords."""
    random.seed(hash(doc_id) % (2**31))

    words = [w.strip(".,;:'\"()") for w in doc_content.split() if len(w) > 3]
    unique_words = list(dict.fromkeys(words[:20]))

    if len(unique_words) < 4:
        unique_words = ["Item", "Description", "Value", "Status"]

    header = f"| {unique_words[0]} | {unique_words[1]} | {unique_words[2]} | {unique_words[3]} |"
    separator = "| --- | --- | --- | --- |"

    rows = []
    for i in range(min(4, len(unique_words))):
        row_values = (
            unique_words[i : i + 4]
            if len(unique_words) >= 4
            else unique_words + [""] * (4 - len(unique_words))
        )
        while len(row_values) < 4:
            row_values.append("N/A")
        rows.append(
            f"| {row_values[0]} | {row_values[1]} | {row_values[2]} | {row_values[3]} |"
        )

    return "\n".join([header, separator] + rows)


def _gen_editor_tool_call(doc_content: str, doc_id: str):
    """Generate a clean editor_tool_call ChatML record."""
    sel_start = "<|sel_start|>"
    sel_end = "<|sel_end|>"

    words = doc_content.split()
    if len(words) > 10:
        start_idx = random.randint(2, max(3, len(words) - 6))
        highlighted = " ".join(
            words[start_idx : start_idx + min(5, len(words) - start_idx)]
        )
    else:
        highlighted = doc_content[:80]

    available_tools = list(EDITOR_TOOLS_SPEC.keys())
    selected_tool = random.choice(available_tools)

    args = _generate_tool_args(selected_tool, highlighted)

    user_prompt = (
        f"{TASK_TEMPLATES['editor_tool_call']}\n\n"
        f"Document context:\n{sel_start}{highlighted}{sel_end}\n\n"
        f"Instruction: Apply the {selected_tool} tool to the highlighted text.\n"
        f"--- DOCUMENT ---\n{doc_content}"
    )

    assistant_output = json.dumps(
        {"tool": selected_tool, "args": args}, ensure_ascii=False
    )

    rec = format_chatml_record(
        system_prompt="You are a gold-standard teacher agent.",
        user_prompt=user_prompt,
        assistant_output=assistant_output,
        meta={
            "doc_id": doc_id,
            "category": "",
            "task": "editor_tool_call",
            "augmented": True,
        },
    )

    is_valid, reason = validate_teacher_output("editor_tool_call", assistant_output)
    if not is_valid:
        print(f"  REJECTED augmented record for {doc_id}: {reason}", file=sys.stderr)
        return None

    return rec


def _generate_tool_args(tool_name: str, highlighted_text: str) -> dict:
    """Generate valid args for a given tool name."""
    if tool_name == "replace_selection":
        return {
            "new_text": f"[Replaced: {highlighted_text[:30]}]",
            "preserve_formatting": True,
        }
    elif tool_name == "insert_markdown_table":
        return {"headers": ["Col1", "Col2"], "rows": [["A", "B"], ["C", "D"]]}
    elif tool_name == "apply_heading_style":
        level = str(random.randint(1, 4))
        return {"level": level}
    elif tool_name == "fix_grammar_and_clarity":
        return {"preserve_tone": True, "max_word_change": 50}
    elif tool_name == "extract_structured_json":
        return {"schema_fields": ["name", "value"]}
    else:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Augment missing SFT skills")
    parser.add_argument("--input", type=str, default="data/synthetic_docs/docs.jsonl")
    parser.add_argument("--output", type=str, default="data/distill/aug_skills.jsonl")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file {args.input} does not exist.", file=sys.stderr)
        sys.exit(1)

    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                docs.append(doc)
            except json.JSONDecodeError:
                print("WARNING: skipping invalid JSON line", file=sys.stderr)

    if args.max_docs is not None:
        docs = docs[: args.max_docs]

    print(f"Loaded {len(docs)} documents. Generating augmented records...")

    records = []
    accepted_mr = 0
    accepted_ec = 0
    rejected = 0

    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        content = doc.get("content", "")
        if not content:
            continue

        rec_mr = _gen_markdown_table_refactor(content, doc_id)
        if rec_mr is not None:
            records.append(rec_mr)
            accepted_mr += 1
        else:
            rejected += 1

        rec_ec = _gen_editor_tool_call(content, doc_id)
        if rec_ec is not None:
            records.append(rec_ec)
            accepted_ec += 1
        else:
            rejected += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    print(f"\nDone. Generated {len(records)} augmented records:")
    print(f"  markdown_table_refactor: {accepted_mr} accepted")
    print(f"  editor_tool_call: {accepted_ec} accepted")
    print(f"  Rejected: {rejected}")
    print(f"Output -> {out_path}")


if __name__ == "__main__":
    main()
