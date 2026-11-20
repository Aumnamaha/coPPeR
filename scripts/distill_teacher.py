"""Distill teacher labels from agy.exe for coPPeR training data."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_schemas import (
    TASK_TEMPLATES,
    content_hash,
    format_chatml_record,
    validate_teacher_output,
)

# agy.exe CLI: --print <prompt> or -p <prompt>, with optional --output-format json
AGY_PRINT_FLAG = "--print"


def run_agy(prompt_text, timeout=120):
    """Run agy.exe in non-interactive print mode and return stdout."""
    agy_path = os.environ.get(
        "AGY_PATH", r"C:\Users\Aum Namaha\AppData\Local\agy\bin\agy.exe"
    )
    cmd = [
        agy_path,
        AGY_PRINT_FLAG,
        prompt_text,
        "--output-format",
        "json",
    ]
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0 and not result.stdout.strip():
            # Fallback: try without --output-format json (text mode)
            cmd2 = [agy_path, AGY_PRINT_FLAG, prompt_text]
            result = subprocess.run(
                cmd2, check=False, capture_output=True, text=True, timeout=timeout
            )
        return result.stdout.strip(), result.returncode == 0 or bool(
            result.stdout.strip()
        )
    except subprocess.TimeoutExpired:
        return "", False
    except FileNotFoundError:
        print(
            f"ERROR: agy.exe not found at {agy_path}. Set AGY_PATH env or use --agy-path.",
            file=sys.stderr,
        )
        return "", False


def main():
    parser = argparse.ArgumentParser(description="Distill teacher labels via agy.exe")
    parser.add_argument("--input", type=str, default="data/synthetic_docs/docs.jsonl")
    parser.add_argument("--train-out", type=str, default="data/distill/train.jsonl")
    parser.add_argument("--eval-out", type=str, default="data/distill/eval.jsonl")
    parser.add_argument(
        "--agy-path",
        type=str,
        default=os.environ.get(
            "AGY_PATH", r"C:\Users\Aum Namaha\AppData\Local\agy\bin\agy.exe"
        ),
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    # Update agy path in env
    os.environ["AGY_PATH"] = args.agy_path

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file {args.input} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read documents
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

    print(f"Loaded {len(docs)} documents. Distilling teacher labels...")

    train_records = []
    eval_records = []
    accepted = 0
    rejected = 0
    seen_hashes = set()

    for i, doc in enumerate(docs):
        doc_id = doc.get("doc_id", f"unknown_{i}")
        content = doc.get("content", "")
        sys_prompt = doc.get("system_prompt", "You are a gold-standard teacher agent.")

        if not content:
            print(f"[{i+1}/{len(docs)}] {doc_id}: empty content, skipping")
            continue

        for task_name in TASK_TEMPLATES:
            # Build user prompt with doc context
            if task_name == "json_extraction":
                fields = "title,date,amount,person"
                template_str = TASK_TEMPLATES[task_name]
                user_prompt = f"{template_str.format(fields=fields)}\n\n--- DOCUMENT ---\n{content}"
            elif task_name == "targeted_section_edit":
                section = "Introduction"
                instructions = "Improve clarity and fix any typos."
                template_str = TASK_TEMPLATES[task_name]
                user_prompt = f"{template_str.format(section=section, instructions=instructions)}\n\n--- DOCUMENT ---\n{content}"
            else:
                user_prompt = (
                    f"{TASK_TEMPLATES[task_name]}\n\n--- DOCUMENT ---\n{content}"
                )

            # Run agy.exe
            output_text, success = run_agy(user_prompt, timeout=args.timeout)

            if not success or not output_text:
                rejected += 1
                continue

            # Validate output
            is_valid, reason = validate_teacher_output(task_name, output_text)
            if not is_valid:
                rejected += 1
                print(f"  REJECTED [{doc_id}/{task_name}]: {reason}", file=sys.stderr)
                continue

            accepted += 1

            # Deduplication by hash of the full record content
            rec = format_chatml_record(
                sys_prompt,
                user_prompt,
                output_text,
                {
                    "doc_id": doc_id,
                    "category": doc.get("category", ""),
                    "task": task_name,
                },
            )
            rec_hash = content_hash(json.dumps(rec["messages"], ensure_ascii=False))

            if rec_hash in seen_hashes:
                continue  # duplicate -- skip
            seen_hashes.add(rec_hash)

            # Deterministic split by doc_id hash
            split_key = int(content_hash(doc_id), 16) % 100
            target_set = "eval" if split_key < (args.eval_ratio * 100) else "train"

            rec["meta"]["split"] = target_set
            record_line = json.dumps(rec, ensure_ascii=False)

            if target_set == "eval":
                eval_records.append(record_line)
            else:
                train_records.append(record_line)

        print(
            f"[{i+1}/{len(docs)}] {doc_id}: accepted={accepted} rejected={rejected}",
            flush=True,
        )

    # Write outputs
    for path, records in [
        (args.train_out, train_records),
        (args.eval_out, eval_records),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(r + "\n" for r in records)

    print(f"\nDone. Accepted={accepted} Rejected={rejected}")
    print(f"Train records: {len(train_records)} -> {args.train_out}")
    print(f"Evaluation records: {len(eval_records)} -> {args.eval_out}")


if __name__ == "__main__":
    main()
