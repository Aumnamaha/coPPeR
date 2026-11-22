"""Distill teacher labels from agy.exe for coPPeR training data."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
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
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        stdout = (result.stdout or "").strip()
        if result.returncode != 0 and not stdout:
            # Fallback: try without --output-format json (text mode)
            cmd2 = [agy_path, AGY_PRINT_FLAG, prompt_text]
            result = subprocess.run(
                cmd2,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            stdout = (result.stdout or "").strip()
        return stdout, result.returncode == 0 or bool(stdout)
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel agy.exe workers (default: 4)",
    )
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

    total_tasks = len(docs) * len(TASK_TEMPLATES)
    print(
        f"Loaded {len(docs)} documents. Distilling teacher labels with {args.workers} workers...",
        flush=True,
    )
    print(f"Total (doc x task) pairs: {total_tasks}", flush=True)

    # Shared state protected by lock for parallel workers
    _seen_hashes = set()
    _records_lock = threading.Lock()
    _accepted_count = 0
    _rejected_count = 0
    _train_records_local = []
    _eval_records_local = []

    def _process_one(doc_idx, total_docs, doc, task_name):
        """Process a single (doc, task) pair. Returns status string."""
        nonlocal _accepted_count, _rejected_count
        doc_id = doc.get("doc_id", f"unknown_{doc_idx}")
        content = doc.get("content", "")
        sys_prompt = doc.get("system_prompt", "You are a gold-standard teacher agent.")

        if not content:
            return "skip"

        # Build user prompt with doc context
        if task_name == "json_extraction":
            fields = "title,date,amount,person"
            template_str = TASK_TEMPLATES[task_name]
            user_prompt = (
                f"{template_str.format(fields=fields)}\n\n--- DOCUMENT ---\n{content}"
            )
        elif task_name == "targeted_section_edit":
            section = "Introduction"
            instructions = "Improve clarity and fix any typos."
            template_str = TASK_TEMPLATES[task_name]
            user_prompt = f"{template_str.format(section=section, instructions=instructions)}\n\n--- DOCUMENT ---\n{content}"
        else:
            user_prompt = f"{TASK_TEMPLATES[task_name]}\n\n--- DOCUMENT ---\n{content}"

        # Run agy.exe
        output_text, success = run_agy(user_prompt, timeout=args.timeout)

        if not success or not output_text:
            with _records_lock:
                _rejected_count += 1
            return "reject"

        # Validate output
        is_valid, reason = validate_teacher_output(task_name, output_text)
        if not is_valid:
            with _records_lock:
                _rejected_count += 1
            print(f"  REJECTED [{doc_id}/{task_name}]: {reason}", file=sys.stderr)
            return "reject"

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

        with _records_lock:
            if rec_hash in _seen_hashes:
                return "dup"
            _seen_hashes.add(rec_hash)

            # Deterministic split by doc_id hash
            split_key = int(content_hash(doc_id), 16) % 100
            target_set = "eval" if split_key < (args.eval_ratio * 100) else "train"

            rec["meta"]["split"] = target_set
            record_line = json.dumps(rec, ensure_ascii=False)

            _accepted_count += 1
            if target_set == "eval":
                _eval_records_local.append(record_line)
            else:
                _train_records_local.append(record_line)
        return "accept"

    # Build all (doc_idx, doc, task_name) tuples for empty-doc filtering
    tasks = []
    for i, doc in enumerate(docs):
        if doc.get("content", ""):
            for task_name in TASK_TEMPLATES:
                tasks.append((i, len(docs), doc, task_name))

    # Process with ThreadPoolExecutor (I/O-bound: subprocess calls)
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one, *t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"Worker error: {exc}", file=sys.stderr)

            processed += 1
            # Progress report every doc's worth of tasks or at completion
            if processed % max(1, len(docs)) == 0 or processed == total_tasks:
                with _records_lock:
                    print(
                        f"Progress: {processed}/{total_tasks} tasks done "
                        f"(accepted={_accepted_count}, rejected={_rejected_count})",
                        flush=True,
                    )

    # Collect results under lock
    with _records_lock:
        train_records = list(_train_records_local)
        eval_records = list(_eval_records_local)
        accepted = _accepted_count
        rejected = _rejected_count

    # Write outputs
    for path, records in [
        (args.train_out, train_records),
        (args.eval_out, eval_records),
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record_line in records:
                f.write(record_line + "\n")
                f.flush()

    print(f"\nDone. Accepted={accepted} Rejected={rejected}")
    print(f"Train records: {len(train_records)} -> {args.train_out}")
    print(f"Evaluation records: {len(eval_records)} -> {args.eval_out}")


if __name__ == "__main__":
    main()
