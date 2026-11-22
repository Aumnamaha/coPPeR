#!/usr/bin/env python3
r"""One-command dataset preparation orchestrator for coPPeR.

Runs the full data pipeline sequentially with resume support and telemetry.

Usage:
    # Full pipeline (default)
    python scripts/prepare_dataset.py --doc-count 300 --max-distill-docs 300

    # Skip generation step if docs already exist
    python scripts/prepare_dataset.py --skip-generate

    # Skip distillation step
    python scripts/prepare_dataset.py --skip-distill --seq-len 2048
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(label: str, cmd: list[str], skip_if_exists: Path | None = None) -> bool:
    """Run a pipeline step. Return True on success."""
    if skip_if_exists is not None and skip_if_exists.exists():
        print(f"  [SKIP] {label} (output already exists)")
        return True

    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=False,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    if result.returncode != 0:
        print(f"\nERROR: {label} failed with exit code {result.returncode}")
        return False

    print(f"\\n  [OK] {label} completed successfully")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="coPPeR dataset preparation orchestrator"
    )
    parser.add_argument("--doc-count", type=int, default=300)
    parser.add_argument("--max-distill-docs", type=int, default=300)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--distill-workers", type=int, default=4)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-distill", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"

    print("=" * 60)
    print("coPPeR Dataset Preparation Orchestrator")
    print("=" * 60)
    print(f"  doc-count:         {args.doc_count}")
    print(f"  max-distill-docs:  {args.max_distill_docs}")
    print(f"  seq-len:           {args.seq_len}")
    print(f"  distill-workers:   {args.distill_workers}")
    print(f"  skip-generate:    {args.skip_generate}")
    print(f"  skip-distill:     {args.skip_distill}")

    python_exe = sys.executable
    all_ok = True

    # Step 1: Generate synthetic documents
    if not args.skip_generate:
        cmd = [
            python_exe,
            str(scripts_dir / "generate_synthetic_docs.py"),
            "--count",
            str(args.doc_count),
            "--output",
            "data/synthetic_docs/docs.jsonl",
        ]
        all_ok &= run_step(
            "Generate synthetic documents", cmd, Path("data/synthetic_docs/docs.jsonl")
        )

    # Step 2: Distill teacher labels
    if not args.skip_distill:
        cmd = [
            python_exe,
            str(scripts_dir / "distill_teacher.py"),
            "--input",
            "data/synthetic_docs/docs.jsonl",
            "--max-docs",
            str(args.max_distill_docs),
            "--train-out",
            "data/distill/train.jsonl",
            "--eval-out",
            "data/distill/eval.jsonl",
            "--workers",
            str(args.distill_workers),
        ]
        all_ok &= run_step("Distill teacher labels (agy)", cmd)

    # Step 3: Train tokenizer
    cmd = [
        python_exe,
        str(scripts_dir / "train_tokenizer.py"),
        "--inputs",
        "data/synthetic_docs/docs.jsonl",
        "data/distill/train.jsonl",
        "--out-dir",
        "data/tokenizer",
        "--vocab-size",
        "32768",
    ]
    all_ok &= run_step("Train domain BPE tokenizer (32k vocab)", cmd)

    # Step 4: Pack datasets
    cmd = [
        python_exe,
        str(scripts_dir / "pack_dataset.py"),
        "--tokenizer",
        "data/tokenizer/copper_tokenizer.json",
        "--docs-jsonl",
        "data/synthetic_docs/docs.jsonl",
        "--sft-train-jsonl",
        "data/distill/train.jsonl",
        "--sft-eval-jsonl",
        "data/distill/eval.jsonl",
        "--out-dir",
        "data/packed",
        "--seq-len",
        str(args.seq_len),
    ]
    all_ok &= run_step("Pack binary datasets (pretrain + SFT)", cmd)

    # Final banner
    print("\n" + "=" * 60)
    if all_ok:
        print("SUCCESS: Dataset preparation complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. Unload Qwen3.6 from LM Studio")
        print("  2. Run training:")
        print("     python scripts/train_copper.py --checkpoint-dir checkpoints/")
        print("  3. Or run smoke test (works with Qwen3.6 loaded):")
        print(
            "     python scripts/train_copper.py --smoke --checkpoint-dir checkpoints/smoke_test"
        )
    else:
        print("ERROR: One or more steps failed.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
