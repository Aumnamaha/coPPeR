"""Train a 32,768-vocab ByteLevel BPE tokenizer on JSONL document corpora.

Streams text from synthetic docs and distilled SFT data into a BPE model,
then pads vocabulary to exactly ``vocab_size`` with reserved tokens if needed.

Usage::

    python scripts/train_tokenizer.py \\
        --inputs data/synthetic_docs/docs.jsonl data/distill/train.jsonl \\
        --out-dir data/tokenizer \\
        --vocab-size 32768
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_texts(jsonl_path: str) -> list[str]:
    """Extract text content from a JSONL file (docs or SFT records)."""
    texts: list[str] = []
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Warning: {jsonl_path} does not exist, skipping.")
        return texts

    with open(path, encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping invalid JSON at {jsonl_path}:{line_num}")
                continue

            # Synthetic docs have a 'content' field with the document text
            if "content" in record and isinstance(record["content"], str):
                texts.append(record["content"])

            # SFT ChatML records have 'messages' - extract assistant content + user context
            elif "messages" in record and isinstance(record["messages"], list):
                for msg in record["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ("system", "user", "assistant") and content:
                        texts.append(content)

    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on JSONL document corpora."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/synthetic_docs/docs.jsonl",
            "data/distill/train.jsonl",
        ],
        help="Input JSONL files to train on.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/tokenizer",
        help="Output directory for tokenizer files.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32768,
        help="Target vocabulary size (default 32768).",
    )

    args = parser.parse_args()

    # Collect all text from input files
    all_texts: list[str] = []
    for inp in args.inputs:
        texts = _extract_texts(inp)
        print(f"Loaded {len(texts)} texts from {inp}")
        all_texts.extend(texts)

    if not all_texts:
        print("ERROR: No text found in any input file. Aborting.")
        sys.exit(1)

    total_chars = sum(len(t) for t in all_texts)
    print(f"Total texts: {len(all_texts)}, Total chars: {total_chars:,}")

    # Import and train the tokenizer (src must be on path first)
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tokenizer_wrapper import (
        SPECIAL_TOKENS,
        save_tokenizer,
        train_tokenizer_on_texts,
    )

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
    tok = train_tokenizer_on_texts(all_texts, vocab_size=args.vocab_size)

    final_vocab = tok.get_vocab_size()
    print(f"Trained tokenizer vocab size: {final_vocab}")

    # Verify special tokens are at expected positions
    vocab = tok.get_vocab()
    for i, token in enumerate(SPECIAL_TOKENS):
        actual_id = vocab.get(token)
        if actual_id != i:
            print(
                f"WARNING: Special token {token!r} has ID {actual_id}, " f"expected {i}"
            )

    # Save tokenizer files
    save_tokenizer(tok, args.out_dir)
    print(f"Saved tokenizer to {args.out_dir}/")


if __name__ == "__main__":
    main()
