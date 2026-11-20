"""Pack pretraining documents and SFT ChatML records into binary .bin files.

Writes fixed-length ``uint16`` token streams for pretraining and paired
``input_ids`` / ``labels`` (``int32``) arrays for SFT fine-tuning.

Usage::

    python scripts/pack_dataset.py \\
        --tokenizer data/tokenizer/copper_tokenizer.json \\
        --out-dir data/packed \\
        --seq-len 2048 \\
        --eval-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_tokenizer(tokenizer_path: str):
    """Load CopperTokenizer from a JSON tokenizer file."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tokenizer_wrapper import load_tokenizer

    return load_tokenizer(tokenizer_path)


def _extract_texts(jsonl_path: str) -> list[str]:
    """Extract text content from a JSONL file."""
    texts: list[str] = []
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Warning: {jsonl_path} does not exist, skipping.")
        return texts

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Synthetic docs have a 'content' field
            if "content" in record and isinstance(record["content"], str):
                texts.append(record["content"])

    return texts


def pack_pretrain(
    texts: list[str],
    tokenizer,
    seq_len: int = 2048,
) -> tuple[np.ndarray, dict]:
    """Pack raw document texts into fixed-length uint16 token streams.

    Documents are concatenated with ``<|eos|>`` separators and split into
    fixed-length chunks of *seq_len* tokens.

    Returns:
        (flat_tokens_array, metadata_dict) where flat_tokens is a 1-D uint16
        NumPy array and metadata contains total_tokens, num_sequences, seq_len.
    """
    eos_id = tokenizer.eos_token_id
    all_ids: list[int] = []

    for text in texts:
        ids = tokenizer.encode(text, add_eos=False)
        if ids:
            all_ids.extend(ids)
            all_ids.append(eos_id)  # separator between documents

    total_tokens = len(all_ids)
    num_sequences = total_tokens // seq_len

    if num_sequences == 0:
        print("WARNING: Not enough tokens for even one sequence. Returning empty.")
        return np.array([], dtype=np.uint16), {
            "total_tokens": 0,
            "num_sequences": 0,
            "seq_len": seq_len,
        }

    # Trim to fit exact multiple of seq_len
    trimmed = all_ids[: num_sequences * seq_len]
    flat = np.array(trimmed, dtype=np.uint16)

    meta = {
        "total_tokens": len(trimmed),
        "num_sequences": num_sequences,
        "seq_len": seq_len,
    }
    return flat, meta


def pack_sft(
    jsonl_path: str,
    tokenizer,
    seq_len: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pack SFT ChatML records into input_ids (uint16) and labels (int32).

    Each record is encoded via ``tokenizer.encode_sft_example(..., max_seq_len=seq_len)``,
    padded to *seq_len* with pad_token in inputs and -100 in labels.

    Returns:
        (inputs_array, labels_array, metadata) where inputs is uint16,
        labels is int32, both shape (num_records, seq_len).
    """
    path = Path(jsonl_path)
    if not path.exists():
        print(f"Warning: {jsonl_path} does not exist.")
        return (
            np.zeros((0, seq_len), dtype=np.uint16),
            np.zeros((0, seq_len), dtype=np.int32),
            {"num_records": 0, "seq_len": seq_len},
        )

    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # SFT ChatML records have 'messages' key
            if "messages" in record and isinstance(record["messages"], list):
                messages = record["messages"]
                # Only process records that look like valid ChatML (have assistant)
                has_assistant = any(m.get("role") == "assistant" for m in messages)
                if has_assistant:
                    records.append(record)

    num_records = len(records)

    inputs_arr = np.zeros((num_records, seq_len), dtype=np.uint16)
    labels_arr = np.full((num_records, seq_len), -100, dtype=np.int32)

    for i, record in enumerate(records):
        input_ids, labels = tokenizer.encode_sft_example(
            record["messages"], max_seq_len=seq_len
        )

        n = len(input_ids)
        inputs_arr[i, :n] = input_ids[:seq_len]
        labels_arr[i, :n] = np.array(labels[:seq_len], dtype=np.int32)

    meta = {"num_records": num_records, "seq_len": seq_len}
    return inputs_arr, labels_arr, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack pretrain docs and SFT records into binary .bin files."
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer/copper_tokenizer.json",
        help="Path to trained tokenizer JSON file.",
    )
    parser.add_argument(
        "--docs-jsonl",
        nargs="+",
        default=["data/synthetic_docs/docs.jsonl"],
        help="Pretraining document JSONL files.",
    )
    parser.add_argument(
        "--sft-train-jsonl",
        type=str,
        default="data/distill/train.jsonl",
        help="SFT training ChatML JSONL file.",
    )
    parser.add_argument(
        "--sft-eval-jsonl",
        type=str,
        default="",
        help="SFT evaluation ChatML JSONL file (optional).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/packed",
        help="Output directory for .bin files.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length for packing (default 2048).",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.05,
        help="Eval split ratio (default 0.05).",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = _load_tokenizer(args.tokenizer)
    print(f"Tokenizer loaded: vocab_size={tokenizer._tok.get_vocab_size()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pretraining pack ---
    all_texts: list[str] = []
    for doc_jsonl in args.docs_jsonl:
        texts = _extract_texts(doc_jsonl)
        print(f"Loaded {len(texts)} docs from {doc_jsonl}")
        all_texts.extend(texts)

    if all_texts:
        flat_tokens, pretrain_meta = pack_pretrain(all_texts, tokenizer, args.seq_len)
        out_path = out_dir / "pretrain_train.bin"
        np.memmap(
            out_path, dtype=np.uint16, mode="w+", shape=(len(flat_tokens),)
        ).setflags(write=True)
        mm = np.memmap(out_path, dtype=np.uint16, mode="r+", shape=(len(flat_tokens),))
        mm[:] = flat_tokens
        del mm

        with open(out_dir / "pretrain_meta.json", "w", encoding="utf-8") as f:
            json.dump(pretrain_meta, f, indent=2)
        print(
            f"Pretrain packed: {len(flat_tokens):,} tokens, "
            f"{pretrain_meta['num_sequences']} sequences -> pretrain_train.bin"
        )

    # --- SFT pack (train) ---
    if args.sft_train_jsonl and Path(args.sft_train_jsonl).exists():
        sft_inputs, sft_labels, sft_train_meta = pack_sft(
            args.sft_train_jsonl, tokenizer, args.seq_len
        )

        for name, arr in [
            ("sft_train_inputs", sft_inputs),
            ("sft_train_labels", sft_labels),
        ]:
            out_path = out_dir / f"{name}.bin"
            dtype = np.uint16 if "inputs" in name else np.int32
            mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=arr.shape)
            mm[:] = arr
            del mm

        with open(out_dir / "sft_meta.json", "w", encoding="utf-8") as f:
            json.dump(sft_train_meta, f, indent=2)
        print(
            f"SFT train packed: {sft_train_meta['num_records']} records -> "
            f"sft_train_inputs.bin + sft_train_labels.bin"
        )

    # --- SFT pack (eval) if provided ---
    if args.sft_eval_jsonl and Path(args.sft_eval_jsonl).exists():
        eval_inputs, eval_labels, eval_meta = pack_sft(
            args.sft_eval_jsonl, tokenizer, args.seq_len
        )

        for name, arr in [
            ("sft_eval_inputs", eval_inputs),
            ("sft_eval_labels", eval_labels),
        ]:
            out_path = out_dir / f"{name}.bin"
            dtype = np.uint16 if "inputs" in name else np.int32
            mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=arr.shape)
            mm[:] = arr
            del mm

        print(
            f"SFT eval packed: {eval_meta['num_records']} records -> "
            f"sft_eval_inputs.bin + sft_eval_labels.bin"
        )

    print("Packing complete.")


if __name__ == "__main__":
    main()
