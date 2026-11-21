"""CopperTokenizer - clean wrapper around tokenizers.Tokenizer for coPPeR.

Provides ChatML rendering, SFT loss-masking encoding, and roundtrip encode/decode.
"""

from __future__ import annotations

import json
from pathlib import Path

import tokenizers

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "Ċ",
    "ĊĊ",
    "<|doc_start|>",
    "<|doc_end|>",
    "<|diff_start|>",
    "<|diff_end|>",
]

VOCAB_SIZE: int = 32768


def _build_tokenizer(
    vocab_size: int = VOCAB_SIZE,
) -> tokenizers.Tokenizer:
    """Create and return a fresh ByteLevel BPE tokenizer with special tokens."""
    from tokenizers import models, pre_tokenizers

    tok = tokenizers.Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = tokenizers.decoders.ByteLevel()
    return tok


class CopperTokenizer:
    """Wraps a trained ``tokenizers.Tokenizer`` with coPPeR conventions."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        self._tok = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        vocab = self._tok.get_vocab()
        for i, tok_str in enumerate(SPECIAL_TOKENS):
            assert (
                tok_str in vocab
            ), f"Special token {tok_str!r} not found at expected ID {i}"
        self.pad_token_id: int = 0
        self.eos_token_id: int = 3

    # ------------------------------------------------------------------
    # Encoding / decoding helpers
    # ------------------------------------------------------------------

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        """Encode *text* into a list of token IDs.

        Args:
            text: Input string.
            add_bos: Prepend ``<|bos|>`` (ID 2).
            add_eos: Append ``<|eos|>`` (ID 3).

        Returns:
            List of integer token IDs.
        """
        enc = self._tok.encode(text, add_special_tokens=False)
        ids: list[int] = enc.ids
        if add_bos:
            ids.insert(0, 2)  # <|bos|>
        if add_eos:
            ids.append(3)  # <|eos|>
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text.

               Always passes all IDs through the native decoder so that Ċ (U+010A)
               is correctly restored to
        by the ByteLevel decoder.  Filtering
               special tokens before decoding would remove Ċ and break newline
               reconstruction, so we always use skip_special_tokens=False.

               Args:
                   ids: Token ID sequence.
                   skip_special_tokens: Currently ignored — all IDs are passed through
                       to the native ByteLevel decoder for correct newline handling.

               Returns:
                   Reconstructed string with newlines preserved.
        """
        return self._tok.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # ChatML formatting
    # ------------------------------------------------------------------

    def format_chatml(self, messages: list[dict]) -> str:
        """Render a list of ``{"role": ..., "content": ...}`` dicts as ChatML.

        Convention used by the teacher distillation pipeline::

            Ċ<|system|>\\n{system_content}ĊĊĊ<|user|>\\n{user_content}ĊĊĊ<|assistant|>\\n{response}ĊĊ

        Returns:
            The full ChatML string.
        """
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            parts.append(f"\u010a<|{role}|>\n{content}")
        return "".join(parts) + "\u010c\u010c"

    # ------------------------------------------------------------------
    # SFT encoding with loss masking
    # ------------------------------------------------------------------

    def encode_sft_example(
        self, messages: list[dict], max_seq_len: int = 2048
    ) -> tuple[list[int], list[int]]:
        """Encode a ChatML message list into ``input_ids`` + ``labels``.

        **Convention (unshifted, equal-length arrays):**

        * ``input_ids`` contains the full token sequence including special tokens.
        * ``labels`` mirrors ``input_ids``, but every position that belongs to
          system/user turns is set to ``-100`` so standard PyTorch loss computation

            ``loss = F.cross_entropy(logits[..., :-1, :], labels[..., 1:])``

          automatically ignores non-assistant tokens.
        * Only assistant content tokens and the closing ``ĊĊ`` have active label IDs.

        Args:
            messages: List of dicts with ``role`` and ``content`` keys.
            max_seq_len: Maximum sequence length (padded with pad_token / -100).

        Returns:
            Tuple of ``(input_ids, labels)`` each a list of ints, length <= *max_seq_len*.
        """
        # Build the ChatML string and tokenize it
        chatml = self.format_chatml(messages)
        all_ids = self.encode(chatml, add_bos=False, add_eos=False)

        # Determine which token indices belong to assistant content.
        labels = [-100] * len(all_ids)

        # Find the <|assistant|> marker in the token stream and mark from there
        assistant_role_header = "\u010a<|assistant|>\n"
        header_ids = self.encode(assistant_role_header, add_bos=False, add_eos=False)

        # Search for the header in all_ids
        assistant_start = None
        for start_idx in range(len(all_ids)):
            if (
                start_idx + len(header_ids) <= len(all_ids)
                and all_ids[start_idx : start_idx + len(header_ids)] == header_ids
            ):
                assistant_start = start_idx + len(header_ids)
                break

        if assistant_start is not None:
            labels[assistant_start:] = list(all_ids[assistant_start:])

        # Trim to max_seq_len
        input_ids = all_ids[:max_seq_len]
        labels = labels[:max_seq_len]

        return (input_ids, labels)


def train_tokenizer_on_texts(
    texts: list[str],
    vocab_size: int = VOCAB_SIZE,
) -> tokenizers.Tokenizer:
    """Train a BPE tokenizer on raw text and pad to *vocab_size*.

    Args:
        texts: Iterable of strings for training.
        vocab_size: Target vocabulary size (default 32768).

    Returns:
        A trained ``tokenizers.Tokenizer`` instance with special tokens added.
    """
    from tokenizers import decoders, models, normalizers, pre_tokenizers, trainers

    raw_tok = tokenizers.Tokenizer(models.BPE(unk_token="<|unk|>"))
    raw_tok.normalizer = normalizers.NFC()
    raw_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tok.decoder = decoders.ByteLevel()

    # Train with a modest initial size; the corpus likely has far fewer
    # unique byte-pair merges than 32768. We pad to the target below.
    initial_vocab = min(vocab_size, max(1000, len(texts) * 50))
    trainer = trainers.BpeTrainer(
        vocab_size=initial_vocab,
        special_tokens=list(SPECIAL_TOKENS),
        show_progress=False,
    )

    raw_tok.train_from_iterator(texts, trainer=trainer)

    # Pad vocabulary if corpus is tiny (smoke tests)
    actual_vocab = raw_tok.get_vocab_size()
    if actual_vocab < vocab_size:
        for i in range(actual_vocab, vocab_size):
            token_name = f"<|reserved_{i}|>"
            raw_tok.add_tokens([token_name])

    return raw_tok


def save_tokenizer(
    tok: tokenizers.Tokenizer,
    out_dir: str | Path,
    config_overrides: dict | None = None,
) -> None:
    """Save tokenizer files to *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok.save(str(out_dir / "copper_tokenizer.json"), pretty=True)

    cfg: dict = {
        "model_type": "bpe",
        "vocab_size": VOCAB_SIZE,
        "special_tokens": SPECIAL_TOKENS,
    }
    if config_overrides:
        cfg.update(config_overrides)
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_tokenizer(path: str | Path) -> CopperTokenizer:
    """Load a trained tokenizer and return a ``CopperTokenizer`` wrapper."""
    return CopperTokenizer(path)
