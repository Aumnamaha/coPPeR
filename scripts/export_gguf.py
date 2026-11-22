#!/usr/bin/env python3
r"""Export a trained coPPeR checkpoint to HuggingFace-compatible format and GGUF v3 binary.

Usage:
    # Default Q8_0 quantized export (smaller file, slightly lower quality)
    python scripts/export_gguf.py --checkpoint checkpoints/copper_1b_final

    # F16 unquantized export (highest quality, larger file)
    python scripts/export_gguf.py --checkpoint checkpoints/copper_1b_final --quant f16

    # Export to specific directory
    python scripts/export_gguf.py --out-dir exports/ --quant q8_0
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import CopperConfig

# ---------------------------------------------------------------------------
# GGUF v3 constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF tensor data types (v3) — only F16 and Q8_0 needed here
GGUF_TYPE_F16 = 5
GGUF_TYPE_Q8_0 = 17

# Q8_0 block size and bytes per block
Q8_BLOCK_SIZE = 32
Q8_BLOCK_BYTES = 34  # 2 (float16 scale) + 32 (int8 weights)


# ---------------------------------------------------------------------------
# GGUF binary helpers
# ---------------------------------------------------------------------------


def _write_u32(f, value: int) -> None:
    f.write(struct.pack("<I", value & 0xFFFFFFFF))


def _write_i64(f, value: int) -> None:
    f.write(struct.pack("<q", value))


def _write_string(f, s: str) -> None:
    encoded = s.encode("utf-8")
    _write_u32(f, len(encoded))
    f.write(encoded)


def _write_type(f, t: int) -> None:
    _write_u32(f, t)


# ---------------------------------------------------------------------------
# Q8_0 quantization (pure NumPy)
# ---------------------------------------------------------------------------


def _quantize_q8_0(arr: np.ndarray) -> bytes:
    """Quantize a float16 array using the GGUF Q8_0 scheme.

    Splits into 32-element blocks. Each block:
      - 1 x float16 scale = max_abs / 112.0
      - 32 x int8 weights clipped to [-112, +112]
    Total per block: 34 bytes.
    """
    result = bytearray()
    n_elements = arr.shape[-1]

    for i in range(0, n_elements, Q8_BLOCK_SIZE):
        block = arr[i : i + Q8_BLOCK_SIZE].astype(np.float32)
        while len(block) < Q8_BLOCK_SIZE:
            block = np.append(block, 0.0)

        abs_block = np.abs(block)
        max_abs = float(np.max(abs_block))
        scale = max_abs / 112.0 if max_abs > 0 else 0.0

        weights_int8 = (block / (scale + 1e-8) * 112.0).round().astype(np.int8)
        np.clip(weights_int8, -112, 112, out=weights_int8)

        result.extend(struct.pack("<e", float(scale)))  # half-precision float
        result.extend(weights_int8.tobytes())

    return bytes(result)


# ---------------------------------------------------------------------------
# Tensor name mapping (PyTorch -> GGUF Llama convention)
# ---------------------------------------------------------------------------


def _tensor_name(layer_idx: int, component: str, config: CopperConfig) -> str:
    """Map internal model tensor to GGUF Llama naming."""
    if component == "token_embd":
        return "token_embd.weight"
    elif component == "output_proj":
        return "output.weight"
    elif component == "output_norm":
        return "output_norm.weight"
    else:
        prefix = f"blk.{layer_idx}."
        mapping = {
            "attn_norm": "attn_norm.weight",
            "ffn_norm": "ffn_norm.weight",
            "attn_q": "attn_q.weight",
            "attn_k": "attn_k.weight",
            "attn_v": "attn_v.weight",
            "attn_output": "attn_output.weight",
            "ffn_gate": "ffn_gate.weight",
            "ffn_up": "ffn_up.weight",
            "ffn_down": "ffn_down.weight",
        }
        return prefix + mapping[component]


def _tensor_dims(name: str, config: CopperConfig) -> list[int]:
    """Return GGUF tensor shape (row-major)."""
    if name == "token_embd.weight" or name == "output.weight":
        return [config.vocab_size, config.hidden_size]
    elif name.startswith("output_norm"):
        return [config.hidden_size]

    # Block tensors
    _parts = name.split(".")
    comp_name = ".".join(_parts[2:])

    hs = config.hidden_size
    num_groups = config.num_attention_heads // config.num_key_value_heads

    if "attn_q" in comp_name:
        return [hs, hs]
    elif "attn_k" in comp_name or "attn_v" in comp_name:
        return [hs, hs // num_groups]
    elif "attn_output" in comp_name:
        return [hs, hs]
    elif "ffn_gate" in comp_name or "ffn_up" in comp_name:
        return [hs, config.intermediate_size]
    elif "ffn_down" in comp_name:
        return [config.intermediate_size, hs]
    else:
        # Norm weights are 1D
        return [hs]


def _is_q8_0_tensor(name: str) -> bool:
    """Return True if this tensor should be Q8_0 quantized (non-norm tensors)."""
    return "norm" not in name and "embd" not in name


# ---------------------------------------------------------------------------
# GGUF export
# ---------------------------------------------------------------------------


def export_gguf(
    checkpoint_dir: Path,
    tokenizer_path: Path,
    out_dir: Path,
    quantize: str | None = "q8_0",
) -> dict[str, str]:
    """Export a trained model to GGUF v3 binary and HF-compatible directory.

    Returns dict of output paths.
    """
    # Load config
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {checkpoint_dir}")

    with open(str(config_path), "r") as f:
        cfg_dict = json.load(f)
    config = CopperConfig(**cfg_dict)

    # Load model weights from safetensors
    st_path = checkpoint_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"No model.safetensors in {checkpoint_dir}")

    from safetensors.torch import load_file as load_safetensors

    state_dict = load_safetensors(str(st_path))

    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Write HuggingFace-compatible directory
    # -----------------------------------------------------------------------
    hf_dir = out_dir / "copper-1b-hf"
    hf_dir.mkdir(exist_ok=True)

    hf_config = {
        "_name_or_path": str(checkpoint_dir),
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 3,
        "hidden_act": "swiglu",
        "hidden_size": config.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "mlp_bias": False,
        "model_type": "llama",
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "pad_token_id": 0,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "vocab_size": config.vocab_size,
    }

    with open(str(hf_dir / "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    # Copy tokenizer if available
    if tokenizer_path.exists():
        import shutil

        shutil.copy2(str(tokenizer_path), str(hf_dir / "tokenizer.json"))

    print(f"HF-compatible directory: {hf_dir}")

    # -----------------------------------------------------------------------
    # 2. Collect tensors with GGUF metadata
    # -----------------------------------------------------------------------
    tensor_info = []  # (gguf_name, torch_tensor, dims, gguf_type)

    # Token embedding
    if "embed_tokens.weight" in state_dict:
        t = state_dict["embed_tokens.weight"]
        name = _tensor_name(0, "token_embd", config)
        tensor_info.append(
            (
                name,
                t.to(torch.float32),
                [config.vocab_size, config.hidden_size],
                GGUF_TYPE_F16,
            )
        )

    # Output projection (lm_head)
    if "lm_head.weight" in state_dict:
        t = state_dict["lm_head.weight"]
        name = _tensor_name(0, "output_proj", config)
        tensor_info.append(
            (
                name,
                t.to(torch.float32),
                [config.vocab_size, config.hidden_size],
                GGUF_TYPE_F16,
            )
        )

    # Output norm (norm)
    if "norm.weight" in state_dict:
        t = state_dict["norm.weight"]
        name = _tensor_name(0, "output_norm", config)
        tensor_info.append(
            (name, t.to(torch.float32), [config.hidden_size], GGUF_TYPE_F16)
        )

    # Decoder layers
    for i in range(config.num_hidden_layers):
        prefix = f"layers.{i}."
        layer_map = {
            "input_layernorm.weight": ("attn_norm",),
            "post_attention_layernorm.weight": ("ffn_norm",),
            "self_attn.q_proj.weight": ("attn_q",),
            "self_attn.k_proj.weight": ("attn_k",),
            "self_attn.v_proj.weight": ("attn_v",),
            "self_attn.o_proj.weight": ("attn_output",),
            "mlp.gate_proj.weight": ("ffn_gate",),
            "mlp.up_proj.weight": ("ffn_up",),
            "mlp.down_proj.weight": ("ffn_down",),
        }

        for key, components in layer_map.items():
            full_key = prefix + key
            if full_key not in state_dict:
                continue
            t = state_dict[full_key].to(torch.float32)
            comp = components[0]
            name = _tensor_name(i, comp, config)
            dims = _tensor_dims(name, config)

            # Q8_0 for non-norm tensors only
            gguf_type = (
                GGUF_TYPE_Q8_0
                if (quantize == "q8_0" and not is_norm_tensor(comp))
                else GGUF_TYPE_F16
            )
            tensor_info.append((name, t, dims, gguf_type))

    total_tensors = len(tensor_info)
    print(f"Exporting {total_tensors} tensors to GGUF v3...")

    # -----------------------------------------------------------------------
    # 3. Build metadata pairs
    # -----------------------------------------------------------------------
    metadata_pairs: list[tuple[str, int, bytes]] = []

    def add_meta_string(key: str, value: str) -> None:
        """Type 1 = string (length-prefixed UTF-8)."""
        data = value.encode("utf-8")
        metadata_pairs.append((key, 1, data))

    def add_meta_int64(key: str, value: int) -> None:
        """Type 0 = int64."""
        metadata_pairs.append((key, 0, struct.pack("<q", value)))

    # Architecture and model parameters
    total_params = (
        config.hidden_size
        * config.num_hidden_layers
        * (2 + 8.0 / 3)  # approx: embed + norm(2) + attn(4) + mlp(3) per layer
        + 2 * config.vocab_size * config.hidden_size
    )

    add_meta_string("general.architecture", "llama")
    add_meta_string("general.name", "coPPeR-1B")
    add_meta_int64("general.parameter_count", int(total_params))
    add_meta_int64("general.quantization_version", 2)

    add_meta_string("llama.context_length", str(config.max_position_embeddings))
    add_meta_string("llama.embedding_length", str(config.hidden_size))
    add_meta_int64("llama.block_count", config.num_hidden_layers)
    add_meta_int64("llama.feed_forward_length", config.intermediate_size)
    add_meta_int64("llama.attention.head_count", config.num_attention_heads)
    add_meta_int64("llama.attention.head_count_kv", config.num_key_value_heads)
    add_meta_string("llama.rope.freq_base", str(config.rope_theta))
    add_meta_string("llama.attention.layer_norm_rms_epsilon", str(config.rms_norm_eps))

    # Tokenizer metadata (if tokenizer available)
    if tokenizer_path.exists():
        from tokenizers import Tokenizer as HFTokenizer

        tok = HFTokenizer.from_file(str(tokenizer_path))
        vocab_dict = tok.get_vocab()
        tokens_list = list(vocab_dict.keys())

        add_meta_string("tokenizer.ggml.model", "gpt2")
        # Type 8 = raw bytes for token arrays
        token_bytes = json.dumps(tokens_list, ensure_ascii=False).encode("utf-8")
        metadata_pairs.append(("tokenizer.ggml.tokens", 8, token_bytes))

        add_meta_int64("tokenizer.ggml.bos_token_id", vocab_dict.get("<|bos|>", 1))
        add_meta_int64("tokenizer.ggml.eos_token_id", vocab_dict.get("<|eos|>", 3))
        add_meta_int64("tokenizer.ggml.padding_token_id", vocab_dict.get("<|pad|>", 0))

    # -----------------------------------------------------------------------
    # 4. Write GGUF v3 binary (streaming, with alignment)
    # -----------------------------------------------------------------------
    quant_tag = "q8_0" if quantize == "q8_0" else "f16"
    gguf_path = out_dir / f"copper-1b-{quant_tag}.gguf"

    buf = bytearray()

    # --- Header: magic(4) + version(4) + tensor_count(4 placeholder) + meta_count(4 placeholder) ---
    buf.extend(GGUF_MAGIC)
    buf.extend(struct.pack("<I", GGUF_VERSION))  # version (4 bytes)
    buf.extend(b"\x00\x00\x00\x00")  # tensor_count placeholder
    buf.extend(b"\x00\x00\x00\x00")  # metadata_count placeholder

    # Positions to patch later (relative from start of header)
    tensor_count_pos = len(buf) - 8  # offset of tensor_count field
    metadata_count_pos = len(buf) - 4  # offset of metadata_count field

    # --- Metadata pairs ---
    _write_u32_buf(buf, len(metadata_pairs))
    for key, vtype, data in metadata_pairs:
        _write_string_buf(buf, key)
        _write_type_buf(buf, vtype)
        if vtype == 1:  # string: length + content
            buf.extend(struct.pack("<I", len(data)))
            buf.extend(data)
        elif vtype == 0:  # int64: raw 8 bytes
            buf.extend(data)

    # --- Tensor headers ---
    _write_u32_buf(buf, total_tensors)
    for name, t, dims, gguf_type in tensor_info:
        _write_string_buf(buf, name)
        buf.extend(struct.pack("<I", len(dims)))  # dimension count
        buf.extend(b"\x00" * (len(dims) * 8))  # placeholder for dimensions

    # --- Tensor data with 32-byte alignment ---
    current_pos = len(buf)
    if current_pos % 32 != 0:
        pad = 32 - (current_pos % 32)
        buf.extend(b"\x00" * pad)

    quantize_flag = "q8_0" if quantize == "q8_0" else None
    for name, t, dims, gguf_type in tensor_info:
        # Convert to float32 first (bfloat16 not supported by NumPy)
        arr = t.detach().to(torch.float32).cpu().numpy()

        if quantize_flag == "q8_0":
            raw_bytes = _quantize_q8_0(arr.astype(np.float16))
        else:
            # F16 — write as float16 bytes
            if arr.dtype != np.float16:
                arr = arr.astype(np.float16)
            raw_bytes = arr.tobytes()

        buf.extend(raw_bytes)

        # Align to 32 bytes after each tensor
        current_pos = len(buf)
        if current_pos % 32 != 0:
            pad = 32 - (current_pos % 32)
            buf.extend(b"\x00" * pad)

    # --- Patch header with actual counts (at known offsets) ---
    buf[tensor_count_pos : tensor_count_pos + 4] = struct.pack("<I", total_tensors)
    buf[metadata_count_pos : metadata_count_pos + 4] = struct.pack(
        "<I", len(metadata_pairs)
    )

    gguf_data = bytes(buf)

    with open(str(gguf_path), "wb") as f:
        f.write(gguf_data)

    file_size = len(gguf_data)
    print(f"GGUF v3 binary written: {gguf_path} ({file_size / 1e6:.2f} MB)")

    return {
        "hf_dir": str(hf_dir),
        "gguf_binary": str(gguf_path),
        "quantization": quantize or "none",
        "tensor_count": total_tensors,
        "file_size_bytes": file_size,
    }


def is_norm_tensor(comp: str) -> bool:
    """Check if a component name corresponds to a normalization layer."""
    return "norm" in comp


# ---------------------------------------------------------------------------
# GGUF binary helpers for bytearray (mutable version of _write_* functions)
# ---------------------------------------------------------------------------


def _write_u32_buf(buf, value: int) -> None:
    """Write a little-endian uint32 to the bytearray buffer."""
    buf.extend(struct.pack("<I", value & 0xFFFFFFFF))


def _write_string_buf(buf, s: str) -> None:
    """Write a length-prefixed UTF-8 string to the bytearray buffer."""
    encoded = s.encode("utf-8")
    buf.extend(struct.pack("<I", len(encoded)))
    buf.extend(encoded)


def _write_type_buf(buf, t: int) -> None:
    """Write a GGUF value type (uint32) to the bytearray buffer."""
    buf.extend(struct.pack("<I", t))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export coPPeR checkpoint to GGUF v3")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/copper_1b_final")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("data/tokenizer/copper_tokenizer.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("exports"))
    parser.add_argument("--quant", choices=["f16", "q8_0"], default="q8_0")
    args = parser.parse_args()

    print("=" * 60)
    print("coPPeR GGUF v3 Exporter")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Quantization: {args.quant}")
    print("=" * 60)

    if not args.checkpoint.exists():
        print(f"ERROR: Checkpoint directory not found: {args.checkpoint}")
        sys.exit(1)

    results = export_gguf(
        args.checkpoint, args.tokenizer, args.out_dir, quantize=args.quant
    )

    print("\nExport complete:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
