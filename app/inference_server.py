#!/usr/bin/env python3
r"""Local OpenAI-compatible inference server for coPPeR.

Binds to 127.0.0.1:8088 with endpoints:
    GET  /health          -- status, backend mode, EDITOR_TOOLS_SPEC
    POST /v1/edit-span    -- screen-aware document editing request
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

_sys_path = str(Path(__file__).resolve().parent.parent / "src")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from data_schemas import EDITOR_TOOLS_SPEC, validate_teacher_output
from tokenizer_wrapper import CopperTokenizer

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EditSpanRequest(BaseModel):
    """Screen-aware edit-span request payload."""

    highlighted_text: str
    surrounding_context: str = ""
    instruction: str
    active_app: str = "generic_editor"
    mode: str = "cursor_span_edit"  # cursor_span_edit | editor_tool_call


class EditSpanResponse(BaseModel):
    """Edit-span response payload."""

    result: str
    valid: bool
    reason: str
    editor_context: dict[str, Any]


# ---------------------------------------------------------------------------
# Lazy model / tokenizer loading
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_backend_mode = "none"  # pytorch | llama_cpp_proxy


def _load_pytorch_model(checkpoint_dir: str) -> None:
    """Load CopperForCausalLM from a safetensors checkpoint (lazy)."""
    global _model, _backend_mode

    if _model is not None:
        return  # Already loaded

    try:
        import torch

        model_path = Path(checkpoint_dir) / "model.safetensors"
        config_path = Path(checkpoint_dir) / "config.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No model.safetensors at {model_path}. "
                "Run `scripts/train_copper.py --smoke` first or point --checkpoint to a valid dir."
            )

        from safetensors.torch import load_file as safe_load

        # Import config and model classes
        _sys_model = str(Path(__file__).resolve().parent.parent / "src")
        if _sys_model not in sys.path:
            sys.path.insert(0, _sys_model)
        from model import CopperConfig, CopperForCausalLM

        cfg_data = json.loads(config_path.read_text())
        config = CopperConfig(
            **{k: v for k, v in cfg_data.items() if hasattr(CopperConfig(), k)}
        )
        state_dict = safe_load(str(model_path))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = CopperForCausalLM(config).to(device).to(dtype)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        _model = model
        _backend_mode = "pytorch"

    except Exception as exc:
        raise RuntimeError(f"Failed to load PyTorch model: {exc}") from exc


def _load_tokenizer(tokenizer_path: str) -> None:
    """Load the domain BPE tokenizer (lazy)."""
    global _tokenizer
    if _tokenizer is not None:
        return

    tp = Path(tokenizer_path)
    if tp.exists():
        _tokenizer = CopperTokenizer(tp)
    else:
        raise FileNotFoundError(
            f"Tokenizer not found at {tp}. " "Run `scripts/train_tokenizer.py` first."
        )


def _build_screen_aware_prompt(
    highlighted_text: str,
    surrounding_context: str,
    instruction: str,
    active_app: str,
) -> list[dict]:
    """Build a ChatML message list with screen-aware tokens."""
    wrapped_doc = (
        "<|doc_start|>"
        f"{surrounding_context}"
        "<|sel_start|>"
        f"{highlighted_text}"
        "<|sel_end|>"
        f"{surrounding_context}<|doc_end|>"
    )

    tools_list = ", ".join(EDITOR_TOOLS_SPEC.keys())

    system_prompt = (
        "You are coPPeR, a screen-aware document editing assistant. "
        "The user has highlighted text in their active editor window. "
        f"Active app: {active_app}. Available tools: {tools_list}. "
        "Follow the instruction precisely."
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Document context:\n{wrapped_doc}\n\n" f"Instruction: {instruction}"
            ),
        },
    ]


def _run_pytorch_inference(messages: list[dict], max_new_tokens: int = 128) -> str:
    """Run inference with the loaded PyTorch model."""
    import torch

    tokenizer = _tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")

    chatml_str = tokenizer.format_chatml(messages)
    input_ids = tokenizer.encode(chatml_str, add_bos=False, add_eos=False)

    device = next(_model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Resolve and ban the <|unk|> token ID during decoding
    unk_id = tokenizer.token_to_id("<|unk|>")
    with torch.no_grad():
        outputs = _model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # greedy for deterministic editing
            eos_token_id=tokenizer.eos_token_id,
            ban_token_ids=[unk_id],
        )

    generated_ids = outputs[0][input_tensor.shape[1] :].tolist()
    result = tokenizer.decode(generated_ids)

    # Strip any stray <|unk|> substrings from decoded text
    result = result.replace("<|unk|>", "")
    return result


def _run_llama_cpp_proxy(prompt: str, max_new_tokens: int = 128) -> str:
    """Proxy to an external llama-server (GGUF) instance."""
    import requests

    url = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8081/v1/completions")
    payload = {
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stop": ["<|doc_end|>", "<|sel_start|>"],
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("text", "")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="coPPeR Inference Server",
    description=(
        "Local OpenAI-compatible inference server for screen-aware document editing. "
        "Binds to 127.0.0.1:8088."
    ),
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "backend_mode": _backend_mode,
        "EDITOR_TOOLS_SPEC": EDITOR_TOOLS_SPEC,
    }


@app.post("/v1/edit-span", response_model=EditSpanResponse)
async def edit_span(req: EditSpanRequest) -> EditSpanResponse:
    """Screen-aware document editing endpoint.

    Accepts highlighted text + instruction, runs inference, validates output.
    """
    # Lazy-load tokenizer and model on first request if not already loaded
    try:
        _load_tokenizer(
            str(
                Path(__file__).resolve().parent.parent
                / "data"
                / "tokenizer"
                / "copper_tokenizer.json"
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Tokenizer not found: {exc}",
        ) from exc

    # Auto-load PyTorch model if CHECKPOINT_DIR env var is set and not yet loaded
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")
    if _backend_mode == "none" and checkpoint_dir:
        try:
            _load_pytorch_model(checkpoint_dir)
        except Exception:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Failed to load model from {checkpoint_dir}. "
                    "Make sure the checkpoint directory is valid."
                ),
            )

    messages = _build_screen_aware_prompt(
        req.highlighted_text,
        req.surrounding_context,
        req.instruction,
        req.active_app,
    )

    if _backend_mode == "pytorch":
        result = _run_pytorch_inference(messages)
    elif _backend_mode == "llama_cpp_proxy":
        chatml_str = _tokenizer.format_chatml(messages)  # type: ignore[name-defined]
        result = _run_llama_cpp_proxy(chatml_str)
    else:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model loaded. Set CHECKPOINT_DIR env var for PyTorch mode, "
                "or LLAMA_CPP_URL for llama-cpp proxy mode."
            ),
        )

    # Validate the output against the requested task mode
    is_valid, reason = validate_teacher_output(req.mode, result)

    return EditSpanResponse(
        result=result.strip(),
        valid=is_valid,
        reason=reason,
        editor_context={
            "active_app": req.active_app,
            "mode": req.mode,
            "available_tools": list(EDITOR_TOOLS_SPEC.keys()),
        },
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the inference server."""
    import uvicorn

    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "")
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "tokenizer"),
    )
    llama_cpp_url = os.environ.get("LLAMA_CPP_URL", "")

    if checkpoint_dir:
        try:
            _load_tokenizer(str(Path(tokenizer_path) / "copper_tokenizer.json"))
            _load_pytorch_model(checkpoint_dir)
        except Exception:  # noqa: BLE001
            print(
                "Warning: PyTorch model load failed, falling back to proxy if "
                "LLAMA_CPP_URL is set."
            )
            llama_cpp_url = os.environ.get("LLAMA_CPP_URL", "")

    if not llama_cpp_url and _backend_mode == "none":
        print(
            "INFO: No backend loaded. Set CHECKPOINT_DIR for PyTorch mode "
            "or LLAMA_CPP_URL for llama-cpp proxy."
        )

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8088,
        log_level="info",
    )


if __name__ == "__main__":
    main()
