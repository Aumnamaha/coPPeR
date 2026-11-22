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
from collections.abc import AsyncGenerator
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
# OpenAI-compatible request/response models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False


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
_backend_mode = "none"
_available_backends: list[str] = []

# Backend configuration from environment
_COPPER_BACKEND = os.environ.get("COPPER_BACKEND", "pytorch")
_LLAMA_CPP_URL = os.environ.get("LLAMA_CPP_URL", "http://127.0.0.1:8081/v1/completions")
_LMSTUDIO_URL = os.environ.get("LMSTUDIO_URL", "http://192.168.56.1:1234/v1")


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

    url = os.environ.get("LLAMA_CPP_URL", _LLAMA_CPP_URL)
    payload = {
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "stop": ["<|doc_end|>", "<|sel_start|>"],
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("text", "")


def _run_lmstudio_proxy(messages: list[dict], max_new_tokens: int = 128) -> str:
    """Proxy to an LM Studio server (OpenAI-compatible API)."""
    import requests

    url = os.environ.get("LMSTUDIO_URL", _LMSTUDIO_URL)
    payload = {
        "messages": messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    resp = requests.post(f"{url}/chat/completions", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def _detect_available_backends() -> list[str]:
    """Detect which backends are available based on environment configuration."""
    backends: list[str] = []
    if os.environ.get("CHECKPOINT_DIR"):
        backends.append("pytorch")
    if (
        _LLAMA_CPP_URL
        and not _LLAMA_CPP_URL.endswith("v1/completions")
        or "8081" in _LLAMA_CPP_URL
    ):
        backends.append("llamacpp")
    if _LMSTUDIO_URL:
        backends.append("lmstudio")
    # Always report pytorch as available config since it can be loaded on demand
    if not backends and os.environ.get("CHECKPOINT_DIR"):
        backends = ["pytorch"]
    elif "llamacpp" in _COPPER_BACKEND or "lmstudio" in _COPPER_BACKEND:
        # If user explicitly requests a backend, check connectivity
        pass
    if not backends:
        backends = ["pytorch"]  # default
    return backends


# ---------------------------------------------------------------------------
# SSE streaming generator
# ---------------------------------------------------------------------------


def _sse_event(data: dict[str, Any]) -> str:
    """Format a JSON-serializable dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


async def _stream_pytorch_responses(
    messages: list[dict],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    """Token-by-token SSE stream for PyTorch inference."""
    import torch

    if _model is None:
        raise RuntimeError("Model not loaded")

    tokenizer = _tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer not loaded")

    chatml_str = tokenizer.format_chatml(messages)
    input_ids = tokenizer.encode(chatml_str, add_bos=False, add_eos=False)

    device = next(_model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    unk_id = tokenizer.token_to_id("<|unk|>")

    with torch.no_grad():
        outputs = _model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            ban_token_ids=[unk_id] if unk_id is not None else [],
        )

    generated_ids = outputs[0][input_tensor.shape[1] :].tolist()

    # Emit tokens one at a time as SSE chunks
    for i, token_id in enumerate(generated_ids):
        if token_id == tokenizer.eos_token_id:
            break
        token_text = tokenizer.decode([token_id])
        # Strip <|unk|> from each emitted chunk
        token_text = token_text.replace("<|unk|>", "")
        if not token_text or token_text.isspace():
            continue
        yield _sse_event(
            {
                "id": f"chatcmpl-{i}",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "copper-1b",
                "choices": [{"index": 0, "delta": {"content": token_text}}],
            }
        )

    yield _sse_event(
        {
            "id": f"chatcmpl-{len(generated_ids)}",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "copper-1b",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield "data: [DONE]\n\n"


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
    """Health check endpoint with backend metadata."""
    available = _detect_available_backends()
    return {
        "status": "ok",
        "backend_mode": _backend_mode,
        "available_backends": available,
        "configured_backend": _COPPER_BACKEND,
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
                "or LLAMA_CPP_URL/LMSTUDIO_URL for proxy backends."
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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Supports both non-streaming (JSON) and streaming (SSE) responses.
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

    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    # Handle streaming vs non-streaming
    if req.stream:
        from fastapi.responses import StreamingResponse

        # Guard: only stream when model is actually loaded
        if _model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Set CHECKPOINT_DIR for PyTorch mode.",
            )

        return StreamingResponse(
            _stream_pytorch_responses(
                messages_dicts, req.max_new_tokens, req.temperature
            ),
            media_type="text/event-stream",
        )

    # Non-streaming: run full inference and return JSON response
    if _backend_mode == "pytorch":
        result = _run_pytorch_inference(
            messages_dicts, max_new_tokens=req.max_new_tokens
        )
    elif _backend_mode == "llama_cpp_proxy":
        chatml_str = _tokenizer.format_chatml(messages_dicts)  # type: ignore[name-defined]
        result = _run_llama_cpp_proxy(chatml_str, max_new_tokens=req.max_new_tokens)
    elif _backend_mode == "lmstudio_proxy":
        result = _run_lmstudio_proxy(messages_dicts, max_new_tokens=req.max_new_tokens)
    else:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model loaded. Set CHECKPOINT_DIR for PyTorch mode, "
                "LLAMA_CPP_URL for llama-cpp proxy, or LMSTUDIO_URL for LM Studio."
            ),
        )

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": "copper-1b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/edit/stream")
async def edit_stream(req: ChatCompletionRequest):
    """Streaming document editing endpoint (SSE token-by-token)."""
    from fastapi.responses import StreamingResponse

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

    messages_dicts = [{"role": m.role, "content": m.content} for m in req.messages]

    return StreamingResponse(
        _stream_pytorch_responses(messages_dicts, req.max_new_tokens, req.temperature),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Workspace UI — static file serving & API endpoints
# ---------------------------------------------------------------------------

_WORKSPACE_DIR = Path(__file__).resolve().parent / "workspace_ui"


def _read_workspace_file(filename: str) -> bytes:
    """Read a workspace asset file (index.html, styles.css, app.js)."""
    path = _WORKSPACE_DIR / filename
    if not path.exists():
        return b"Not Found"
    return path.read_bytes()


@app.get("/workspace")
async def serve_workspace_index() -> Any:
    """Serve the main workspace HTML page."""
    from fastapi.responses import Response

    html = _read_workspace_file("index.html")
    return Response(content=html, media_type="text/html; charset=utf-8")


@app.get("/workspace/styles.css")
async def serve_workspace_css() -> Any:
    """Serve workspace CSS."""
    from fastapi.responses import Response

    css = _read_workspace_file("styles.css")
    return Response(content=css, media_type="text/css; charset=utf-8")


@app.get("/workspace/app.js")
async def serve_workspace_js() -> Any:
    """Serve workspace JavaScript."""
    from fastapi.responses import Response

    js = _read_workspace_file("app.js")
    return Response(content=js, media_type="application/javascript; charset=utf-8")


@app.post("/api/workspace/capture")
async def api_capture() -> dict[str, Any]:
    """Capture highlighted selection from the active window.

    Returns JSON with `selection` text and `charCount`.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from editor_bridge import capture_highlighted_selection

        result = capture_highlighted_selection()
        selection_text = result.get("wrapped_document", "")
        return {
            "selection": selection_text,
            "charCount": len(selection_text),
            "filename": None,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Capture failed: {exc}") from exc


@app.post("/api/workspace/apply")
async def api_apply(req: dict[str, str]) -> dict[str, Any]:
    """Apply edited text to the active editor window.

    Accepts JSON body `{\"text\": \"...\"}`.
    """
    try:
        text = req.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'text' field")

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from editor_bridge import apply_selection_replacement

        ok = apply_selection_replacement(text)
        return {"success": bool(ok)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Apply failed: {exc}") from exc


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

    # Select backend based on COPPER_BACKEND env var
    backend = _COPPER_BACKEND.lower()
    if backend == "pytorch":
        if checkpoint_dir:
            try:
                _load_tokenizer(str(Path(tokenizer_path) / "copper_tokenizer.json"))
                _load_pytorch_model(checkpoint_dir)
            except Exception:  # noqa: BLE001
                print(
                    "Warning: PyTorch model load failed, falling back to proxy if "
                    "LLAMA_CPP_URL or LMSTUDIO_URL is set."
                )
        else:
            print("INFO: CHECKPOINT_DIR not set; running in proxy-only mode.")
    elif backend == "llamacpp":
        _backend_mode = "llama_cpp_proxy"
        print(f"INFO: Using llama-cpp proxy at {_LLAMA_CPP_URL}")
    elif backend == "lmstudio":
        _backend_mode = "lmstudio_proxy"
        print(f"INFO: Using LM Studio proxy at {_LMSTUDIO_URL}")
    else:
        # Unknown backend — default to pytorch and warn
        print(
            f"WARNING: Unknown COPPER_BACKEND='{_COPPER_BACKEND}', defaulting to pytorch"
        )
        if checkpoint_dir:
            try:
                _load_tokenizer(str(Path(tokenizer_path) / "copper_tokenizer.json"))
                _load_pytorch_model(checkpoint_dir)
            except Exception:  # noqa: BLE001
                print("Warning: PyTorch model load failed.")
        else:
            print("INFO: CHECKPOINT_DIR not set; running in proxy-only mode.")

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8088,
        log_level="info",
    )


if __name__ == "__main__":
    main()
