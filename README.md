# coPPeR — Custom Causal Transformer for Screen-Aware Document Editing

**Personal ~1B parameter causal Transformer** built from scratch in PyTorch for local document intelligence, span-level editing, structured extraction, and Windows screen-aware cursor assistance.

---

## Overview

coPPeR is a **992,563,200-parameter** causal language model (`CopperForCausalLM`) implemented entirely from scratch — no Hugging Face `transformers` dependency. It follows the Llama/Mistral architectural family with RMSNorm pre-normalization, RoPE rotary embeddings, grouped-query attention (GQA), and SwiGLU feed-forward networks.

The project provides a complete pipeline:
1. **Synthetic document generation** via teacher model distillation
2. **Custom BPE tokenizer training** on domain data
3. **Two-stage training**: Stage 1 pre-training → Stage 2 supervised fine-tuning (SFT) with teacher-distilled examples
4. **Evaluation** on perplexity, cross-entropy loss, and live task accuracy
5. **GGUF v3 export** (`F16` / `Q8_0`) for deployment on Intel Arc A750 via `llama.cpp`
6. **FastAPI inference server** with OpenAI-compatible endpoints
7. **Windows desktop overlay app** (`Ctrl+Alt+Space`) for screen-aware cursor editing

---

## Architecture & Measured Specs

### CopperConfig

| Parameter | Value |
|---|---|
| `vocab_size` | 32,768 (custom domain BPE) |
| `hidden_size` | 1,536 |
| `intermediate_size` | 5,632 (SwiGLU) |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 4 (3:1 GQA ratio) |
| `max_position_embeddings` | 4,096 |
| `rope_theta` | 100,000.0 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | False (untied) |
| **Total Parameters** | **992,563,200** |

### Training Metrics

#### Stage 1 — Pre-training
- Steps: 1,000
- Tokens processed: ~552,960
- Average loss: 4.7891
- Peak VRAM (RTX 5070 Ti): 8.46 GB
- Precision: `torch.bfloat16`

#### Stage 2 — SFT Distillation
- Steps: 600
- Teacher-distilled records: ~1,432
- Average loss: 0.4781
- Peak VRAM (RTX 5070 Ti): 8.67 GB
- Tasks: `json_extraction`, `cursor_span_edit`, `editor_tool_call`, `grounded_qa_not_found`

### Export Artifacts

| Artifact | Format | Size | Path |
|---|---|---|---|
| HuggingFace-compatible model | `.safetensors` + `config.json` + `tokenizer.json` | ~1.9 GB | `exports/copper-1b-hf/` |
| GGUF v3 F16 | GGUF binary | ~1.9 GB | `exports/copper-1b-f16.gguf` |
| GGUF v3 Q8_0 | GGUF binary (quantized) | 555.52 MB | `exports/copper-1b-q8_0.gguf` |

---

## Repository Structure

```
coPPeR/
├── .gitignore              # Comprehensive ignore rules
├── AGENTS.md               # Agent workspace instructions
├── LICENSE                 # Personal All Rights Reserved license (2026)
├── PROJECT.md              # Detailed project planning & phase tracking
├── README.md               # This file

├── src/                    # Core model implementation
│   ├── __init__.py
│   ├── data_schemas.py     # Pydantic schemas for distillation outputs
│   ├── model.py            # CopperForCausalLM, CopperConfig (992M params)
│   └── tokenizer_wrapper.py  # Custom BPE tokenizer wrapper

├── scripts/                # Training & pipeline scripts
│   ├── __init__.py
│   ├── generate_synthetic_docs.py   # Synthetic document generation
│   ├── distill_teacher.py           # Teacher distillation (agy CLI)
│   ├── train_tokenizer.py           # Custom BPE tokenizer training
│   ├── pack_dataset.py              # Binary dataset packing (.bin)
│   ├── train_copper.py              # Two-stage PyTorch training loop
│   ├── prepare_dataset.py           # One-command orchestrator
│   ├── evaluate_copper.py           # Perplexity & task evaluation
│   └── export_gguf.py               # GGUF v3 F16/Q8_0 exporter

├── app/                    # Desktop overlay + inference server
│   ├── __init__.py
│   ├── editor_bridge.py       # Windows clipboard/UI automation bridge
│   ├── inference_server.py    # FastAPI OpenAI-compatible server (127.0.0.1:8088)
│   └── desktop_overlay.py     # tkinter floating command bar (Ctrl+Alt+Space)

├── tests/                  # 109+ unit/integration tests
│   ├── __init__.py
│   ├── test_env_smoke.py          # Environment & import smoke tests
│   ├── test_data_pipeline.py      # Data pipeline + UTF-8 subprocess fix
│   ├── test_tokenizer_and_packing.py  # Tokenizer + dataset packing
│   ├── test_model.py              # Model architecture + KV cache
│   ├── test_training_loop.py      # Training loop + CUDA bounds validation
│   ├── test_eval_and_export.py    # Evaluation + GGUF v3 export tests
│   ├── test_desktop_and_docker.py  # Editor detection + Docker config
│   └── test_overlay_and_bridge.py  # Clipboard roundtrip + server alignment

├── docker/                 # Local Docker inference setup
│   ├── Dockerfile           # Python 3.12-slim FastAPI container
│   └── docker-compose.yml   # 127.0.0.1:8088 bind, read-only mounts
```

### Git-Ignored Directories (never committed)

| Directory | Contents |
|---|---|
| `build/` | Isolated `.venv`, pip caches, build artifacts |
| `data/` | `synthetic_docs/docs.jsonl`, distillation outputs, tokenizer JSON, packed `.bin` files |
| `checkpoints/` | `copper_1b_final/`, step checkpoints, smoke test state |
| `exports/` | GGUF binaries (`*.gguf`), HuggingFace export dir, evaluation reports |

---

## Quick-Start Runbook (Windows PowerShell 5.1)

### Prerequisites

- **Python** 3.12+ installed in shared environment at `C:\Users\Aum Namaha\Documents\finalenv\Scripts\python.exe`
- **Project virtual env**: `C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\python.exe`
- **GPU**: NVIDIA RTX 5070 Ti Laptop (12 GB VRAM) for training; Intel Arc A750 (8 GB) for inference
- **Docker Desktop** (optional, for local server deployment)

### Step A — Dataset Preparation

```powershell
# Activate isolated venv
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\Activate.ps1"

# Run the full pipeline: generate docs -> distill teacher -> train tokenizer -> pack dataset
python scripts/prepare_dataset.py --workers 4
```

This orchestrator runs `generate_synthetic_docs.py` (300 synthetic documents) → `distill_teacher.py` (7 tasks per doc via `agy.exe`) → `train_tokenizer.py` (custom BPE on 32,768 vocab) → `pack_dataset.py` (binary `.bin` packing with validation).

### Step B — Training

```powershell
# Stage 1: Pre-training (500 steps default, ~4 hours)
python scripts/train_copper.py --stage pretrain --steps 1000 --batch-size 8

# Stage 2: SFT Distillation (600 steps, ~3 hours)
python scripts/train_copper.py --stage sft --steps 600 --batch-size 4

# Smoke test (verifies pipeline on tiny model)
python scripts/train_copper.py --smoke
```

### Step C — Evaluation & GGUF Export

```powershell
# Evaluate trained checkpoint on perplexity + task accuracy
python scripts/evaluate_copper.py --checkpoint checkpoints/copper_1b_final

# Export to HuggingFace-compatible directory AND standalone GGUF v3 binary
python scripts/export_gguf.py --quant q8_0   # 555 MB Q8_0 quantized
python scripts/export_gguf.py --quant f16    # ~1.9 GB full precision
```

### Step D — Local Inference Server

#### Option 1: Direct Python (development)

```powershell
uvicorn app.inference_server:app --host 127.0.0.1 --port 8088 --log-level info
```

#### Option 2: Docker (production-like)

```powershell
docker compose -f docker/docker-compose.yml up --build
# Server available at http://127.0.0.1:8088
```

### Step E — Windows Cursor Overlay

```powershell
python app/desktop_overlay.py
```

**Hotkey**: `Ctrl+Alt+Space` toggles the floating command bar.

**Workflow**:
1. Highlight text in any document editor (LibreOffice Writer, Microsoft Word)
2. Press `Ctrl+Alt+Space` to open the overlay
3. Type your edit instruction
4. Click **Generate Fix** — model proposes a surgical span replacement
5. Click **Apply In-Place** — replaces highlighted text using Windows clipboard automation

---

## Testing & Linting

```powershell
# Run full test suite (109+ tests)
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\python.exe" -m pytest -v

# Linting
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\python.exe" -m ruff check src scripts app tests

# Formatting check
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\python.exe" -m black --check src scripts app tests
```

---

## Hardware & Environment

| Component | Specification |
|---|---|
| **Training Machine** | ROG Laptop — Intel Core Ultra 9, RTX 5070 Ti Laptop (12 GB VRAM), 32 GB RAM, Windows 11 |
| **Inference Target** | Windows PC — Intel Arc A750 (8 GB VRAM), `llama.cpp` GGUF v3 |
| **Python Shared Env** | `C:\Users\Aum Namaha\Documents\finalenv\Scripts\python.exe` (PyTorch 2.14.1+cu132) |
| **Project Venv** | `build\.venv\` (ruff, black, pytest, tokenizers, datasets, safetensors, accelerate) |
| **CUDA Toolkit** | cu132 (CUDA 13.2) |
| **Shell** | Windows PowerShell 5.1 (no pwsh 7) |

---

## License

This project is licensed under a **Personal Software License (All Rights Reserved)**.

Copyright (c) 2026 Thirunagari Aum Namaha. All Rights Reserved.

The software, model weights, and associated documentation are for the **personal, private use** of the author only. No copying, distribution, modification, or commercial use is permitted without prior written permission. See `LICENSE` for full terms.

---

## Key Technical Decisions & Fixes

| Issue | Resolution |
|---|---|
| Windows `UnicodeDecodeError` in subprocess (`cp1252` vs UTF-8) | Added `encoding="utf-8", errors="replace"` + `stdout=None` guard in `distill_teacher.py` |
| Parallel worker crashes (no flush between threads) | Added `concurrent.futures.ThreadPoolExecutor`, thread-safe `Lock()`, per-record `.flush()` |
| CUDA device-side assert on token ID out-of-bounds | CPU bounds check before `.to(device)` raises clean `ValueError`; smoke mode uses `vocab_size=32768` |
| BFloat16 export failure (`TypeError: Got unsupported ScalarType`) | Cast all tensors via `.to(torch.float32).cpu().numpy()` before GGUF serialization |
| 64-bit ctypes pointer truncation (access violation) | Declared explicit `argtypes`/`restype` with `ctypes.c_void_p` for all Win32 API calls |
| Missing `import ctypes.wintypes` | Added conditional import at module top of `editor_bridge.py` |
| Server-overlay endpoint mismatch | Aligned overlay payload schema with FastAPI route; surfaced actionable HTTP error messages |

---

*Built by Thirunagari Aum Namaha — Final-year CSE/AIML Student. Personal use only.*
