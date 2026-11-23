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

See [`LICENSE`](LICENSE) for full terms.

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



---

## coPPeR v2 - Phases 8-12 Features (Zephium-Inspired Document Workspace)

### Phase 8: SFT Refinement & Data Recovery

- **`<|unk|>` Token Suppression** (`ban_token_ids` in `generate()`): The model's decoding pipeline now masks unknown token logits with `-inf`, preventing `<|unk|>` emission during generation.
- **SFT Skill Augmentation**: `scripts/augment_missing_skills.py` generates clean SFT records for previously-rejected tasks (`markdown_table_refactor`, `editor_tool_call`).
- **7-Skill SFT Packing** (`--extra-sft-jsonl`): All 2,032 SFT records (1,432 + 600) packed into binary dataset.
- **Checkpoint Resume** (`--resume-from`): Stage 2 SFT can warm-start from existing checkpoints without resetting weights.

### Phase 9: Zephium-Inspired Native WebView2 Split-Canvas Workspace

- **Native Edge WebView2 Window**: `app/workspace_app.py` launches a native Windows desktop app using `pywebview` (no browser tab, no Node.js).
- **Split-Canvas Layout**: Left AI Assistant Rail (`360px`) + Right Document & Diff Canvas.
- **Dashed-Square Selection Tool**: Custom SVG icon button that captures highlighted text from any active window via Win32 bridge.
- **Numbered Point-Back Citations** (circle numbers): AI responses include citation badges that highlight the corresponding document section when clicked.
- **SSE Token Streaming**: Real-time token-by-token streaming on `/v1/chat/completions` and `/edit/stream`.
- **Global Hotkey** (`Ctrl+Alt+Space`): Captures highlighted text from any Windows editor and opens the workspace.

### Phase 10: Structured Workspace Artifacts & Local Persistence

- **Local Artifact Store**: `app/artifact_store.py` manages `.md` notes, checklists, and tables in `workspace_notes/`.
- **Interactive Checklists**: `- [ ]` / `- [x]` items rendered as clickable checkboxes with auto-save.
- **Editable Tables**: GFM Markdown tables rendered as inline-editable grids (`+ Add Row`, `+ Add Column`).
- **DOCX Export**: `export_to_docx()` converts artifacts to `.docx` using `python-docx`.

### Phase 11: Parallel Multi-Agent Document Runner ("Send to Work")

- **4 Specialized Sub-Agents** running in parallel:
  - Extractor: Structured JSON key-facts & entity extraction
  - Table Builder: Unstructured to clean GFM Markdown table
  - Proofreader: Grammar, clarity, executive span refinement
  - Artifact Packager: Synthesizes combined outputs into candidate artifact

- **Human-in-the-Loop Approval Gate**: Jobs transition to `awaiting_approval` state -- proposed writes stay in memory until explicit `approve_job()` or `reject_job()`.
- **Live Progress Graph**: Real-time per-agent status cards (`Queued`, `Running`, `Done`).

### Phase 12: Live Hybrid Model Switcher & Final v2 Release

- **Model Profile Router** (`app/model_router.py`): Pluggable inference profiles with local persistence.
  - `coPPeR-1B Local` (PyTorch CUDA/CPU)
  - `coPPeR-1B GGUF / Arc A750` (llama-server at `http://127.0.0.1:8080/v1`)
  - `Qwen-35B LAN` (LM Studio at `http://192.168.56.1:1234/v1`)
  - `BYOK Cloud OpenAI-Compatible` (custom endpoint, env var key)

- **Live Model Switcher Pill**: Dropdown in composer UI fetches `/api/models`, switches via `/api/models/select`.
- **Endpoint Configuration Popover**: Click gear icon to edit base URL and model name per profile.
- **Intel Arc A750 Sync Bundle** (`/api/models/arc-bundle`): Generates `arc_a750_launch.ps1` + `manifest.json` with SHA256 verification of GGUF binary.

- **Windows App Packaging**: `scripts/build_windows_app.py --check-only` / `--build` (PyInstaller).
- **One-Click Launcher**: `Launch_coPPeR.bat` starts uvicorn server + WebView2 workspace.

---

## API Reference (v2 Endpoints)

### Inference Server (`http://127.0.0.1:8088`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status, backend mode, available backends |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (supports `stream: true`) |
| `POST` | `/v1/edit-span` | Screen-aware document editing |
| `GET` | `/workspace` | WebView2 workspace UI |
| `POST` | `/api/workspace/capture` | Capture highlighted selection from active window |
| `POST` | `/api/workspace/apply` | Apply text to active window via clipboard |

### Artifacts & Documents (`http://127.0.0.1:8088`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/artifacts` | List saved notes/tables/checklists |
| `GET` | `/api/artifacts/{slug}` | Load specific artifact |
| `POST` | `/api/artifacts` | Create/update artifact on local disk |
| `DELETE` | `/api/artifacts/{slug}` | Delete artifact |
| `POST` | `/api/artifacts/export-docx` | Export to `.docx` |

### Work Agents (`http://127.0.0.1:8088`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/work/jobs` | Launch parallel multi-agent job |
| `GET` | `/api/work/jobs` | List recent work jobs |
| `GET` | `/api/work/jobs/{job_id}` | Poll live per-agent progress |
| `POST` | `/api/work/jobs/{job_id}/approve` | Human-in-the-loop approval (persists artifact) |
| `POST` | `/api/work/jobs/{job_id}/reject` | Reject proposed edits (no disk write) |

### Model Switcher (`http://127.0.0.1:8088`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | Return active profile + all configured profiles |
| `POST` | `/api/models/select` | Switch active model profile live |
| `POST` | `/api/models/arc-bundle` | Generate Intel Arc A750 deployment bundle |

---

## Deployment: Intel Arc A750 (8GB VRAM)

1. Copy `exports/copper-1b-q8_0.gguf` to the target machine.
2. Install Python 3.12+ and run:
   ```powershell
   pip install llama-cpp-python openai
   ```
3. Start llama-server:
   ```powershell
   python -m llama_cpp.server --model copper-1b-q8_0.gguf --host 127.0.0.1 --port 8080
   ```
4. Set environment variables and start the workspace:
   ```powershell
   $env:COPPER_BACKEND = "llamacpp"
   $env:LLAMA_CPP_URL = "http://127.0.0.1:8080/v1"
   python app/workspace_app.py
   ```

Or use the auto-generated Arc A750 bundle:
```powershell
# Generate bundle via API first, then run:
.\workspace_notesrc_a750_launch.ps1
```

---

## Testing & Quality Assurance

All tests are in `tests/` and run with:
```powershell
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\python.exe" -m pytest -v
```

The test suite covers: model architecture, KV cache equivalence, tokenizer + dataset packing, training loop validation, evaluation + GGUF export, desktop overlay bridge, workspace UI (SSE streaming, citations, artifacts), multi-agent runner with human-in-the-loop approval, and Phase 12 model router / packaging.

---

## License

**Personal Software License - Copyright (c) 2026 Thirunagari Aum Namaha. All Rights Reserved.**

This software is the confidential property of Thirunagari Aum Namaha. Unauthorized copying, modification, distribution, or use of this software via any medium is strictly prohibited except for personal, private use by the copyright holder. This software is provided "AS IS", without warranty of any kind.
