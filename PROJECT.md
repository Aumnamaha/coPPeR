# coPPeR — From-Scratch ~1B Document Transformer

## Overview
coPPeR is a from-scratch **~1 billion parameter causal Transformer** trained on synthetic
document data, designed for document reading, structured extraction, and document editing.

### Data Pipeline
- **Synthetic documents**: Generated via Qwen3.6-35B-A3B at `http://192.168.56.1:1234/v1`
- **Teacher labels**: Produced by Antigravity CLI (`agy`) at `C:\Users\Aum Namaha\AppData\Local\agy\bin\agy.exe`
- Data stored under `data/`, tokenized datasets packed for training

### Model
- Llama/Mistral-style causal Transformer architecture, built from scratch (~1B params)
- **NOT** a fine-tune of any pretrained model
- Domain BPE tokenizer (32k vocab) trained on document corpora

### Training Hardware
- NVIDIA GeForce RTX 5070 Ti Laptop GPU — 12GB VRAM, Blackwell `sm_120`
- PyTorch CUDA `cu132`, `TORCH_CUDA_ARCH_LIST=12.0`
- 32GB system RAM

## Phases

### [x] Phase 1: Repo scaffold, AGENTS.md, and isolated build\.venv
- Git repo initialized with remote origin (GitHub)
- `.gitignore`, `AGENTS.md` (teacher rules), `PROJECT.md` created
- `src/`, `scripts/`, `tests/`, `build/`, `data/`, `checkpoints/`, `exports/` directories
- Isolated venv at `build\.venv` with `tokenizers`, `datasets`, `safetensors`, `accelerate`, `tqdm`
- Smoke test passing

### [x] Phase 2: Synthetic document generator + teacher distillation pipeline
- Qwen3.6-35B-A3B script for synthetic doc generation
- Antigravity CLI (`agy`) integration for structured label production
- Data loading and cleaning scripts

### [x] Phase 3: Custom domain BPE tokenizer (32k vocab) + tokenized dataset packing
- Trained ByteLevel BPE tokenizer with `vocab_size == 32768` and special tokens at IDs 0–9
- Binary `.bin` packer producing `uint16` pretrain streams and `int32` SFT input/label arrays

### [x] Phase 4: From-scratch ~1B causal Transformer architecture & unit tests
- `CopperConfig` (~992M params) with RMSNorm, RoPE, GQA (3:1), SwiGLU
- Full forward pass with shifted cross-entropy loss and `-100` ignore masking
- Gradient checkpointing via `torch.utils.checkpoint` support
- KV-cache autoregressive generation (`generate()`) verified equivalent to full forwards
- Causal mask isolation, safetensors save/load roundtrip, lossless newline tokenizer roundtrip
- 61 total tests passing (ruff + black clean)
- Implement causal attention, RoPE, SwiGLU feedforward, RMSNorm
- Unit tests for each architectural component
- Correctness checks against reference implementations

### [x] Phase 5: Two-stage training (Domain Pretraining + Teacher SFT) on RTX 5070 Ti
- Single-command trainer (`scripts/train_copper.py`) with VRAM preflight guard
- Stage 1: Causal LM pretraining on packed `pretrain_train.bin` (cosine LR, gradient checkpointing)
- Stage 2: SFT distillation on packed `sft_train_inputs.bin` + `sft_train_labels.bin` (-100 masked loss)
- Checkpoint save/resume (`--resume` flag), periodic snapshots
- `--smoke` mode: tiny 2-layer config runs in <15 s even with Qwen3.6 loaded on GPU
- Screen-aware selection tokens `<|sel_start|>` (ID 10) and `<|sel_end|>` (ID 11)
- Editor tool schemas (`EDITOR_TOOLS_SPEC`): replace_selection, insert_markdown_table,
  apply_heading_style, fix_grammar_and_clarity, extract_structured_json
- 79 total tests passing (ruff + black clean)

### [x] Phase 6: Evaluation & GGUF export for multi-platform inference
- Dataset preparation orchestrator (`scripts/prepare_dataset.py`): one-command pipeline (generate → distill → tokenizer train → pack)
- Model evaluator (`scripts/evaluate_copper.py`): perplexity on pretrain eval, SFT cross-entropy loss, 4 deterministic task probes; saves JSON report to `exports/eval_report.json`
- GGUF v3 exporter (`scripts/export_gguf.py`): pure Python + NumPy writer producing both HF-compatible directory and standalone GGUF binary with F16 / Q8_0 quantization
- HuggingFace-compatible export: `copper-1b-hf/` with config.json, model.safetensors, tokenizer.json (compatible with `convert_hf_to_gguf.py`)
- Llama architecture metadata keys in GGUF (`llama.context_length`, `llama.attention.head_count_kv`, etc.)
- Q8_0 block quantization: 34 bytes per 32-element block (float16 scale + int8 weights)
- 89 total tests passing (ruff + black clean)

### [x] Phase 7: Windows screen-aware cursor overlay app, editor bridge & local Docker server
- `app/editor_bridge.py`: Detects installed document editors (LibreOffice Writer, Microsoft Word), reads foreground window title via ctypes `user32.GetForegroundWindow`, captures cursor-highlighted selection via clipboard backup/copy/restore, applies surgical span replacements via `ctypes SendInput` (Ctrl+V)
- `app/inference_server.py`: FastAPI server bound to `127.0.0.1:8088` with `GET /health` and `POST /v1/edit-span` endpoints; lazy model loading (PyTorch from checkpoint or llama-cpp proxy via `LLAMA_CPP_URL` env var); builds screen-aware ChatML prompts with `<|doc_start|>...<|sel_start|>{highlight}<|sel_end|>...<|doc_end|>` tokens
- `app/desktop_overlay.py`: Native Windows tkinter floating command bar triggered via global hotkey `Ctrl+Alt+Space` (RegisterHotKey); shows editor badge, captures selection, calls inference server, previews edit, applies in-place; install prompt modal when no DOCX/ODT editor found (`winget install TheDocumentFoundation.LibreOffice`)
- `docker/Dockerfile`: python:3.12-slim base with FastAPI + PyTorch CPU/CUDA-selectable
- `docker/docker-compose.yml`: binds strictly `127.0.0.1:8088:8088` (never public), mounts `../checkpoints:/app/checkpoints:ro` and `../exports:/app/exports:ro` read-only for offline personal use
- Screen-aware selection tokens `<|sel_start|>` (ID 10) and `<|sel_end|>` (ID 11) with full pipeline integration
- Editor tool schemas (`EDITOR_TOOLS_SPEC`) validated by `validate_teacher_output()`
- All Phase 7 files written as UTF-8 without BOM; all tests passing

## Project Structure
```
coPPeR/
├── .gitignore
├── AGENTS.md              # Teacher agent output schema rules
├── PROJECT.md             # This file — phase tracker
├── src/                   # Model architecture code
│   ├── __init__.py
│   ├── data_schemas.py    # TASK_TEMPLATES, EDITOR_TOOLS_SPEC, validation
│   ├── tokenizer_wrapper.py  # CopperTokenizer (ChatML, SFT loss masking)
│   └── model.py           # CopperConfig + CopperForCausalLM (~1B params)
├── scripts/               # Data generation / training / export scripts
│   ├── __init__.py
│   ├── generate_synthetic_docs.py  # Qwen3.6-35B-A3B doc generator
│   ├── distill_teacher.py         # Antigravity CLI teacher labels
│   ├── train_tokenizer.py         # Domain BPE tokenizer training
│   ├── pack_dataset.py            # Binary .bin dataset packing
│   ├── train_copper.py            # Two-stage trainer (pretrain + SFT)
│   ├── prepare_dataset.py         # One-command pipeline orchestrator
│   ├── evaluate_copper.py         # Perplexity + task accuracy evaluation
│   └── export_gguf.py             # GGUF v3 F16/Q8_0 exporter
├── app/                   # Desktop overlay, editor bridge, inference server
│   ├── __init__.py
│   ├── editor_bridge.py     # Editor detection, clipboard capture, paste-back
│   ├── inference_server.py  # FastAPI /v1/edit-span endpoint (127.0.0.1:8088)
│   └── desktop_overlay.py   # tkinter floating bar + global hotkey Ctrl+Alt+Space
├── docker/                # Local Docker setup
│   ├── Dockerfile           # python:3.12-slim with FastAPI + PyTorch
│   └── docker-compose.yml   # 127.0.0.1:8088 binding, :ro mounts
├── tests/                 # Unit + integration tests
│   ├── test_env_smoke.py
│   ├── test_data_pipeline.py    # Data loading & cleaning (29 tests)
│   ├── test_tokenizer_and_packing.py  # Tokenizer training + packing (16 tests)
│   ├── test_model.py            # Model architecture (~1B, causal mask, KV cache) (16 tests)
│   ├── test_training_loop.py    # VRAM guard, smoke training, checkpoint save/resume (18 tests)
│   ├── test_eval_and_export.py  # Evaluation + GGUF v3 export (10 tests)
│   └── test_desktop_and_docker.py  # Editor detection, FastAPI endpoints, Docker config (10 tests)
├── build/
│   └── .venv/             # Isolated project virtual environment
├── data/                  # Git-ignored: synthetic docs, packed datasets, tokenizer
├── checkpoints/           # Training checkpoints (.safetensors + config.json)
└── exports/               # GGUF exports, eval reports (git-ignored)
```

## Quick-Start Runbook

### Step A — Dataset Preparation (one command)
```powershell
cd "C:\Users\Aum Namaha\Documents\Proj\coPPeR"
& "build\.venv\Scripts\python.exe" scripts/prepare_dataset.py --doc-count 300 --max-distill-docs 300
```
This runs: generate synthetic docs → distill teacher labels → train tokenizer → pack binary datasets.

### Step B — Unload LM Studio & Run Training
```powershell
# IMPORTANT: Close/unload Qwen3.6 from LM Studio first (free VRAM)
& "build\.venv\Scripts\python.exe" scripts/train_copper.py --stage all --checkpoint-dir checkpoints/
```
For smoke verification (< 15 seconds):
```powershell
& "build\.venv\Scripts\python.exe" scripts/train_copper.py --smoke --checkpoint-dir checkpoints/smoke_test
```

### Step C — Evaluate & Export GGUF
```powershell
# Run evaluation (perplexity + task probes)
& "build\.venv\Scripts\python.exe" scripts/evaluate_copper.py --checkpoint checkpoints/copper_1b_final

# Export to GGUF Q8_0 for Intel Arc A750 inference
& "build\.venv\Scripts\python.exe" scripts/export_gguf.py --checkpoint checkpoints/copper_1b_final --quant q8_0
```

### Step D — Launch Local Server + Overlay App
```powershell
# Option 1: Python FastAPI server (direct)
$env:CHECKPOINT_DIR = "checkpoints/copper_1b_final"
& "build\.venv\Scripts\python.exe" -m app.inference_server

# Option 2: Docker container (recommended for isolation)
docker compose -f docker/docker-compose.yml up --build

# Launch the Windows cursor overlay (in a separate terminal)
$env:CHECKPOINT_DIR = "checkpoints/copper_1b_final"
& "build\.venv\Scripts\python.exe" app/desktop_overlay.py
```
The overlay triggers via **Ctrl+Alt+Space** from anywhere on your desktop.
Highlight text in any editor → press Ctrl+Alt+Space → type instruction → click **Generate Fix** → review → click **Apply In-Place**.

## Quick Start
```powershell
# Activate isolated venv
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\Activate.ps1"

# Run tests
pytest -v

# Set CUDA arch for Blackwell GPU
$env:TORCH_CUDA_ARCH_LIST = "12.0"
```
