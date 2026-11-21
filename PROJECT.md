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

### [ ] Phase 7: Windows screen-aware overlay app + Dockerized local server
- Python desktop overlay (PyQt6 / Dear PyGui) for real-time document highlighting
- LibreOffice UNO / python-docx editor integration with tool-call dispatch
- Docker container for local inference REST API (FastAPI + vLLM-compatible endpoint)
- Auto-prompt installer: detects active editor window and injects screen-aware context

## Project Structure
```
coPPeR/
├── .gitignore
├── AGENTS.md              # Teacher agent output schema rules
├── PROJECT.md             # This file — phase tracker
├── src/                   # Model architecture code
│   └── __init__.py
├── scripts/               # Data generation / training scripts
│   └── __init__.py
├── tests/                 # Unit + integration tests
│   └── test_env_smoke.py
├── build/
│   └── .venv/             # Isolated project virtual environment
├── data/
│   ├── synthetic_docs/    # Raw generated documents
│   ├── distill/           # Teacher-distilled labels
│   └── tokenizer/         # Tokenizer vocab and metadata
├── checkpoints/           # Training checkpoints (.pt)
└── exports/               # Final models, GGUF exports
```

## Quick Start
```powershell
# Activate isolated venv
& "C:\Users\Aum Namaha\Documents\Proj\coPPeR\build\.venv\Scripts\Activate.ps1"

# Run tests
pytest -v

# Set CUDA arch for Blackwell GPU
$env:TORCH_CUDA_ARCH_LIST = "12.0"
```
