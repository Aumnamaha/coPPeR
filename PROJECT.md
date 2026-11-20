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

### [ ] Phase 4: From-scratch ~1B causal Transformer architecture & unit tests
- Implement causal attention, RoPE, SwiGLU feedforward, RMSNorm
- Unit tests for each architectural component
- Correctness checks against reference implementations

### [ ] Phase 5: Two-stage training (Domain Pretraining + Teacher SFT) on RTX 5070 Ti
- Stage 1: Domain pretraining on synthetic document corpus
- Stage 2: Supervised fine-tuning with teacher labels from `agy`
- Checkpoint management, early stopping, LR scheduling

### [ ] Phase 6: Evaluation & GGUF export for Intel Arc A750 inference
- Perplexity evaluation on held-out test set
- Task-specific benchmarks (extraction accuracy, edit quality)
- Export to GGUF format for CPU/GPU inference on Intel Arc A750

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
