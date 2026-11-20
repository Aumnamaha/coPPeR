"""Environment smoke tests for Doc-Ai Phase 1."""

import os


def test_torch_cuda_available():
    import torch

    assert torch.cuda.is_available(), "CUDA not available"


def test_tokenizers_import():
    import tokenizers

    assert hasattr(tokenizers, "Tokenizer")


def test_datasets_import():
    import datasets

    assert hasattr(datasets, "load_dataset")


def test_safetensors_import():
    import safetensors

    assert hasattr(safetensors, "safe_open")


def test_accelerate_import():
    import accelerate

    assert hasattr(accelerate, "Accelerator")


def test_tqdm_import():
    from tqdm import tqdm

    assert callable(tqdm)


def test_pymupdf_import():
    import fitz

    assert hasattr(fitz, "Document")


def test_agents_md_exists_no_bom():
    path = os.path.join(os.path.dirname(__file__), "..", "AGENTS.md")
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) > 0, "AGENTS.md is empty"
    assert not data.startswith(b"\xef\xbb\xbf"), "AGENTS.md has UTF-8 BOM"


def test_project_md_exists_no_bom():
    path = os.path.join(os.path.dirname(__file__), "..", "PROJECT.md")
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) > 0, "PROJECT.md is empty"
    assert not data.startswith(b"\xef\xbb\xbf"), "PROJECT.md has UTF-8 BOM"
