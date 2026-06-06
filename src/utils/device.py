"""Centralized CUDA / device detection for the pipeline."""

from __future__ import annotations

import os
from typing import Tuple

import torch


def setup_cuda() -> None:
    """Configure environment for GPU-accelerated backends (gcastle, PyTorch)."""
    os.environ.setdefault("CASTLE_BACKEND", "pytorch")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    setup_cuda()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_xgboost_device() -> Tuple[str, str]:
    """
    Return (tree_method, device) for XGBoost.

    Uses GPU histogram training when CUDA is available.
    """
    if torch.cuda.is_available():
        return "hist", "cuda"
    return "hist", "cpu"


def device_summary() -> str:
    """Human-readable device status line."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA available — {name} (torch {torch.__version__}, cuda {torch.version.cuda})"
    return f"CUDA unavailable — running on CPU (torch {torch.__version__})"
