"""Runtime configuration: dotenv loading, device detection, sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def detect_vram_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        return None


def default_compute_type(device: str) -> str:
    if device == "cpu":
        return "int8"
    vram = detect_vram_gb()
    # <6 GB GPUs can't fit large-v3 in float16 alongside desktop VRAM overhead.
    if vram is not None and vram < 6:
        return "int8"
    return "float16"


@dataclass(frozen=True)
class Config:
    """All tunables for a single transcription run."""

    model: str = "large-v3"
    compute_type: str = field(default_factory=lambda: default_compute_type(detect_device()))
    device: str = field(default_factory=detect_device)
    language: str = "en"
    min_speakers: int = 2
    max_speakers: int = 10
    batch_size: int = 4
    denoise: bool = False
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    formats: tuple[str, ...] = ("json", "txt", "srt")

    @property
    def hf_token(self) -> str | None:
        return os.environ.get("HF_TOKEN")
