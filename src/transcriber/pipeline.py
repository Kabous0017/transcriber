"""WhisperX orchestration — transcribe, align, diarize with explicit VRAM cleanup."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from rich.console import Console

from transcriber.config import Config

console = Console()


class MissingHfTokenError(RuntimeError):
    """Raised when HF_TOKEN is required but not set."""


def _free_vram() -> None:
    """Release GPU memory between pipeline stages.

    Load-bearing on 4 GB GPUs: the three models cannot coexist in VRAM, so each stage
    must fully release its model before the next one loads. Do not remove.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def transcribe(audio_path: Path, config: Config) -> dict[str, Any]:
    """Run the three-stage WhisperX pipeline and return the merged result."""
    if not config.hf_token:
        raise MissingHfTokenError(
            "HF_TOKEN is not set. Copy .env.example to .env and paste a Hugging Face "
            "read token from https://huggingface.co/settings/tokens."
        )

    import whisperx

    audio = whisperx.load_audio(str(audio_path))

    with console.status(f"[cyan]Loading Whisper model ({config.model}, {config.compute_type})..."):
        model = whisperx.load_model(config.model, config.device, compute_type=config.compute_type)

    with console.status("[cyan]Transcribing..."):
        result = model.transcribe(audio, batch_size=config.batch_size, language=config.language)
    del model
    _free_vram()

    with console.status("[cyan]Loading alignment model..."):
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=config.device
        )

    with console.status("[cyan]Aligning word-level timestamps..."):
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, config.device
        )
    del align_model
    _free_vram()

    with console.status("[cyan]Loading pyannote diarization pipeline..."):
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=config.hf_token, device=config.device
        )

    with console.status("[cyan]Diarizing speakers..."):
        diarize_segments = diarize_model(
            audio, min_speakers=config.min_speakers, max_speakers=config.max_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
    del diarize_model
    _free_vram()

    return result
