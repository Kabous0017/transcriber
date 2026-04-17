"""Audio preprocessing via ffmpeg — convert anything to 16 kHz mono WAV."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FfmpegMissingError(RuntimeError):
    """Raised when ffmpeg is not on PATH."""


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise FfmpegMissingError(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/download.html "
            "and ensure the binary directory is on your system PATH."
        )
    return path


def preprocess(input_path: Path, out_path: Path, denoise: bool = False) -> Path:
    """Convert any audio/video file to 16 kHz mono PCM WAV.

    Whisper was trained on 16 kHz mono audio and pyannote expects the same format, so
    normalising up front avoids subtle issues with variable bitrates or video containers.
    """
    ffmpeg = _require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [ffmpeg, "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000"]
    if denoise:
        # Light denoise: gentle bandpass + FFT noise reduction. Heavier filters hurt Whisper.
        cmd += ["-af", "highpass=f=80, lowpass=f=8000, afftdn=nf=-25"]
    cmd += ["-c:a", "pcm_s16le", str(out_path)]

    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
