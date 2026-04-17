"""Write transcription results to JSON / TXT / SRT / VTT with speaker labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def write_json(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def write_txt(result: dict[str, Any], path: Path) -> None:
    """Human-readable transcript, grouped by contiguous speaker turn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []
    turn_start: float = 0.0

    def flush() -> None:
        if buffer and current_speaker is not None:
            lines.append(f"[{turn_start:.1f}s] {current_speaker}: {' '.join(buffer).strip()}")

    for segment in result.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            flush()
            buffer = []
            current_speaker = speaker
            turn_start = float(segment.get("start", 0.0))
        buffer.append(text)

    flush()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_subtitle(result: dict[str, Any], path: Path, vtt: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "." if vtt else ","
    lines: list[str] = ["WEBVTT", ""] if vtt else []
    for i, segment in enumerate(result.get("segments", []), start=1):
        start = _fmt_timestamp(float(segment.get("start", 0.0)), sep=sep)
        end = _fmt_timestamp(float(segment.get("end", 0.0)), sep=sep)
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        if vtt:
            lines += [f"{start} --> {end}", f"{speaker}: {text}", ""]
        else:
            lines += [str(i), f"{start} --> {end}", f"{speaker}: {text}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srt(result: dict[str, Any], path: Path) -> None:
    _write_subtitle(result, path, vtt=False)


def write_vtt(result: dict[str, Any], path: Path) -> None:
    _write_subtitle(result, path, vtt=True)


WRITERS = {
    "json": write_json,
    "txt": write_txt,
    "srt": write_srt,
    "vtt": write_vtt,
}


def write_all(result: dict[str, Any], base_path: Path, formats: tuple[str, ...]) -> list[Path]:
    """Write requested formats; returns the list of written paths."""
    written: list[Path] = []
    for fmt in formats:
        writer = WRITERS.get(fmt)
        if writer is None:
            raise ValueError(f"Unknown output format: {fmt!r}. Supported: {list(WRITERS)}")
        out = base_path.with_suffix(f".{fmt}")
        writer(result, out)
        written.append(out)
    return written
