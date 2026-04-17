"""Typer CLI — `transcriber run` and `transcriber doctor`."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from transcriber.audio import FfmpegMissingError, preprocess
from transcriber.config import Config, default_compute_type, detect_device, detect_vram_gb
from transcriber.output import WRITERS, write_all
from transcriber.pipeline import MissingHfTokenError, transcribe

app = typer.Typer(
    help="Local + Colab speech transcription with speaker diarization.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def run(
    audio_file: Path = typer.Argument(..., exists=True, dir_okay=False, resolve_path=True),
    model: str = typer.Option("large-v3", help="Whisper model size: large-v3, medium, small."),
    compute_type: str = typer.Option(
        "",
        help="float16 | int8 | float32. Auto-detected if empty (int8 on CPU or <6 GB GPU).",
    ),
    language: str = typer.Option("en", help="Language code; set explicitly for long files."),
    min_speakers: int = typer.Option(2, help="Minimum expected speakers."),
    max_speakers: int = typer.Option(10, help="Maximum expected speakers."),
    batch_size: int = typer.Option(4, help="Whisper batch size. Drop to 2 if OOM on GPU."),
    denoise: bool = typer.Option(False, "--denoise/--no-denoise", help="Light ffmpeg denoise."),
    output_dir: Path = typer.Option(Path("outputs"), help="Where to write transcripts."),
    formats: str = typer.Option("json,txt,srt", help="Comma-separated: json,txt,srt,vtt."),
) -> None:
    """Transcribe AUDIO_FILE and write diarized outputs."""
    fmt_tuple = tuple(f.strip() for f in formats.split(",") if f.strip())
    unknown = [f for f in fmt_tuple if f not in WRITERS]
    if unknown:
        console.print(f"[red]Unknown format(s): {unknown}. Supported: {list(WRITERS)}[/red]")
        raise typer.Exit(2)

    device = detect_device()
    config = Config(
        model=model,
        compute_type=compute_type or default_compute_type(device),
        device=device,
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        batch_size=batch_size,
        denoise=denoise,
        output_dir=output_dir,
        formats=fmt_tuple,
    )

    console.print(
        f"[bold]Transcribing[/bold] {audio_file.name} "
        f"(model={config.model}, compute={config.compute_type}, device={config.device})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prepped = output_dir / f"{audio_file.stem}.16k.wav"
    try:
        preprocess(audio_file, prepped, denoise=config.denoise)
    except FfmpegMissingError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    try:
        result = transcribe(prepped, config)
    except MissingHfTokenError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    base = output_dir / audio_file.stem
    written = write_all(result, base, config.formats)
    console.print("[green]Done.[/green] Wrote:")
    for path in written:
        console.print(f"  - {path}")


@app.command()
def doctor() -> None:
    """Check Python, CUDA, ffmpeg, and HF_TOKEN status."""
    table = Table(title="Transcriber environment check", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Result")

    table.add_row("Python", f"{sys.version.split()[0]}")

    device = detect_device()
    table.add_row("Torch device", device)
    vram = detect_vram_gb()
    table.add_row("VRAM", f"{vram:.1f} GB" if vram else "n/a")
    table.add_row("Default compute_type", default_compute_type(device))

    ffmpeg_path = shutil.which("ffmpeg")
    table.add_row("ffmpeg", ffmpeg_path or "[red]missing[/red]")

    # Load Config to pick up .env, but never print the token value.
    config = Config()
    table.add_row("HF_TOKEN", "[green]set[/green]" if config.hf_token else "[red]missing[/red]")

    console.print(table)
    if not ffmpeg_path or not config.hf_token:
        raise typer.Exit(1)
