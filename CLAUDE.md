# Transcriber — Claude Code guidelines

## What this is

Free local + Colab pipeline for transcribing long audio files with speaker diarization, built on
WhisperX (faster-whisper + wav2vec2 alignment + pyannote 3.1). Two entry points: a Colab notebook
(primary) and a local Typer CLI (fallback for privacy-sensitive work).

## Stack

- Python 3.10–3.11 (WhisperX pins torch/CUDA versions incompatible with 3.12+)
- `whisperx` for the orchestration, `typer[all]` for the CLI, `rich` for progress, `python-dotenv`
  for `HF_TOKEN` loading
- `ruff` for linting/formatting (`black`-compatible, 100-char lines)
- Hatchling build backend; installed locally with `pip install -e .`

## Layout

```
src/transcriber/
├── __init__.py
├── __main__.py     # enables `python -m transcriber`
├── cli.py          # Typer app: `transcriber run` and `transcriber doctor`
├── config.py       # dotenv loading, device + compute_type auto-detection
├── audio.py        # ffmpeg preprocessing (16 kHz mono WAV, optional denoise)
├── pipeline.py     # WhisperX three-stage orchestration with VRAM cleanup
└── output.py       # JSON / TXT / SRT / VTT writers
notebooks/transcribe_colab.ipynb   # primary execution path (free T4 GPU)
```

## Run commands

```bash
transcriber doctor                        # sanity-check env (CUDA, ffmpeg, HF_TOKEN)
transcriber run audio/file.mp3            # full pipeline with defaults
transcriber run audio/file.mp3 --min-speakers 5 --max-speakers 10 --denoise
```

## Conventions

- **Type hints required** on public functions. `from __future__ import annotations` at the top of
  every module.
- **Lint before commit**: `ruff format . && ruff check .`. CI (`.github/workflows/lint.yml`)
  enforces this on push/PR.
- **VRAM cleanup between pipeline stages is load-bearing.** The `del model; gc.collect();
  torch.cuda.empty_cache()` pattern in `pipeline.py` exists so the three models don't collide in
  VRAM on a 4 GB GPU. Do NOT "simplify" this away — it breaks the user's hardware target.
- **Never hardcode `HF_TOKEN`.** Read from `os.environ['HF_TOKEN']` (local) or `userdata.get('HF_TOKEN')`
  (Colab). Any new notebook cell must use Colab Secrets, not an inline string.
- **Never read `.env` directly.** It's blocked in `.claude/settings.json` deny-list. If you need
  to inspect config, use `transcriber doctor` which reports token presence without printing it.
- **This is a public GitHub repo.** Before any commit that touches secrets or external URLs,
  run `/security-review`. Never `git add -A` or `git add .` — stage specific files.

## Non-goals

Don't propose these without explicit user ask:

- Unit test suite (the pipeline is a thin wrapper over WhisperX; verify via the run command)
- Docker image (venv + Colab already cover both execution paths)
- Cross-chunk speaker re-clustering for >3h files (workaround documented in README; only build if
  needed)
- Web UI / dashboard
- Project-level `.mcp.json` (user-level MCP config already provides Drive access)

## Verification

After any non-trivial change:

1. `ruff format . && ruff check --fix .`
2. `transcriber doctor` — all green
3. For pipeline changes: run `transcriber run` on a short test file and inspect `outputs/test.txt`
   for correct speaker labels

## MCPs / external tools

- **Google Drive MCP** — available at user level; useful for pulling Colab-generated transcripts
  back into the repo for post-processing.
- **`gh` CLI** — available via Bash for all GitHub ops (PRs, issues, releases). No GitHub MCP
  needed.
