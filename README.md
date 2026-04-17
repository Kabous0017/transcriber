# Transcriber

Free, local + Colab speech transcription with speaker diarization for long audio files. Built on
[WhisperX](https://github.com/m-bain/whisperX) (faster-whisper + wav2vec2 alignment + pyannote
3.1).

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kabous0017/transcriber/blob/main/notebooks/transcribe_colab.ipynb)

## Why this exists

Paid transcription services cap free-tier files at 30 minutes and require subscriptions for long
recordings. This repo is a free alternative with:

- **No file-length cap** — transcribe 6-hour recordings in one pass
- **Accurate speaker diarization** — pyannote 3.1, the current best open model
- **Two paths** — Colab (fast, free T4 GPU) for convenience; local for privacy
- **No telemetry, no account, no limits** once set up

## Quickstart — Colab (recommended)

For your first run, plan on ~5 minutes of setup plus whatever it takes Colab to transcribe your
file (roughly 15–25 minutes for a 6-hour recording on a free T4).

1. **Accept pyannote model terms** (once, free). Log in to Hugging Face, then click **"Agree"** on all three:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
2. **Create a Hugging Face read token** at <https://huggingface.co/settings/tokens>. Name it
   something like `transcriber-colab`.
3. **Open the notebook** via the "Open in Colab" badge above.
4. **Runtime → Change runtime type → T4 GPU**.
5. **Add the token to Colab Secrets** — click the 🔑 icon in the sidebar, add a secret named
   `HF_TOKEN`, paste the value, toggle "Notebook access" on.
6. **Upload your audio** to Google Drive, update `AUDIO_FILE` in the config cell.
7. **Runtime → Run all**. Transcripts land in the Drive folder you picked.

## Quickstart — Local

For privacy-sensitive recordings or offline use.

**Prereqs**: Python 3.10 or 3.11, [ffmpeg](https://ffmpeg.org/download.html) on `PATH`, ideally an
NVIDIA GPU with CUDA (CPU works but is slow).

```bash
git clone https://github.com/Kabous0017/transcriber.git
cd transcriber
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -e .
cp .env.example .env                   # then fill in HF_TOKEN
transcriber doctor                     # sanity-check the environment
transcriber run samples/your-file.mp3 --min-speakers 2 --max-speakers 10
```

Input conventions:

- `samples/` — short clips for testing the pipeline (gitignored)
- `audio/` — real work input (gitignored)
- `outputs/` — transcripts land here as `.json`, `.txt`, and `.srt` (gitignored)

## Hardware notes

The pipeline loads one model at a time and clears VRAM between stages, so you don't need memory
for all three simultaneously. Rough VRAM needs for the transcription step:

| Model      | float16  | int8     |
| ---------- | -------- | -------- |
| large-v3   | ~5.0 GB  | ~3.0 GB  |
| medium     | ~2.5 GB  | ~1.5 GB  |
| small      | ~1.0 GB  | ~0.5 GB  |

Pyannote diarization adds another ~1–2 GB during its own stage. On a 4 GB GPU use
`--compute-type int8` and close other GPU-using apps (browsers, Discord) before running.

## Output formats

- **`transcript.json`** — full result, including word-level timestamps and per-word speaker labels.
  Use this for any downstream processing.
- **`transcript.txt`** — human-readable log with `[12.3s] SPEAKER_01: ...` lines, grouped by
  speaker turn.
- **`transcript.srt`** — standard subtitle file with speaker prefixes.

## How it works

```
audio file → ffmpeg (16 kHz mono WAV)
         → faster-whisper  (transcription)
         → wav2vec2 align  (word-level timestamps)
         → pyannote 3.1    (diarization)
         → assign speakers per word
         → write JSON / TXT / SRT
```

Each model is loaded, used, then explicitly freed before the next one loads — which is what makes
the pipeline fit on a 4 GB GPU.

## Troubleshooting

- **`401 Unauthorized` from Hugging Face** — you haven't accepted the terms on the two pyannote
  model pages linked in the Quickstart. Accept them, then retry.
- **`CUDA out of memory`** — drop to `--compute-type int8` or `--model medium`, lower
  `--batch-size` to 2, and close GPU-using desktop apps.
- **Speaker labels inconsistent across a file longer than ~3 hours** — known pyannote limitation
  on very long files. Either set `--max-speakers` tightly (e.g. if you know there are exactly 3,
  use `--min-speakers 3 --max-speakers 3`), or split the file in half and re-cluster manually.
- **Garbled transcription** — try `--denoise` or check that the source audio isn't already heavily
  compressed (64 kbps MP3 hurts both transcription and diarization).

Run `transcriber doctor` anytime to check Python, CUDA, ffmpeg, and token status.

## Licence

MIT — see [LICENSE](./LICENSE).
