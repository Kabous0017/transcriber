"""PreToolUse hook: block Bash commands that contain a raw Hugging Face token.

Stops Claude from ever putting a literal `hf_...` value into a shell command (where it
would land in shell history and conversation transcripts). Legitimate uses should rely on
the `HF_TOKEN` environment variable loaded from `.env`.
"""

from __future__ import annotations

import json
import re
import sys

HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    cmd = data.get("tool_input", {}).get("command", "") or ""
    if HF_TOKEN_PATTERN.search(cmd):
        print(
            "Detected a raw Hugging Face token in the Bash command. "
            "Use the $HF_TOKEN env var (loaded from .env) instead of embedding the value.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
