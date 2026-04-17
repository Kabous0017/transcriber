"""PreToolUse hook: refuse Write/Edit on any file ending with `.env`.

Secret hygiene for a public repo. Even with .gitignore in place, we never want Claude to
modify .env directly — that file holds the real HF_TOKEN and edits could leak into diffs
or conversation transcripts.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    path = data.get("tool_input", {}).get("file_path", "") or ""
    if path.endswith(".env") or path.endswith("/.env") or path.endswith("\\.env"):
        print(
            "Refusing to modify .env — edit it manually. "
            "Use .env.example for the committed template.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
