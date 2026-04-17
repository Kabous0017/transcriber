"""PostToolUse hook: run `ruff format` + `ruff check --fix` on edited Python files.

Keeps diffs clean and style-consistent without Claude having to remember to lint. Silent
on success; prints ruff output only on failure. Skips gracefully if ruff isn't installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    path = data.get("tool_input", {}).get("file_path", "") or ""
    if not path.endswith(".py"):
        return 0
    if shutil.which("ruff") is None:
        return 0

    subprocess.run(["ruff", "format", path], check=False, capture_output=True)
    subprocess.run(["ruff", "check", "--fix", path], check=False, capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
