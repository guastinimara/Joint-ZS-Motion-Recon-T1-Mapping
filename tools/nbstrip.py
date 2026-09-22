#!/usr/bin/env python3
"""Git clean filter: strip outputs from notebooks before they are staged.

Reads a notebook on stdin, writes the stripped notebook to stdout. The file on
disk is left untouched, so outputs stay visible while you work but never reach
a commit.

Enable it once per clone with:

    git config filter.nbstrip.clean "python3 tools/nbstrip.py"
    git config filter.nbstrip.smudge cat
    git config filter.nbstrip.required true
"""

import json
import sys


def strip(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            cell.get("metadata", {}).pop("execution", None)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        sys.stdout.write(raw)
        return
    try:
        nb = json.loads(raw)
    except ValueError:
        # Not valid JSON (merge conflict markers, partial file): pass through
        # untouched rather than corrupting whatever is there.
        sys.stdout.write(raw)
        return
    json.dump(strip(nb), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
