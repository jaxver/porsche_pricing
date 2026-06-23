"""Check committed notebooks for unsafe outputs and local paths."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable


LOCAL_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\+Users\\+", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\+.*\\+AppData\\+Local\\+Temp", re.IGNORECASE),
    re.compile(r"\\+NAS_[^\s\"']+", re.IGNORECASE),
    re.compile(r"/Users/[^\s\"']+", re.IGNORECASE),
    re.compile(r"/home/[^\s\"']+", re.IGNORECASE),
)


def find_notebooks(paths: Iterable[str | Path]) -> list[Path]:
    notebooks: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            notebooks.extend(sorted(path.rglob("*.ipynb")))
        elif path.suffix == ".ipynb":
            notebooks.append(path)
    return sorted(notebooks, key=lambda path: (len(path.parts), str(path).lower()))


def _contains_local_path(value: object) -> bool:
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in LOCAL_PATH_PATTERNS)
    if isinstance(value, dict):
        return any(_contains_local_path(item) for pair in value.items() for item in pair)
    if isinstance(value, list):
        return any(_contains_local_path(item) for item in value)
    return False


def check_notebook(path: str | Path) -> list[str]:
    notebook_path = Path(path)
    findings: list[str] = []
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    if _contains_local_path(notebook):
        findings.append(f"{notebook_path}: contains local path")

    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        outputs = cell.get("outputs") or []
        keep_output = bool(cell.get("metadata", {}).get("keep_output"))
        if outputs and not keep_output:
            findings.append(f"{notebook_path}: cell {index} has output without keep_output metadata")

        if cell.get("execution_count") is not None:
            findings.append(f"{notebook_path}: cell {index} has execution_count")

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Notebook files or directories to check")
    args = parser.parse_args(argv)

    findings: list[str] = []
    for notebook in find_notebooks(args.paths):
        findings.extend(check_notebook(notebook))

    if findings:
        print("Notebook hygiene check failed:")
        for finding in findings:
            print(f"- {finding}")
        print("\nClear outputs or set cell metadata {\"keep_output\": true} for intentional demo output.")
        return 1

    print("Notebook hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
