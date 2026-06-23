import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_notebook_hygiene import check_notebook, find_notebooks


def write_notebook(path: Path, cells: list[dict], metadata: dict | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": metadata or {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )


def code_cell(source: str, outputs: list[dict] | None = None, metadata: dict | None = None, execution_count=None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": metadata or {},
        "outputs": outputs or [],
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def test_clean_notebook_has_no_findings(tmp_path):
    notebook = tmp_path / "clean.ipynb"
    write_notebook(notebook, [markdown_cell("# Demo"), code_cell("print('safe')")])

    assert check_notebook(notebook) == []


def test_notebook_output_requires_explicit_keep_output(tmp_path):
    notebook = tmp_path / "with_output.ipynb"
    write_notebook(
        notebook,
        [
            code_cell(
                "print('demo')",
                outputs=[{"output_type": "stream", "name": "stdout", "text": "demo\n"}],
                execution_count=1,
            )
        ],
    )

    findings = check_notebook(notebook)

    assert any("output without keep_output" in finding for finding in findings)
    assert any("execution_count" in finding for finding in findings)


def test_keep_output_cell_is_allowed_but_execution_count_is_not(tmp_path):
    notebook = tmp_path / "kept_output.ipynb"
    write_notebook(
        notebook,
        [
            code_cell(
                "display(chart)",
                outputs=[{"output_type": "display_data", "data": {"text/plain": "chart"}, "metadata": {}}],
                metadata={"keep_output": True},
                execution_count=3,
            )
        ],
    )

    findings = check_notebook(notebook)

    assert not any("output without keep_output" in finding for finding in findings)
    assert any("execution_count" in finding for finding in findings)


@pytest.mark.parametrize(
    "path_text",
    [
        r"C:\\Users\\USER\\project\\data.xlsx",
        r"\\\\NAS_REDACTED\\Software\\Python\\Elferspot_prod",
        "/Users/USER/project/data.xlsx",
        "/home/USER/project/data.xlsx",
        r"C:\\Users\\USER\\AppData\\Local\\Temp\\opencode",
    ],
)
def test_notebook_local_paths_are_reported(tmp_path, path_text):
    notebook = tmp_path / "paths.ipynb"
    write_notebook(notebook, [code_cell(f"DATA_PATH = {path_text!r}")])

    findings = check_notebook(notebook)

    assert any("local path" in finding for finding in findings)


def test_find_notebooks_walks_directories(tmp_path):
    write_notebook(tmp_path / "one.ipynb", [])
    nested = tmp_path / "nested"
    nested.mkdir()
    write_notebook(nested / "two.ipynb", [])

    assert [path.name for path in find_notebooks([tmp_path])] == ["one.ipynb", "two.ipynb"]


def test_cli_returns_nonzero_for_dirty_notebook(tmp_path):
    notebook = tmp_path / "dirty.ipynb"
    write_notebook(notebook, [code_cell(r"ROOT = 'C:\\Users\\USER'")])

    result = subprocess.run(
        [sys.executable, "scripts/check_notebook_hygiene.py", str(notebook)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "local path" in result.stdout
