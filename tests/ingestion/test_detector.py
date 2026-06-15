from __future__ import annotations

from pathlib import Path

from ingestion.detector import detect
from ingestion.types import RawFile


def _raw(path: Path) -> RawFile:
    return RawFile(
        path=path,
        rel_path=Path(path.name),
        size_bytes=path.stat().st_size,
        sha256="0" * 64,
        oversized=False,
    )


def test_python_extension(tmp_path: Path) -> None:
    p = tmp_path / "main.py"
    p.write_text("print('hi')\n")
    assert detect(_raw(p)).language == "python"


def test_dockerfile_no_extension(tmp_path: Path) -> None:
    p = tmp_path / "Dockerfile"
    p.write_text("FROM python:3.11\n")
    assert detect(_raw(p)).language == "dockerfile"


def test_shebang_python(tmp_path: Path) -> None:
    p = tmp_path / "script"
    p.write_text("#!/usr/bin/env python3\nprint('hi')\n")
    assert detect(_raw(p)).language == "python"


def test_shebang_bash(tmp_path: Path) -> None:
    p = tmp_path / "run"
    p.write_text("#!/bin/bash\necho hi\n")
    assert detect(_raw(p)).language == "bash"


def test_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "thing.xyz"
    p.write_text("???\n")
    sf = detect(_raw(p))
    assert sf.language == "unknown"
    assert sf.parser_name is None


def test_typescript(tmp_path: Path) -> None:
    p = tmp_path / "a.ts"
    p.write_text("const x = 1;\n")
    assert detect(_raw(p)).language == "typescript"
