from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.types import RawFile, SourceFile
from parsing.ast_parser import parse


def _source_file(path: Path, language: str = "python") -> SourceFile:
    return SourceFile(
        raw=RawFile(
            path=path,
            rel_path=Path(path.name),
            size_bytes=path.stat().st_size,
            sha256="0" * 64,
            oversized=False,
        ),
        language=language,
        parser_name=language if language != "unknown" else None,
    )


def test_single_python_function(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("def foo(x):\n    bar(x)\n    return baz(x)\n")
    result = parse(_source_file(p))
    assert result.is_ok()
    units = result.unwrap()
    assert len(units) == 1
    u = units[0]
    assert u.unit_type == "function"
    assert u.name == "foo"
    assert u.start_line == 1
    assert u.end_line == 3
    assert set(u.calls) == {"bar", "baz"}


def test_class_with_two_methods(tmp_path: Path) -> None:
    p = tmp_path / "auth.py"
    p.write_text(
        "class Auth:\n"
        "    def verify(self, token):\n"
        "        return self.lookup(token)\n"
        "    def lookup(self, token):\n"
        "        return db_query(token)\n"
    )
    result = parse(_source_file(p))
    assert result.is_ok()
    units = result.unwrap()
    types = sorted(u.unit_type for u in units)
    assert types == ["class", "method", "method"]
    names = {u.name for u in units}
    assert names == {"Auth", "verify", "lookup"}
    methods = [u for u in units if u.unit_type == "method"]
    for m in methods:
        assert m.parent == "Auth"
    verify = next(u for u in units if u.name == "verify")
    assert "lookup" in verify.calls


def test_unknown_language_falls_back_to_line_blocks(tmp_path: Path) -> None:
    p = tmp_path / "weird.xyz"
    p.write_text("\n".join(f"line {i}" for i in range(120)))
    result = parse(_source_file(p, language="unknown"))
    assert result.is_ok()
    units = result.unwrap()
    # 120 lines, 50-line blocks → 3 blocks
    assert len(units) == 3
    assert all(u.unit_type == "block" for u in units)
    assert units[0].start_line == 1
    assert units[0].end_line == 50
    assert units[-1].end_line == 120


def test_syntax_errors_are_tolerated(tmp_path: Path) -> None:
    # Broken Python — unmatched paren in one function, valid function after.
    p = tmp_path / "broken.py"
    p.write_text(
        "def busted(:\n"
        "    return 1\n"
        "\n"
        "def good(x):\n"
        "    return x + 1\n"
    )
    result = parse(_source_file(p))
    assert result.is_ok()  # never raises, never returns Err
    units = result.unwrap()
    # We expect at least the well-formed function to come through.
    assert any(u.name == "good" and u.unit_type == "function" for u in units)


def test_empty_python_file_falls_back_to_blocks(tmp_path: Path) -> None:
    p = tmp_path / "empty.py"
    p.write_text("# just a comment\n")
    result = parse(_source_file(p))
    assert result.is_ok()
    # No functions/classes → block fallback rather than empty list
    units = result.unwrap()
    assert len(units) >= 1
    assert units[0].unit_type == "block"


@pytest.mark.parametrize("missing_path", ["does_not_exist.py"])
def test_unreadable_file_returns_err(tmp_path: Path, missing_path: str) -> None:
    p = tmp_path / missing_path
    sf = SourceFile(
        raw=RawFile(
            path=p, rel_path=Path(p.name), size_bytes=0, sha256="0" * 64, oversized=False
        ),
        language="python",
        parser_name="python",
    )
    result = parse(sf)
    assert not result.is_ok()
