"""AST-driven semantic unit extraction.

Reads a `SourceFile`, parses it with tree-sitter (via `tree-sitter-language-pack`),
and emits a list of `CodeUnit` records — one per function, class, or method. For
languages we have no grammar for (or `language="unknown"`, or unreadable files),
falls back to 50-line block chunking so no file is ever silently dropped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ingestion.types import CodeUnit, Err, Ok, Result, SourceFile

FALLBACK_BLOCK_LINES = 50

# Per-language node-kind rules. Keys are the language slugs emitted by
# `ingestion/detector.py`. Adding a new language is just a new entry.
@dataclass(frozen=True, slots=True)
class _LangRules:
    function_kinds: frozenset[str]
    class_kinds: frozenset[str]
    import_kinds: frozenset[str]
    call_kinds: frozenset[str]
    # For each call node, which named-child index holds the function expression.
    call_fn_child: int = 0


_LANG_RULES: dict[str, _LangRules] = {
    "python": _LangRules(
        function_kinds=frozenset({"function_definition"}),
        class_kinds=frozenset({"class_definition"}),
        import_kinds=frozenset({"import_statement", "import_from_statement"}),
        call_kinds=frozenset({"call"}),
    ),
    "javascript": _LangRules(
        function_kinds=frozenset(
            {"function_declaration", "function_expression", "method_definition", "arrow_function"}
        ),
        class_kinds=frozenset({"class_declaration"}),
        import_kinds=frozenset({"import_statement"}),
        call_kinds=frozenset({"call_expression"}),
    ),
    "typescript": _LangRules(
        function_kinds=frozenset(
            {"function_declaration", "function_expression", "method_definition", "arrow_function"}
        ),
        class_kinds=frozenset({"class_declaration"}),
        import_kinds=frozenset({"import_statement"}),
        call_kinds=frozenset({"call_expression"}),
    ),
    "tsx": _LangRules(
        function_kinds=frozenset(
            {"function_declaration", "function_expression", "method_definition", "arrow_function"}
        ),
        class_kinds=frozenset({"class_declaration"}),
        import_kinds=frozenset({"import_statement"}),
        call_kinds=frozenset({"call_expression"}),
    ),
    "go": _LangRules(
        function_kinds=frozenset({"function_declaration", "method_declaration"}),
        class_kinds=frozenset({"type_declaration"}),
        import_kinds=frozenset({"import_declaration"}),
        call_kinds=frozenset({"call_expression"}),
    ),
    "rust": _LangRules(
        function_kinds=frozenset({"function_item"}),
        class_kinds=frozenset({"struct_item", "impl_item", "trait_item"}),
        import_kinds=frozenset({"use_declaration"}),
        call_kinds=frozenset({"call_expression"}),
    ),
}


@dataclass(frozen=True, slots=True)
class ParseError:
    file_path: Path
    reason: str


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _named_children(node) -> list:  # noqa: ANN001 — tree_sitter.Node
    return [node.named_child(i) for i in range(node.named_child_count())]


def _node_text(source_bytes: bytes, node) -> str:  # noqa: ANN001
    return source_bytes[node.start_byte() : node.end_byte()].decode("utf-8", errors="replace")


def _node_lines(node) -> tuple[int, int]:  # noqa: ANN001
    return node.start_position().row + 1, node.end_position().row + 1


def _node_name(node, source: str) -> str:  # noqa: ANN001
    """Best-effort name extraction: first child of kind `identifier` (or 'type_identifier')."""
    for child in _named_children(node):
        k = child.kind()
        if k in ("identifier", "type_identifier", "property_identifier", "field_identifier"):
            return _node_text(source, child)
    return "<anonymous>"


def _collect_calls(node, rules: _LangRules, source: str, out: list[str]) -> None:  # noqa: ANN001
    if node.kind() in rules.call_kinds and node.named_child_count() > 0:
        fn = node.named_child(rules.call_fn_child)
        text = _node_text(source, fn)
        # For attribute access (foo.bar() -> "foo.bar"), keep only the final segment.
        # That's what matches simple call-graph wiring; richer resolution comes later.
        leaf = text.split(".")[-1].split("[")[0].strip()
        if leaf and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", leaf):
            out.append(leaf)
    for c in _named_children(node):
        _collect_calls(c, rules, source, out)


def _collect_imports(root, rules: _LangRules, source: str) -> list[str]:  # noqa: ANN001
    out: list[str] = []

    def walk(n) -> None:  # noqa: ANN001
        if n.kind() in rules.import_kinds:
            text = _node_text(source, n).strip()
            if text:
                out.append(text)
        for c in _named_children(n):
            walk(c)

    walk(root)
    return out


def _body_node(node) -> object | None:  # noqa: ANN001
    """Find the body block of a class/function (for finding nested methods)."""
    for c in _named_children(node):
        if c.kind() in ("block", "class_body", "statement_block", "field_declaration_list"):
            return c
    return node


def _line_block_units(
    file_path: Path, language: str, source: str
) -> list[CodeUnit]:
    lines = source.splitlines() or [""]
    units: list[CodeUnit] = []
    for start in range(0, len(lines), FALLBACK_BLOCK_LINES):
        end = min(start + FALLBACK_BLOCK_LINES, len(lines))
        block = "\n".join(lines[start:end])
        units.append(
            CodeUnit(
                file_path=file_path,
                language=language,
                unit_type="block",
                name=f"block_{start + 1}_{end}",
                start_line=start + 1,
                end_line=end,
                raw_source=block,
                imports=(),
                calls=(),
                parent=None,
            )
        )
    return units


def _extract_class(
    cls_node, rules: _LangRules, source: str, file_path: Path, language: str  # noqa: ANN001
) -> list[CodeUnit]:
    class_name = _node_name(cls_node, source)
    start, end = _node_lines(cls_node)
    class_calls: list[str] = []
    _collect_calls(cls_node, rules, source, class_calls)
    cls_unit = CodeUnit(
        file_path=file_path,
        language=language,
        unit_type="class",
        name=class_name,
        start_line=start,
        end_line=end,
        raw_source=_node_text(source, cls_node),
        imports=(),
        calls=tuple(class_calls),
        parent=None,
    )
    units: list[CodeUnit] = [cls_unit]

    body = _body_node(cls_node)
    if body is None:
        return units

    for child in _named_children(body):
        if child.kind() in rules.function_kinds:
            m_start, m_end = _node_lines(child)
            m_calls: list[str] = []
            _collect_calls(child, rules, source, m_calls)
            units.append(
                CodeUnit(
                    file_path=file_path,
                    language=language,
                    unit_type="method",
                    name=_node_name(child, source),
                    start_line=m_start,
                    end_line=m_end,
                    raw_source=_node_text(source, child),
                    imports=(),
                    calls=tuple(m_calls),
                    parent=class_name,
                )
            )
    return units


def _extract_function(
    fn_node, rules: _LangRules, source: str, file_path: Path, language: str  # noqa: ANN001
) -> CodeUnit:
    start, end = _node_lines(fn_node)
    calls: list[str] = []
    _collect_calls(fn_node, rules, source, calls)
    return CodeUnit(
        file_path=file_path,
        language=language,
        unit_type="function",
        name=_node_name(fn_node, source),
        start_line=start,
        end_line=end,
        raw_source=_node_text(source, fn_node),
        imports=(),
        calls=tuple(calls),
        parent=None,
    )


def _get_parser(language: str):  # noqa: ANN001
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:
        return None
    try:
        return get_parser(language)
    except Exception:
        return None


def parse(source_file: SourceFile) -> Result[list[CodeUnit], ParseError]:
    """Parse a `SourceFile` into a list of `CodeUnit` records.

    Unknown languages, missing grammars, or read failures all fall back to 50-line
    block chunking. AST-level errors inside an otherwise parseable file are tolerated
    (tree-sitter returns a partial tree with ERROR nodes; we ignore those and keep
    whatever well-formed units we found).
    """
    path = source_file.raw.path
    language = source_file.language

    text = _read_text(path)
    if text is None:
        return Err(ParseError(file_path=path, reason="could not read file"))

    rules = _LANG_RULES.get(language)
    if language == "unknown" or rules is None:
        return Ok(_line_block_units(path, language, text))

    parser = _get_parser(source_file.parser_name or language)
    if parser is None:
        return Ok(_line_block_units(path, language, text))

    try:
        tree = parser.parse(text)
    except Exception:
        return Ok(_line_block_units(path, language, text))

    root = tree.root_node()
    source_bytes = text.encode("utf-8")
    imports = tuple(_collect_imports(root, rules, source_bytes))

    units: list[CodeUnit] = []

    def visit(node) -> None:  # noqa: ANN001
        kind = node.kind()
        if kind in rules.class_kinds:
            units.extend(_extract_class(node, rules, source_bytes, path, language))
            return
        if kind in rules.function_kinds:
            units.append(_extract_function(node, rules, source_bytes, path, language))
            return
        for c in _named_children(node):
            visit(c)

    visit(root)

    if not units:
        return Ok(_line_block_units(path, language, text))

    # Stamp imports on every unit (cheap, lets agents see file-level deps per chunk).
    units = [
        CodeUnit(
            file_path=u.file_path,
            language=u.language,
            unit_type=u.unit_type,
            name=u.name,
            start_line=u.start_line,
            end_line=u.end_line,
            raw_source=u.raw_source,
            imports=imports,
            calls=u.calls,
            parent=u.parent,
        )
        for u in units
    ]
    return Ok(units)
