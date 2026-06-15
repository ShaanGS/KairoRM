from __future__ import annotations

import re

from ingestion.types import RawFile, SourceFile

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".m": "objc",
    ".scala": "scala",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".lua": "lua",
    ".r": "r",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".clj": "clojure",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".sql": "sql",
    ".proto": "proto",
    ".dockerfile": "dockerfile",
    ".tf": "hcl",
    ".hcl": "hcl",
    ".vue": "vue",
    ".svelte": "svelte",
}

FILENAME_MAP: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Containerfile": "dockerfile",
    "Makefile": "make",
    "GNUmakefile": "make",
    "Rakefile": "ruby",
    "Gemfile": "ruby",
    "Procfile": "yaml",
    "CMakeLists.txt": "cmake",
}

_SHEBANG = re.compile(rb"^#!.*\b(python\d?|node|bash|sh|ruby|perl|lua)\b")


def _from_shebang(file: RawFile) -> str | None:
    try:
        with file.path.open("rb") as f:
            head = f.read(256)
    except OSError:
        return None
    m = _SHEBANG.match(head)
    if not m:
        return None
    name = m.group(1).decode("ascii")
    if name.startswith("python"):
        return "python"
    if name == "node":
        return "javascript"
    if name in ("bash", "sh"):
        return "bash"
    if name == "ruby":
        return "ruby"
    if name == "perl":
        return "perl"
    if name == "lua":
        return "lua"
    return None


def detect(file: RawFile) -> SourceFile:
    """Tag a `RawFile` with a language by extension, filename, or shebang."""
    name = file.path.name
    if name in FILENAME_MAP:
        lang = FILENAME_MAP[name]
        return SourceFile(raw=file, language=lang, parser_name=lang)

    suffix = file.path.suffix.lower()
    if suffix in EXTENSION_MAP:
        lang = EXTENSION_MAP[suffix]
        return SourceFile(raw=file, language=lang, parser_name=lang)

    if not suffix:
        lang = _from_shebang(file)
        if lang is not None:
            return SourceFile(raw=file, language=lang, parser_name=lang)

    return SourceFile(raw=file, language="unknown", parser_name=None)
