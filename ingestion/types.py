from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T

    def is_ok(self) -> bool:
        return True

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E

    def is_ok(self) -> bool:
        return False

    def unwrap(self) -> T:  # type: ignore[type-var]
        raise RuntimeError(f"unwrap on Err: {self.error!r}")


Result = Ok[T] | Err[E]


@dataclass(frozen=True, slots=True)
class FetchedRepo:
    root: Path
    source_url: str
    commit_sha: str | None
    fetched_at: datetime


@dataclass(frozen=True, slots=True)
class RawFile:
    path: Path
    rel_path: Path
    size_bytes: int
    sha256: str
    oversized: bool


@dataclass(frozen=True, slots=True)
class SourceFile:
    raw: RawFile
    language: str
    parser_name: str | None


@dataclass(frozen=True, slots=True)
class InvalidSourceError:
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class AuthError:
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class NetworkError:
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class RepoTooLargeError:
    source: str
    file_count: int
    limit: int


FetchError = InvalidSourceError | AuthError | NetworkError | RepoTooLargeError


@dataclass(frozen=True, slots=True)
class CodeUnit:
    """One semantic unit extracted from a source file: a function, class, method, or a
    line-block fallback for files we can't parse."""

    file_path: Path
    language: str
    unit_type: str  # "function" | "class" | "method" | "block"
    name: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    raw_source: str
    imports: tuple[str, ...]
    calls: tuple[str, ...]
    parent: str | None = None  # e.g. enclosing class name for a method


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    file_path: Path
    language: str
    unit_type: str
    name: str
    start_line: int
    end_line: int
    content: str
    token_count: int
    imports: tuple[str, ...]
    calls: tuple[str, ...]
    context_header: str


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk: Chunk
    importance_score: float


@dataclass(frozen=True, slots=True)
class RankResult:
    """Output of `ranker.rank`: scored chunks plus the back-edges broken to make the
    call graph acyclic (each `(caller_chunk_id, callee_chunk_id)`)."""

    chunks: list[RankedChunk]
    cycles: list[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    ranked: RankedChunk
    embedding: list[float]


@dataclass(frozen=True, slots=True)
class AgentOutputs:
    """The four parallel agent results. Any field is `None` if that agent failed —
    one agent failing never nulls the others."""

    modules: dict | None
    arch: dict | None
    deps: dict | None
    contributor: dict | None


@dataclass(frozen=True, slots=True)
class SynthesisModule:
    name: str
    path: str
    responsibility: str


@dataclass(frozen=True, slots=True)
class SynthesisEntryPoint:
    name: str
    file: str
    description: str


@dataclass(frozen=True, slots=True)
class ReadingStep:
    """One step in the 'start here' reading order: a file and why to read it."""

    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    repo_id: str
    architecture_summary: str  # 3 sentences max
    modules: list[SynthesisModule]
    key_dependencies: list[str]
    circular_risks: list[str]
    entry_points: list[SynthesisEntryPoint]
    contributor_quickstart: list[str]  # ordered steps, max 6
    complexity_score: int  # 1-10
    generated_at: datetime
    # Deterministic "read these files in this order" guide (entry points first, then the
    # most call-graph-central files). Defaulted so older constructions stay valid.
    reading_order: list[ReadingStep] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompressedContext:
    content: str
    token_count: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class ExportManifest:
    output_dir: Path
    files: list[Path]
    repo_name: str
