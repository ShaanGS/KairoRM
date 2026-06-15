"""Export a `SynthesisResult` to disk as three artefacts.

Writes a human-readable `architecture.md`, a machine-readable `kairomap.json`, and a
raw `context.txt` (the compressed Q&A context, for piping into other tools) under
`output_dir/<repo_name>/`. Returns an `ExportManifest` listing exactly what was
written; any filesystem failure comes back as an `ExportError`, never an exception.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ingestion.types import (
    CompressedContext,
    Err,
    ExportManifest,
    Ok,
    Result,
    SynthesisResult,
)


@dataclass(frozen=True, slots=True)
class ExportError:
    reason: str


def _markdown(result: SynthesisResult, repo_name: str) -> str:
    lines: list[str] = [f"# {repo_name} — KairoRM Analysis", ""]

    lines += ["## Architecture", "", result.architecture_summary or "_No summary._", ""]
    lines += [f"**Complexity:** {result.complexity_score}/10", ""]

    lines += ["## Modules", ""]
    if result.modules:
        lines += ["| Name | Path | Responsibility |", "| --- | --- | --- |"]
        for m in result.modules:
            lines.append(f"| {m.name} | `{m.path}` | {m.responsibility} |")
    else:
        lines.append("_None identified._")
    lines.append("")

    lines += ["## Entry Points", ""]
    if result.entry_points:
        lines += ["| Name | File | Description |", "| --- | --- | --- |"]
        for e in result.entry_points:
            lines.append(f"| {e.name} | `{e.file}` | {e.description} |")
    else:
        lines.append("_None identified._")
    lines.append("")

    lines += ["## Dependencies", ""]
    if result.key_dependencies:
        lines += [f"- {dep}" for dep in result.key_dependencies]
    else:
        lines.append("_None identified._")
    lines.append("")

    lines += ["## Circular Risks", ""]
    if result.circular_risks:
        lines += [f"- ⚠ {risk}" for risk in result.circular_risks]
    else:
        lines.append("_None detected._")
    lines.append("")

    lines += ["## Contributor Quickstart", ""]
    if result.contributor_quickstart:
        lines += [f"{i}. {step}" for i, step in enumerate(result.contributor_quickstart, 1)]
    else:
        lines.append("_No steps provided._")
    lines.append("")

    return "\n".join(lines)


def export(
    result: SynthesisResult,
    compressed: CompressedContext,
    *,
    output_dir: Path,
    repo_name: str | None = None,
) -> Result[ExportManifest, ExportError]:
    """Write the three artefacts under `output_dir/<repo_name>/`."""
    name = repo_name or result.repo_id[:12]
    try:
        target = output_dir / name
        target.mkdir(parents=True, exist_ok=True)

        md_path = target / "architecture.md"
        json_path = target / "kairomap.json"
        ctx_path = target / "context.txt"

        md_path.write_text(_markdown(result, name), encoding="utf-8")
        # default=str handles the datetime; result stays fully deserializable.
        json_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        ctx_path.write_text(compressed.content, encoding="utf-8")

        return Ok(
            ExportManifest(
                output_dir=target,
                files=[md_path, json_path, ctx_path],
                repo_name=name,
            )
        )
    except OSError as exc:
        return Err(ExportError(reason=str(exc)))
