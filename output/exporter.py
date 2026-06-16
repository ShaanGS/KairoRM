"""Export a `SynthesisResult` to disk as three artefacts.

Writes a human-readable `architecture.md`, a machine-readable `kairomap.json`, and a
raw `context.txt` (the compressed Q&A context, for piping into other tools) under
`output_dir/<repo_name>/`. Returns an `ExportManifest` listing exactly what was
written; any filesystem failure comes back as an `ExportError`, never an exception.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
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

log = logging.getLogger("kairo")

MAX_GRAPH_NODES = 20


@dataclass(frozen=True, slots=True)
class ExportError:
    reason: str


def _node_id(name: str) -> str:
    """A mermaid/dot-safe node id derived from a module name."""
    return re.sub(r"\W", "_", name) or "m"


def generate_mermaid(result: SynthesisResult, max_nodes: int = MAX_GRAPH_NODES) -> str:
    """A mermaid `flowchart TD` of module→module dependencies.

    Nodes are the top modules by importance (`result.modules` is already importance-
    ordered); edges are the cross-module dependencies among them. Labels are clean
    module names — no paths, no extensions.
    """
    keep = [m.name for m in result.modules[:max_nodes]]
    keepset = set(keep)
    lines = ["flowchart TD"]
    for name in keep:
        lines.append(f'    {_node_id(name)}["{name}"]')
    seen: set[tuple[str, str]] = set()
    for src, dst in result.module_graph:
        if src in keepset and dst in keepset and (src, dst) not in seen:
            seen.add((src, dst))
            lines.append(f"    {_node_id(src)} --> {_node_id(dst)}")
    return "\n".join(lines)


def _export_graphviz(result: SynthesisResult, target: Path) -> None:
    """Write architecture.dot and render architecture.png via networkx + pydot.

    Best-effort: if pydot or the graphviz `dot` binary is missing, skip silently with a
    logfile line — never crash the export. The .dot is always written when pydot exists.
    """
    try:
        # pydot/pyparsing emit a noisy DeprecationWarning on import — keep it off-screen.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import networkx as nx
            from networkx.drawing.nx_pydot import write_dot
    except Exception as exc:  # pydot/pyparsing not installed
        log.info("graphviz export skipped — pydot unavailable (%s)", exc)
        return

    keep = [m.name for m in result.modules[:MAX_GRAPH_NODES]]
    keepset = set(keep)
    g = nx.DiGraph()
    g.graph["bgcolor"] = "#1A1A14"
    for name in keep:
        g.add_node(
            _node_id(name),
            label=name,
            style="filled",
            shape="box",
            fillcolor="#7CB87A",
            fontcolor="#1A1A14",
            color="#3D3D2E",
            fontname="monospace",
        )
    for src, dst in result.module_graph:
        if src in keepset and dst in keepset:
            g.add_edge(_node_id(src), _node_id(dst), color="#3D3D2E")

    try:
        write_dot(g, str(target / "architecture.dot"))
    except Exception as exc:
        log.info("graphviz export skipped — could not write .dot (%s)", exc)
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pydot

            graphs = pydot.graph_from_dot_file(str(target / "architecture.dot"))
            graphs[0].write_png(str(target / "architecture.png"))
    except Exception as exc:  # graphviz `dot` binary missing, etc.
        log.info("PNG render skipped — graphviz binary unavailable (%s)", exc)


def _markdown(result: SynthesisResult, repo_name: str) -> str:
    lines: list[str] = [f"# {repo_name} — KairoRM Analysis", ""]

    # Mermaid block first so GitHub renders the dependency graph at the top.
    if result.modules:
        lines += ["```mermaid", generate_mermaid(result), "```", ""]
    if result.graph_summary:
        lines += [f"_{result.graph_summary}_", ""]

    lines += ["## Architecture", "", result.architecture_summary or "_No summary._", ""]
    lines += [f"**Complexity:** {result.complexity_score}/10", ""]

    if result.reading_order:
        lines += ["## Start Here", "", "New to this codebase? Read these files in order:", ""]
        for i, step in enumerate(result.reading_order, 1):
            lines.append(f"{i}. `{step.path}` — {step.reason}")
        lines.append("")

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
        mmd_path = target / "architecture.mmd"

        md_path.write_text(_markdown(result, name), encoding="utf-8")
        # default=str handles the datetime; result stays fully deserializable.
        json_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        ctx_path.write_text(compressed.content, encoding="utf-8")
        mmd_path.write_text(generate_mermaid(result), encoding="utf-8")

        files = [md_path, json_path, ctx_path, mmd_path]
        # Best-effort GraphViz .dot/.png — never blocks the export if tooling is absent.
        _export_graphviz(result, target)
        files += [
            p for p in (target / "architecture.dot", target / "architecture.png") if p.exists()
        ]

        return Ok(ExportManifest(output_dir=target, files=files, repo_name=name))
    except OSError as exc:
        return Err(ExportError(reason=str(exc)))
