from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from ingestion.types import (
    CompressedContext,
    ExportManifest,
    SynthesisEntryPoint,
    SynthesisModule,
    SynthesisResult,
)
from output.exporter import export


def _result() -> SynthesisResult:
    return SynthesisResult(
        repo_id="a" * 64,
        architecture_summary="A small layered service.",
        modules=[SynthesisModule(name="cli", path="cli/main.py", responsibility="entry")],
        key_dependencies=["click", "rich"],
        circular_risks=["auth -> db -> auth"],
        entry_points=[SynthesisEntryPoint(name="main", file="cli/main.py", description="cmd")],
        contributor_quickstart=["clone", "install", "test"],
        complexity_score=4,
        generated_at=datetime.now(UTC),
    )


def _compressed() -> CompressedContext:
    return CompressedContext(content='{"hello": "world"}', token_count=7, truncated=False)


def test_export_creates_three_files(tmp_path: Path) -> None:
    result = export(_result(), _compressed(), output_dir=tmp_path, repo_name="myrepo")
    assert result.is_ok()
    target = tmp_path / "myrepo"
    assert (target / "architecture.md").exists()
    assert (target / "kairomap.json").exists()
    assert (target / "context.txt").exists()


def test_architecture_md_has_repo_name_header(tmp_path: Path) -> None:
    export(_result(), _compressed(), output_dir=tmp_path, repo_name="myrepo")
    md = (tmp_path / "myrepo" / "architecture.md").read_text()
    assert md.startswith("# myrepo — KairoRM Analysis")
    # Human-readable sections, not raw JSON.
    assert "## Architecture" in md
    assert "## Modules" in md
    assert "## Contributor Quickstart" in md


def test_kairomap_json_is_valid_and_deserializable(tmp_path: Path) -> None:
    export(_result(), _compressed(), output_dir=tmp_path, repo_name="myrepo")
    raw = (tmp_path / "myrepo" / "kairomap.json").read_text()
    data = json.loads(raw)  # raises if invalid
    assert data["repo_id"] == "a" * 64
    assert data["complexity_score"] == 4
    assert data["modules"][0]["name"] == "cli"


def test_context_txt_matches_compressed_content(tmp_path: Path) -> None:
    compressed = _compressed()
    export(_result(), compressed, output_dir=tmp_path, repo_name="myrepo")
    content = (tmp_path / "myrepo" / "context.txt").read_text()
    assert content == compressed.content


def test_manifest_lists_three_existing_files(tmp_path: Path) -> None:
    result = export(_result(), _compressed(), output_dir=tmp_path, repo_name="myrepo")
    manifest = result.unwrap()
    assert isinstance(manifest, ExportManifest)
    assert manifest.repo_name == "myrepo"
    assert len(manifest.files) == 3
    assert all(p.exists() for p in manifest.files)
