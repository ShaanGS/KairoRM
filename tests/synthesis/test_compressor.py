from __future__ import annotations

import json
from datetime import UTC, datetime

import tiktoken

from ingestion.types import SynthesisEntryPoint, SynthesisModule, SynthesisResult
from synthesis.compressor import compress


def _result(
    *,
    n_modules: int = 1,
    n_deps: int = 1,
    n_quickstart: int = 2,
    summary: str = "A small auth library.",
    responsibility: str = "auth logic",
) -> SynthesisResult:
    return SynthesisResult(
        repo_id="a" * 64,
        architecture_summary=summary,
        modules=[
            SynthesisModule(
                name=f"module_with_a_fairly_long_name_{i}",
                path=f"pkg/module_{i}.py",
                responsibility=responsibility,
            )
            for i in range(n_modules)
        ],
        key_dependencies=[f"dependency_package_number_{i}" for i in range(n_deps)],
        circular_risks=[f"risk_{i}" for i in range(3)],
        entry_points=[
            SynthesisEntryPoint(name="main", file="cli/main.py", description="entry point")
        ],
        contributor_quickstart=[f"step number {i} do the thing" for i in range(n_quickstart)],
        complexity_score=5,
        generated_at=datetime.now(UTC),
    )


def _tiktoken_count(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def test_small_result_returned_as_is() -> None:
    result = _result(n_modules=1, n_deps=1, n_quickstart=2)
    compressed = compress(result, token_budget=3000)
    assert compressed.truncated is False
    assert compressed.token_count < 3000
    # Round-trips as valid JSON with all fields intact.
    data = json.loads(compressed.content)
    assert data["architecture_summary"] == result.architecture_summary
    assert len(data["modules"]) == 1


def test_large_result_is_truncated_under_budget() -> None:
    # Hundreds of modules + deps + quickstart steps blow well past the budget.
    result = _result(n_modules=400, n_deps=300, n_quickstart=50)
    budget = 1000
    compressed = compress(result, token_budget=budget)
    assert compressed.truncated is True
    assert compressed.token_count <= budget


def test_summary_and_entry_points_survive_truncation() -> None:
    result = _result(n_modules=400, n_deps=300, n_quickstart=50)
    compressed = compress(result, token_budget=500)
    data = json.loads(compressed.content)
    assert data["architecture_summary"] == result.architecture_summary
    assert data["entry_points"] == [
        {"name": "main", "file": "cli/main.py", "description": "entry point"}
    ]


def test_token_count_matches_actual_tiktoken() -> None:
    result = _result(n_modules=50, n_deps=40)
    compressed = compress(result, token_budget=800)
    assert compressed.token_count == _tiktoken_count(compressed.content)


def test_top_modules_kept_by_name_length() -> None:
    # Give modules distinct name lengths; after heavy truncation the longest survive.
    result = SynthesisResult(
        repo_id="a" * 64,
        architecture_summary="x",
        modules=[
            SynthesisModule(name="a", path="a.py", responsibility="r" * 200),
            SynthesisModule(name="bb", path="b.py", responsibility="r" * 200),
            SynthesisModule(name="cccccccccc", path="c.py", responsibility="r" * 200),
            SynthesisModule(name="dddddddddddddddddddd", path="d.py", responsibility="r" * 200),
        ],
        key_dependencies=[f"dep_{i}" for i in range(100)],
        circular_risks=[],
        entry_points=[],
        contributor_quickstart=[],
        complexity_score=5,
        generated_at=datetime.now(UTC),
    )
    compressed = compress(result, token_budget=120)
    data = json.loads(compressed.content)
    kept_names = [m["name"] for m in data["modules"]]
    # Whatever survived must be the longest-named ones (proxy for importance).
    if kept_names:
        assert "dddddddddddddddddddd" in kept_names
        assert "a" not in kept_names
