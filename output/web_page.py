"""Render the standalone HTML report page from a `SynthesisResult`.

The page itself (markup, styling, motion, Q&A wiring) lives in the static
`report_template.html` asset; this module only serialises the analysis into a JSON
payload and injects it. All rendering happens client-side, so Python stays thin.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources

from ingestion.types import SynthesisResult

_PLACEHOLDER = '"__KAIRO_DATA__"'


@lru_cache(maxsize=1)
def _template() -> str:
    return resources.files("output").joinpath("report_template.html").read_text(encoding="utf-8")


def _payload(result: SynthesisResult, repo_name: str, stats: dict) -> dict:
    return {
        "repo_name": repo_name,
        "architecture_summary": result.architecture_summary,
        "complexity_score": result.complexity_score,
        "modules": [
            {"name": m.name, "path": m.path, "responsibility": m.responsibility}
            for m in result.modules
        ],
        "entry_points": [
            {"name": e.name, "file": e.file, "description": e.description}
            for e in result.entry_points
        ],
        "key_dependencies": list(result.key_dependencies),
        "circular_risks": list(result.circular_risks),
        "contributor_quickstart": list(result.contributor_quickstart),
        "generated_at": result.generated_at.isoformat(),
        "stats": stats,
    }


def build_report_html(result: SynthesisResult, *, repo_name: str, stats: dict) -> str:
    """Inject the analysis payload into the template and return a complete HTML page."""
    data = json.dumps(_payload(result, repo_name, stats), ensure_ascii=False)
    # Neutralise any "</script>" hiding in the data so it can't break out of the tag.
    data = data.replace("</", "<\\/")
    return _template().replace(_PLACEHOLDER, data)
