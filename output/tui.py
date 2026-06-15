"""Interactive code-intelligence console (Textual TUI).

Launched after `kairo map` finishes its scan: the analysed codebase becomes a navigable
map alongside a streaming Q&A chat, fully keyboard-driven in the terminal. This is the
one interactive surface — it replaces the old browser/FastAPI server.
"""

from __future__ import annotations

from pathlib import Path

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Input, Markdown, Static

from ingestion.types import CompressedContext, SynthesisResult
from output import qa

_SUGGESTED = (
    "What does this project do?",
    "How is the code structured?",
    "Where should I start contributing?",
)


def build_map_markdown(result: SynthesisResult, stats: dict) -> str:
    """Render the analysed codebase as a markdown document for the map pane."""
    langs = stats.get("languages") or {}
    lang_str = ", ".join(f"{k} {v}" for k, v in list(langs.items())[:4]) or "—"
    lines: list[str] = [
        "# Codebase map",
        "",
        result.architecture_summary or "_No architecture summary._",
        "",
        "## Stats",
        f"- **{stats.get('files', '—')}** files · **{stats.get('chunks', '—')}** chunks",
        f"- languages: {lang_str}",
        f"- complexity: **{result.complexity_score}/10**",
        "",
        "## Modules",
    ]
    if result.modules:
        for m in result.modules:
            lines.append(f"- **{m.name}** — {m.responsibility}")
    else:
        lines.append("_None identified._")

    if result.entry_points:
        lines += ["", "## Entry points"]
        for e in result.entry_points:
            lines.append(f"- `{e.file}` — {e.description}")

    if result.key_dependencies:
        lines += ["", "## Key dependencies", ", ".join(f"`{d}`" for d in result.key_dependencies)]

    if result.circular_risks:
        lines += ["", "## ⚠ Circular dependencies"]
        lines += [f"- {r}" for r in result.circular_risks]

    if result.contributor_quickstart:
        lines += ["", "## Contributor quickstart"]
        for i, step in enumerate(result.contributor_quickstart, 1):
            lines.append(f"{i}. {step}")

    return "\n".join(lines)


class KairoConsole(App):
    """The interactive console shown after a scan completes."""

    CSS = """
    Screen { layers: base; }
    #header {
        height: 3;
        padding: 1 2;
        background: $boost;
        color: $text;
        border-bottom: solid $primary;
    }
    #body { height: 1fr; }
    #map {
        width: 40%;
        padding: 0 2;
        border-right: solid $primary-darken-2;
    }
    #chat { width: 1fr; padding: 0 2; }
    .user-msg { margin: 1 0 0 0; color: $accent; text-style: bold; }
    .bot-msg { margin: 0 0 1 0; padding: 0 0 0 0; }
    Input { dock: bottom; border: tall $primary-darken-2; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "focus_prompt", "Ask"),
    ]

    def __init__(
        self,
        *,
        result: SynthesisResult,
        stats: dict,
        compressed: CompressedContext,
        repo_id: str | None,
        db_path: Path | None,
        repo_name: str,
    ) -> None:
        super().__init__()
        self._result = result
        self._stats = stats
        self._compressed = compressed
        self._repo_id = repo_id
        self._db_path = db_path
        self._repo_name = repo_name
        self._busy = False

    def compose(self) -> ComposeResult:
        files = self._stats.get("files", "—")
        chunks = self._stats.get("chunks", "—")
        # Explicit colours so the bar stays legible on the panel background (theme
        # `dim`/default text can vanish against the lighter header surface).
        sep = ("   ·   ", "#6b7785")
        header = Text.assemble(
            ("◆ KairoRM", "bold #e6edf3"),
            sep,
            (self._repo_name, "bold #4ec9b0"),
            sep,
            (
                f"{files} files · {chunks} chunks · complexity {self._result.complexity_score}/10",
                "#aab4c0",
            ),
        )
        yield Static(header, id="header")
        with Horizontal(id="body"):
            with VerticalScroll(id="map"):
                yield Markdown(build_map_markdown(self._result, self._stats))
            with VerticalScroll(id="chat"):
                intro = (
                    "**Ask anything about this codebase** — answers are grounded in its "
                    "actual code.\n\nTry:\n" + "\n".join(f"- {q}" for q in _SUGGESTED)
                )
                yield Markdown(intro, classes="bot-msg")
        yield Input(
            placeholder="Ask about this codebase…  (Esc to focus, Ctrl+C to quit)", id="prompt"
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    @on(Input.Submitted, "#prompt")
    async def on_submit(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question or self._busy:
            return
        prompt = self.query_one("#prompt", Input)
        prompt.value = ""
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Static(Text(f"❯ {question}"), classes="user-msg"))
        answer = Static("…", markup=False, classes="bot-msg")
        await chat.mount(answer)
        chat.scroll_end(animate=False)
        self._answer(question, answer)

    @work(exclusive=True)
    async def _answer(self, question: str, target: Static) -> None:
        self._busy = True
        chat = self.query_one("#chat", VerticalScroll)
        try:
            chunks, stream = await qa.answer(
                question,
                compressed=self._compressed,
                repo_id=self._repo_id,
                db_path=self._db_path,
            )
            buf = ""
            async for delta in stream:
                buf += delta
                target.update(buf)  # raw text streams fast and is markup-safe
                chat.scroll_end(animate=False)
            footer = f"\n\ngrounded in {len(chunks)} code chunk(s)" if chunks else ""
            target.update(RichMarkdown(buf or "_No answer._"))
            if footer:
                await chat.mount(Static(Text(footer.strip(), style="dim"), classes="bot-msg"))
            chat.scroll_end(animate=False)
        finally:
            self._busy = False
