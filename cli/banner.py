"""The KairoRM startup banner — minimal cyan ASCII plus a dim tagline."""

from __future__ import annotations

from rich.console import Console

console = Console(stderr=False)

_BANNER = r"""
 _  __     _           ____  __  __
| |/ /__ _(_)_ __ ___ |  _ \|  \/  |
| ' // _` | | '__/ _ \| |_) | |\/| |
| . \ (_| | | | | (_) |  _ <| |  | |
|_|\_\__,_|_|_|  \___/|_| \_\_|  |_|
"""


def print_banner() -> None:
    console.print(f"[cyan]{_BANNER}[/]")
    console.print("[dim]code intelligence engine[/]\n")
