"""Shared terminal colour palette — warm 'Stardew' olive / sage / gold.

One accent (sage green) and one highlight (warm gold); everything else cream or muted.
Used as Rich markup hex (`f"[{ACCENT}]…[/]"`) and as Textual CSS colours.
"""

from __future__ import annotations

BG = "#1A1A14"  # dark warm olive-black — app background
SURFACE = "#222218"  # panel / header backgrounds
BORDER = "#3D3D2E"  # panel borders — dim, earthy
TEXT = "#E8E4D0"  # main text — warm cream
MUTED = "#7A7A5A"  # secondary text, labels, quiet errors
ACCENT = "#7CB87A"  # sage green — repo name, success ticks, key numbers (sparingly)
HIGHLIGHT = "#C4A96B"  # warm gold — entry points, file paths, "start here"
