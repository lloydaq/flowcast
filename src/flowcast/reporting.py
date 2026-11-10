"""Shared helpers for model training reports: provenance, memory, and the figure style."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import psutil

from flowcast import paths

# Reference palette (dataviz skill, light mode) and the data hues fixed in architect.md §10.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FORECAST, COMPARISON = "#2a78d6", "#a86f0c"

MODEL_FIGURE_STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "grid.color": GRID,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
}


def git_commit() -> str:
    """Short commit hash, marked when the working tree had uncommitted changes to tracked files."""
    try:

        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=paths.PROJECT_ROOT, check=True
            ).stdout

        dirty = git("status", "--porcelain", "--untracked-files=no").strip()
        return git("rev-parse", "--short", "HEAD").strip() + (" + uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def peak_memory_mb() -> int:
    """Peak resident memory of this process so far (lifetime peak on Windows; current RSS elsewhere)."""
    mem = psutil.Process().memory_info()
    return getattr(mem, "peak_wset", mem.rss) // 2**20


def print_report(path: Path) -> None:
    """Echo a generated Markdown report. Reports hold non-ASCII (→, −) that a Windows console's
    default cp1252 encoding cannot print, so stdout is switched to UTF-8 first."""
    sys.stdout.reconfigure(encoding="utf-8")
    print(path.read_text(encoding="utf-8"))
