"""Shared plotting helpers, saved to reports/figures/ for the pitch deck."""

import matplotlib.pyplot as plt

from visual_inspection_fm.config import FIGURES_DIR


def savefig(fig: plt.Figure, name: str) -> None:
    """Save a figure to reports/figures/<name>.png, creating the dir if needed."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    print(f"Saved {FIGURES_DIR / f'{name}.png'}")
