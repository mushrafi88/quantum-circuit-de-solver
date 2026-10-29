"""Shared figure style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "draft_paper" / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Validated categorical order (blue, orange, aqua); exact curves in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
MUTED = "#52514e"
ARCH_COLOR = {"ms": SERIES[0], "hea": SERIES[1], "reupload": SERIES[2],
              "trotter": "#6250d6"}
ARCH_LABEL = {"ms": "MS", "hea": "HEA", "reupload": "Re-upload", "trotter": "Trotter"}
ARCH_STYLE = {"ms": "-", "hea": "--", "reupload": "-.", "trotter": ":"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 1.5, "lines.markersize": 4,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> draft_paper/figures/{name}.pdf")
