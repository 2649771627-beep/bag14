"""Common print settings for the added Measurement figures."""

from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / ".mplconfig"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

NEAR = "#2166a3"
VARIABLE = "#a44514"
FAST = "#137a65"


def configure(language):
    """Use Times for English and a CJK font for the Chinese labels."""
    mpl.rcParams.update({
        "font.family": "serif" if language == "en" else "sans-serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"],
        "font.size": 10,
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def finish(fig, stem):
    """Save a vector original and a 300 dpi copy for Word."""
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06,
                metadata={"CreationDate": None})
    fig.savefig(HERE / f"{stem}.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
