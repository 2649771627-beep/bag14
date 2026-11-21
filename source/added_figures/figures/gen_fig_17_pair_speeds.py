"""Plot the speed terms on the strictly opposite AIRY–ARS408 pairs."""

import csv
from pathlib import Path

import numpy as np

from figure_style import FAST, NEAR, VARIABLE, configure, finish, plt

HERE = Path(__file__).resolve().parent
ARMS = ("near_constant", "variable", "variable_fast")
COLORS = (NEAR, VARIABLE, FAST)
MARKERS = ("o", "^", "s")
LABELS = {
    "en": {
        "arms": ("Near-constant", "Primary variable", "Faster segment"),
        "short": ("Near-constant", "Primary\nvariable", "Faster\nsegment"),
        "sum": "Speed sum",
        "difference": "Speed difference",
        "spread": "Centred sum of squares (m²/s²)",
    },
    "zh": {
        "arms": ("近恒速", "主变速", "末段快推"),
        "short": ("近恒速", "主变速", "末段快推"),
        "sum": "速度大小之和",
        "difference": "速度大小之差",
        "spread": "中心化平方和 (m²/s²)",
    },
}


def load_speeds():
    data = {arm: [] for arm in ARMS}
    with (HERE.parent / "figure_data" / "field_pair_speeds.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            early, late = float(row["early_axis_mps"]), float(row["late_axis_mps"])
            assert int(row["strict_opposite"]) == (early * late < 0)
            if early * late < 0:
                outbound, inbound = max(early, late), -min(early, late)
                data[row["arm"]].append((outbound + inbound, outbound - inbound))
    assert tuple(map(len, data.values())) == (37, 108, 49)
    return [np.array(data[arm]) for arm in ARMS]


def plot(groups, lang):
    configure(lang)
    labels = LABELS[lang]
    fig, (scatter, bars) = plt.subplots(
        1, 2, figsize=(7.0, 3.25), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for ax, panel in ((scatter, "(a)"), (bars, "(b)")):
        ax.text(-0.14, 1.05, panel, transform=ax.transAxes, fontweight="bold")
        ax.grid(axis="y", color="#d9dde2", linewidth=0.6)
    for group, label, color, marker in zip(groups, labels["arms"], COLORS, MARKERS, strict=True):
        scatter.scatter(group[:, 0], group[:, 1], s=20, alpha=0.76, c=color, marker=marker,
                        edgecolors="white", linewidths=0.25, label=f"{label} ({len(group)})")
    scatter.axhline(0, color="#666666", linewidth=0.8, zorder=1)
    scatter.set_xlabel(f'{labels["sum"]} $S$ (m/s)')
    scatter.set_ylabel(f'{labels["difference"]} $D$ (m/s)')
    scatter.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, labelspacing=0.25)

    spread = np.array([((group - group.mean(axis=0)) ** 2).sum(axis=0) for group in groups])
    centre = np.arange(len(groups))
    width = 0.3
    for j, (symbol, face, hatch) in enumerate((("$S$", "#777c83", ""), ("$D$", "white", "///"))):
        bar = bars.bar(centre + (j - 0.5) * width, spread[:, j], width, label=symbol,
                       facecolor=face, edgecolor="#30343a", hatch=hatch, linewidth=0.8)
        bars.bar_label(bar, fmt="%.2f", padding=2, fontsize=8)
    bars.set_xticks(centre, labels["short"])
    bars.set_ylabel(labels["spread"])
    bars.set_ylim(0, spread.max() * 1.2)
    bars.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncols=2)
    finish(fig, f"pair_speed_terms_{lang}")
    return spread


if __name__ == "__main__":
    speeds = load_speeds()
    for language in LABELS:
        values = plot(speeds, language)
    print("Strict opposite sample sizes:", [len(group) for group in speeds])
    print("Centred speed-sum and speed-difference squares:", values.tolist())
