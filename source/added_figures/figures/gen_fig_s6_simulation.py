"""Plot archived known-offset trials by motion type and planted offset."""

import csv
from pathlib import Path

from figure_style import NEAR, VARIABLE, configure, finish

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
import numpy as np


DATA = Path(__file__).resolve().parent.parent / "figure_data"
MOTIONS = ("near_constant", "variable")
OFFSETS = (70, 160)
COHORTS = tuple((motion, offset) for offset in OFFSETS for motion in MOTIONS)
MOTION_LABELS = {
    "en": ("Near-constant speed", "Variable speed"),
    "zh": ("近恒速", "变速"),
}
AXIS_LABELS = {
    "en": ("Longitudinal local timing scale (ms)", "Terminal absolute error (ms)"),
    "zh": ("纵向局部时偏尺度 (ms)", "终点绝对误差 (ms)"),
}


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    end = counts.cumsum()
    mean_rank = end - (counts - 1) / 2
    return mean_rank[inverse]


def checked_data():
    trials = read_csv("simulation_scale_trials.csv")
    summaries = {
        (row["motion"], int(row["truth_ms"]) if row["truth_ms"] else None): row
        for row in read_csv("simulation_scale_summary.csv")
    }
    keys = [(row["motion"], int(row["truth_ms"]), int(row["seed"])) for row in trials]
    if len(trials) != len(set(keys)):
        raise ValueError("The simulation trial ledger contains duplicate keys.")
    if {key[:2] for key in keys} != set(COHORTS):
        raise ValueError("The simulation trial ledger has unexpected cohorts.")
    if any(
        row["method"] != "robust_axis_paired"
        or row["observation"] != "tracked"
        or float(row["radar_step_m"]) != 0
        or row["valid_for_correlation"] not in {"0", "1"}
        for row in trials
    ):
        raise ValueError("The simulation trial ledger has unexpected settings.")

    excluded = [
        (row["motion"], int(row["truth_ms"]), int(row["seed"]), row["trial_status"])
        for row in trials if row["valid_for_correlation"] != "1"
    ]
    if excluded != [("variable", 70, 110, "failed_reference_update")]:
        raise ValueError(f"Unexpected excluded trials {excluded}.")

    groups = {}
    for motion, offset in COHORTS:
        rows = [row for row in trials if row["motion"] == motion and int(row["truth_ms"]) == offset]
        groups[motion, offset] = checked_points(rows, summaries[motion, offset])
    pooled = checked_points(trials, summaries["pooled", None])
    print(f"Pooled simulation trials {len(pooled[0])}/{len(trials)}, Spearman rho {pooled[2]:+.3f}")
    return groups


def checked_points(rows, summary):
    valid = [row for row in rows if row["valid_for_correlation"] == "1"]
    counts = len(rows), len(valid), len(rows) - len(valid)
    columns = ("scheduled", "valid", "excluded_from_correlation")
    if counts != tuple(int(summary[column]) for column in columns):
        raise ValueError(f"Trial counts differ from the archive summary {counts}.")
    x = np.array([float(row["local_x_data_scale_ms"]) for row in valid])
    y = np.array([float(row["abs_error_ms"]) for row in valid])
    if not (np.all((x > 80) & (x < 175)) and np.all((y > 0.5) & (y < 500))):
        raise ValueError("A plotted value lies outside the requested shared axes.")
    rho = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
    if not np.isclose(rho, float(summary["spearman_rho"]), atol=1e-12, rtol=0):
        raise ValueError(f"Spearman rho differs from the archive summary {rho}.")
    return x, y, rho


def plot(groups, language):
    configure(language)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.1, 4.5), layout="none")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.84, wspace=0.13, hspace=0.39)
    for col, label in enumerate(MOTION_LABELS[language]):
        position = axes[0, col].get_position()
        fig.text((position.x0 + position.x1) / 2, 0.96, label, ha="center", va="top", fontweight="bold")
    for i, ((motion, offset), ax) in enumerate(zip(COHORTS, axes.flat)):
        x, y, rho = groups[motion, offset]
        color, marker = (NEAR, "o") if motion == "near_constant" else (VARIABLE, "^")
        ax.scatter(x, y, s=24, marker=marker, facecolor=color, edgecolor="white", linewidth=0.35, alpha=0.92)
        ax.set(xlim=(80, 175), ylim=(0.5, 500), yscale="log", xticks=(80, 100, 120, 140, 160))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", color="#cccccc", linewidth=0.55)
        ax.set_axisbelow(True)
        rho_label = f"{rho:+.3f}".replace("-", "−")
        offset_label = "Planted" if language == "en" else "植入时偏"
        detail = f"({chr(97 + i)})  {offset_label} +{offset} ms    n = {len(x)}    ρ = {rho_label}"
        ax.text(0, 1.075, detail, transform=ax.transAxes, va="bottom", fontsize=9)
    xlabel, ylabel = AXIS_LABELS[language]
    fig.supxlabel(xlabel, x=0.55, y=0.035, fontsize=10)
    fig.supylabel(ylabel, x=0.025, y=0.49, fontsize=10)
    finish(fig, f"simulation_scale_{language}")


if __name__ == "__main__":
    groups = checked_data()
    for language in ("en", "zh"):
        plot(groups, language)
