"""Plot fixed-pair AIRY timing comparisons from the replay CSVs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ARMS = {
    "near_constant": "Near-constant",
    "variable": "Variable",
    "variable_fast": "Faster segment",
}
METHODS = {
    "original_paired": "Original paired EKF",
    "motion_axis_paired": "Direction-center paired EKF",
    "robust_motion_axis_paired": "Huber direction-center paired EKF",
    "robust_motion_axis_framewise": "Huber direction-center framewise EKF",
}
COST_METHODS = {
    "original_paired": "Paired differences",
    "motion_axis_paired": "With direction center",
    "robust_motion_axis_paired": "With direction center\n+ Huber loss",
}
TIMING = {
    "frame_header": ("AIRY frame header", "#176083", "-", "o"),
    "board_point": ("AIRY board-point time", "#AE451B", "--", "s"),
}


def read_grouped(path, x_column, y_column):
    grouped = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            group = (row["arm"], row["method"], row["timing"])
            value = row[y_column].strip()
            grouped.setdefault(group, []).append(
                (float(row[x_column]), float(value) if value else math.nan)
            )
    return {group: sorted(points) for group, points in grouped.items()}


def panels(grouped):
    arms = [arm for arm in ARMS if any(key[0] == arm for key in grouped)]
    methods = [method for method in METHODS if any(key[1] == method for key in grouped)]
    if not arms or not methods:
        raise ValueError("No AIRY timing comparison rows match the plot labels.")
    fig, axes = plt.subplots(
        len(methods), len(arms), squeeze=False, sharex="col",
        figsize=(max(4.5, 3.65 * len(arms)), 1.25 + 2.05 * len(methods)),
    )
    fig.subplots_adjust(left=0.13, right=0.97, top=0.76 if len(methods) == 1 else 0.89,
                        bottom=0.15 if len(methods) == 1 else 0.12,
                        hspace=0.28, wspace=0.31)
    for axis, arm in zip(axes[0], arms):
        axis.set_title(ARMS[arm], fontweight="bold", pad=8)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D7E0E5", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, axes, arms, methods


def method_label(method):
    return METHODS[method].replace(" paired EKF", "\npaired EKF").replace(
        " framewise EKF", "\nframewise EKF"
    )


def draw_legend(fig, failures=False, no_omissions=False):
    handles = [
        Line2D([0], [0], label=label, color=color, linestyle=line, linewidth=1.7,
               marker=marker, markersize=5)
        for label, color, line, marker in TIMING.values()
    ]
    if failures:
        handles.append(Line2D([0], [0], label="No estimate", color="#454D54",
                              marker="x", linestyle="none", markersize=6))
    if no_omissions:
        handles.append(Line2D([0], [0], label="No pair removed", color="#454D54",
                              marker="o", markerfacecolor="white", linestyle="none", markersize=5))
    fig.legend(handles=handles, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 0.985), frameon=False, columnspacing=2.05)


def draw_costs(grouped, output):
    fig, axes, arms, methods = panels(grouped)
    for i, method in enumerate(methods):
        axes[i, 0].set_ylabel(f"{COST_METHODS[method]}\nMean loss per pair", labelpad=9)
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            for timing, (_, color, line, marker) in TIMING.items():
                points = grouped.get((arm, method, timing))
                if not points:
                    continue
                x, y = zip(*points)
                axis.plot(x, y, color=color, linestyle=line, linewidth=1.7)
                finite = [(a, b) for a, b in points if math.isfinite(b)]
                if finite:
                    xmin, ymin = min(finite, key=lambda point: point[1])
                    axis.scatter(xmin, ymin, color=color, marker=marker, s=25,
                                 edgecolor="white", linewidth=0.45, zorder=5)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    draw_legend(fig)
    fig.supxlabel("Candidate time offset (ms)", y=0.038)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def read_unchanged_trials(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            (row["arm"], row["method"], row["timing"], float(row["omitted_trip"]))
            for row in csv.DictReader(stream)
            if row.get("n_pairs_removed") == "0"
        }


def draw_stability(grouped, unchanged, output):
    fig, axes, arms, methods = panels(grouped)
    failures = False
    for i, method in enumerate(methods):
        axes[i, 0].set_ylabel(f"{method_label(method)}\nTerminal offset (ms)", labelpad=9)
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            rounds = set()
            for timing, (_, color, line, marker) in TIMING.items():
                points = grouped.get((arm, method, timing))
                if not points:
                    continue
                trips = [trip for trip, _ in points]
                rounds.update(trips)
                informative_y = [offset if (arm, method, timing, trip) not in unchanged
                                 else math.nan for trip, offset in points]
                axis.plot(trips, informative_y, color=color, linestyle=line, linewidth=1.4,
                          marker=marker, markersize=4.5, markeredgecolor="white",
                          markeredgewidth=0.45)
                intact = [(trip, offset) for trip, offset in points
                          if (arm, method, timing, trip) in unchanged and math.isfinite(offset)]
                if intact:
                    unchanged_x, unchanged_y = zip(*intact)
                    axis.scatter(unchanged_x, unchanged_y, edgecolors=color, marker=marker,
                                 facecolors="white", linewidth=0.95, s=18, zorder=4)
                unavailable = [trip for trip, offset in points if not math.isfinite(offset)]
                if unavailable:
                    failures = True
                    axis.scatter(unavailable, [0.04] * len(unavailable), color=color,
                                 marker="x", transform=axis.get_xaxis_transform(), s=18)
            axis.set_xticks(sorted(rounds))
            axis.margins(x=0.07)
    draw_legend(fig, failures=failures, no_omissions=bool(unchanged))
    fig.supxlabel("Omitted round trip", y=0.038)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)


def render(out):
    out = Path(out)
    costs = read_grouped(out / "cost_curves.csv", "offset_ms", "mean_cost")
    stability_path = out / "stability.csv"
    stability = read_grouped(stability_path, "omitted_trip", "estimate_ms")
    unchanged = read_unchanged_trials(stability_path)
    with matplotlib.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.5,
    }):
        draw_costs(costs, out / "cost_curves.png")
        draw_stability(stability, unchanged, out / "stability.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="Directory containing the two replay CSVs")
    render(parser.parse_args().out)
