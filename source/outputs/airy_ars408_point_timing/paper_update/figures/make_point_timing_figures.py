"""Render two bilingual figures from the fixed-pair AIRY timing replay."""

from pathlib import Path
import math
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
REPLAY = HERE.parents[1]
sys.path.insert(0, str(REPLAY))

from plot_comparison import ARMS, COST_METHODS, METHODS, TIMING, read_grouped, read_unchanged_trials


LABELS = {
    "en": {
        "arms": ("Near-constant", "Variable speed", "Faster segment"),
        "cost_methods": ("Paired differences", "Direction center", "Direction center\n+ Huber loss"),
        "ekf_methods": ("Original paired EKF", "Direction-center\npaired EKF",
                        "Huber + center\npaired EKF", "Huber + center\nframewise EKF"),
        "timing": ("AIRY frame header", "AIRY board-point time"),
        "cost_x": "Candidate time offset (ms)",
        "cost_y": "Mean loss per pair",
        "stability_x": "Omitted round trip",
        "stability_y": "Terminal offset (ms)",
        "boundary": "Minimum at search boundary",
        "no_pairs": "No pair removed",
    },
    "zh": {
        "arms": ("近恒速", "主变速", "末段快推"),
        "cost_methods": ("原配对差分", "加入方向中心", "方向中心 + Huber"),
        "ekf_methods": ("原配对 EKF", "方向中心\n配对 EKF", "方向中心 + Huber\n配对 EKF",
                        "方向中心 + Huber\n同模型逐帧 EKF"),
        "timing": ("AIRY 帧头时标", "AIRY 板点时标"),
        "cost_x": "候选时偏（ms）",
        "cost_y": "每对平均代价",
        "stability_x": "留出的往返趟次",
        "stability_y": "终点时偏（ms）",
        "boundary": "最低点位于搜索边界",
        "no_pairs": "未移除配对",
    },
}


def setup_panels(methods, arm_titles, y_title):
    fig, axes = plt.subplots(len(methods), len(arm_titles), sharex="col",
                             figsize=(7.2, 1.28 + 1.36 * len(methods)), squeeze=False)
    compact = len(methods) == 2
    fig.subplots_adjust(left=0.20, right=0.975, top=0.83 if compact else 0.88,
                        bottom=0.16 if compact else 0.11,
                        wspace=0.38, hspace=0.31)
    for axis, title in zip(axes[0], arm_titles):
        axis.set_title(title, fontweight="bold", pad=6)
    for axis, method in zip(axes[:, 0], methods):
        axis.set_ylabel(method, labelpad=5)
    fig.supylabel(y_title, x=0.028)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D7E0E5", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    return fig, axes


def panel_legend(fig, labels, helper_label, helper_marker):
    handles = [
        Line2D([0], [0], label=label, color=color, linestyle=line, linewidth=1.6,
               marker=marker, markersize=5)
        for label, (_, color, line, marker) in zip(labels, TIMING.values())
    ]
    handles.append(Line2D([0], [0], label=helper_label, color="#25313B",
                          marker=helper_marker, markerfacecolor="none", linestyle="none", markersize=6))
    fig.legend(handles=handles, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.55, 0.986), frameon=False, columnspacing=1.1, handletextpad=0.35)


def render_costs(costs, labels, path, arms):
    titles = dict(zip(ARMS, labels["arms"]))
    fig, axes = setup_panels(labels["cost_methods"], [titles[a] for a in arms], labels["cost_y"])
    for i, method in enumerate(COST_METHODS):
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            for timing, (_, color, line, marker) in TIMING.items():
                points = costs[arm, method, timing]
                assert len(points) == 1201 and all(math.isfinite(y) for _, y in points)
                x, y = zip(*points)
                axis.plot(x, y, color=color, linestyle=line, linewidth=1.6)
                xmin, ymin = min(points, key=lambda point: point[1])
                axis.scatter(xmin, ymin, color=color, marker=marker, s=25,
                             edgecolor="white", linewidth=0.45, zorder=5)
                if xmin in (x[0], x[-1]):
                    axis.scatter(xmin, ymin, facecolor="none", edgecolor="#25313B",
                                 marker="D", s=58, linewidth=0.9, zorder=6)
            axis.set_xticks([-600, -300, 0, 300, 600])
            axis.margins(x=0.07, y=0.085)
    panel_legend(fig, labels["timing"], labels["boundary"], "D")
    fig.supxlabel(labels["cost_x"], y=0.03)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def render_stability(stability, unchanged, labels, path, arms, methods):
    titles = dict(zip(ARMS, labels["arms"]))
    method_titles = dict(zip(METHODS, labels["ekf_methods"]))
    fig, axes = setup_panels([method_titles[m] for m in methods], [titles[a] for a in arms],
                             labels["stability_y"])
    for i, method in enumerate(methods):
        for j, arm in enumerate(arms):
            axis = axes[i, j]
            for timing, (_, color, line, marker) in TIMING.items():
                points = stability[arm, method, timing]
                assert all(math.isfinite(y) for _, y in points)
                x = [trip for trip, _ in points]
                y = [offset if (arm, method, timing, trip) not in unchanged else math.nan
                     for trip, offset in points]
                axis.plot(x, y, color=color, linestyle=line, linewidth=1.35,
                          marker=marker, markersize=4.3, markeredgecolor="white", markeredgewidth=0.45)
                intact = [(trip, offset) for trip, offset in points
                          if (arm, method, timing, trip) in unchanged]
                if intact:
                    axis.scatter(*zip(*intact), edgecolors=color, marker=marker,
                                 facecolors="white", linewidth=0.9, s=19, zorder=4)
            axis.set_xticks(x)
            axis.margins(x=0.07)
    panel_legend(fig, labels["timing"], labels["no_pairs"], "o")
    fig.supxlabel(labels["stability_x"], y=0.03)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def render(replay, output, cjk_font=None):
    output.mkdir(parents=True, exist_ok=True)
    cost_path = replay / "cost_curves.csv"
    stability_path = replay / "stability.csv"
    costs = read_grouped(cost_path, "offset_ms", "mean_cost")
    stability = read_grouped(stability_path, "omitted_trip", "estimate_ms")
    unchanged = read_unchanged_trials(stability_path)
    languages = ["en"]
    if cjk_font is not None:
        font_manager.fontManager.addfont(cjk_font)
        chinese_font = font_manager.FontProperties(fname=cjk_font).get_name()
        languages.append("zh")
    for language in languages:
        labels = LABELS[language]
        style = {
            "font.family": chinese_font if language == "zh" else "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.4,
        }
        with matplotlib.rc_context(style):
            main_arms = ("near_constant", "variable")
            main_methods = ("original_paired", "robust_motion_axis_paired")
            render_costs(costs, labels, output / f"point_timing_cost_main_{language}.png", main_arms)
            render_stability(stability, unchanged, labels,
                             output / f"point_timing_stability_main_{language}.png", main_arms, main_methods)
            render_costs(costs, labels, output / f"point_timing_cost_{language}.png", tuple(ARMS))
            render_stability(stability, unchanged, labels,
                             output / f"point_timing_stability_{language}.png", tuple(ARMS), tuple(METHODS))

    return sorted(output / f"point_timing_{kind}{layout}_{language}.png"
                  for language in languages for kind in ("cost", "stability")
                  for layout in ("", "_main"))
