"""Draw association-rule and leave-one-traversal sensitivity for Figure S5."""

import json
import math
from pathlib import Path

from figure_style import NEAR, VARIABLE, configure, finish

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


DATA = Path(__file__).resolve().parent.parent / "figure_data" / "field_stability.json"
ARMS = ("near_constant", "variable")
MODES = ("geometry", "direction")
RULES = ("lidar_gated", "radar_only")
COLOURS = (NEAR, VARIABLE)
LABELS = {
    "en": {
        "arm": ("Near constant", "Variable speed"),
        "mode": ("Geometry only", "Direction centre"),
        "rule": ("LiDAR gated", "Radar only"),
        "offset": "Fitted time offset (ms)",
        "trip": "Omitted traversal",
    },
    "zh": {
        "arm": ("近恒速", "变速"),
        "mode": ("仅几何项", "含方向中心"),
        "rule": ("激光雷达门控", "仅毫米波雷达"),
        "offset": "拟合时偏（毫秒）",
        "trip": "省略的往返趟次",
    },
}


def offset(entry, mode):
    fit = entry["fits"][mode]
    value = fit["candidate_offset_ms"]
    if not fit["solver_converged"] or not math.isfinite(value):
        raise ValueError(f"Invalid candidate from the {mode} fit")
    return value


def read_results():
    source = json.loads(DATA.read_text(encoding="utf-8"))
    common = source["association_common_schedule"]["arms"]
    association = {}
    for arm in ARMS:
        result = common[arm]
        if len(set(result["receives_relative_ns"])) != 409:
            raise ValueError(f"Expected 409 shared frames for {arm}")
        if any(result[rule]["frames"] != 409 for rule in RULES):
            raise ValueError(f"Association comparisons must use the same 409 frames for {arm}")
        for mode in MODES:
            association[arm, mode] = tuple(offset(result[rule], mode) for rule in RULES)

    original = source["original"]["arms"]
    full = [offset(original[arm], "direction") for arm in ARMS]
    if any(not math.isclose(value, association[arm, "direction"][0], abs_tol=1e-9)
           for arm, value in zip(ARMS, full)):
        raise ValueError("Full fits differ from the LiDAR-gated reference fits")

    recorded_order = {arm: index for index, arm in enumerate(ARMS)}
    omissions = sorted(source["leave_one_traversal"],
                       key=lambda row: (recorded_order[row["removed_arm"]], row["removed_cycle"]))
    keys = [(row["removed_arm"], row["removed_cycle"]) for row in omissions]
    expected = [(arm, cycle) for arm, count in zip(ARMS, (7, 8)) for cycle in range(1, count + 1)]
    if keys != expected:
        raise ValueError("Expected one rematch for each of the seven and eight traversals")
    for row in omissions:
        if any(row[arm]["frames"] != row["matched_frames_per_arm"] for arm in ARMS):
            raise ValueError("Leave-one-traversal fits must use the same rematched frame count")
    rematched = [[offset(row[arm], "direction") for row in omissions] for arm in ARMS]
    frame_counts = [row["matched_frames_per_arm"] for row in omissions]
    print(f"Association n=409 per arm; {len(omissions)} omissions with "
          f"{min(frame_counts)}–{max(frame_counts)} matched frames per arm")
    for arm, values in zip(ARMS, rematched):
        print(f"{arm}: {min(values):+.2f} to {max(values):+.2f} ms after leave-one rematching")
    return association, full, rematched


def draw_association(ax, association, labels):
    row_labels = []
    for i, arm in enumerate(ARMS):
        colour = COLOURS[i]
        for j, mode in enumerate(MODES):
            y = 2 * i + j
            gated, radar = association[arm, mode]
            ax.plot((gated, radar), (y, y), color=colour, linewidth=1.2, zorder=2)
            ax.plot(gated, y, marker="o", ms=6, color=colour, linestyle="none", zorder=3)
            ax.plot(radar, y, marker="s", ms=5.4, markerfacecolor="white", markeredgecolor=colour,
                    markeredgewidth=1.25, linestyle="none", zorder=4)
            row_labels.append(f"{labels['arm'][i]}\n{labels['mode'][j]}")
    ax.set_yticks(range(4), row_labels)
    ax.set_ylim(3.54, -0.54)
    ax.set_xlim(-58, 92)
    ax.xaxis.set_major_locator(MultipleLocator(30))
    ax.axvline(0, color="0.75", linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlabel(labels["offset"])
    legend = [
        Line2D([], [], marker="o", color="0.23", markersize=5.5, linestyle="none", label=labels["rule"][0]),
        Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor="0.23", markersize=5.2,
               linestyle="none", label=labels["rule"][1]),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncols=2,
              handletextpad=0.25, columnspacing=1.05, borderaxespad=0, frameon=False)


def draw_rematches(ax, full, rematched, labels):
    visits = list(range(1, 8)) + list(range(9, 17))
    ax.axhline(0, color="0.75", linewidth=0.8, zorder=0)
    ax.axvline(8, color="0.8", linewidth=0.8, zorder=0)
    for i, (reference, values) in enumerate(zip(full, rematched)):
        colour = COLOURS[i]
        # Separate the two refits visually without changing their offset values.
        positions = [visit + (-0.18, 0.18)[i] for visit in visits]
        ax.axhline(reference, color=colour, linewidth=0.8, linestyle="--", zorder=1)
        ax.plot(positions, values, markersize=4.6, marker=("o", "s")[i], color=colour,
                linestyle="none", zorder=2, label=labels["arm"][i])
    ax.set_xlim(0.25, 16.75)
    ax.set_ylim(-144, 154)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.set_xticks(visits, list(map(str, range(1, 8))) + list(map(str, range(1, 9))))
    ax.set_ylabel(labels["offset"])
    ax.set_xlabel(labels["trip"], labelpad=28)
    for center, name in zip((4, 12.5), labels["arm"]):
        ax.text(center, -0.135, name, ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.03), ncols=2, handletextpad=0.25,
              columnspacing=1.05, borderaxespad=0, frameon=False)


def main():
    association, full, rematched = read_results()
    for language, labels in LABELS.items():
        configure(language)
        fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.65),
                                         gridspec_kw={"width_ratios": (1, 1.43)})
        fig.subplots_adjust(left=0.145, right=0.988, bottom=0.215, top=0.865, wspace=0.42)
        draw_association(left, association, labels)
        draw_rematches(right, full, rematched, labels)
        for ax, panel in zip((left, right), ("(a)", "(b)")):
            ax.text(0, 1.115, panel, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")
        finish(fig, f"association_sensitivity_{language}")


if __name__ == "__main__":
    main()
