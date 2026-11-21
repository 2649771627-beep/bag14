"""Draw bilingual figures from the archived simulation and local-information ablation."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SIMULATION = (HERE.parent / "mst_q2_20261118" / "replay_extension"
              / "augmented_baseline_results.json")
ABLATION = PROJECT / "review-stage" / "experiments" / "local_information_ablation" / "results.json"
METHODS = ("framewise_ekf", "paired_ekf", "framewise_shared_bias_std_0.30m")
SUPPORTS = ("08_near_constant", "08_variable", "09_full", "ghost18_right")
BLUE, ORANGE, GREY = "#21638a", "#be5828", "#687782"
GRID = "#dce3e8"
WIDTH_INCHES = 16.0 / 2.54


LABELS = {
    "en": {
        "methods": ("Timing-only framewise EKF", "Paired EKF", "Framewise EKF + translation"),
        "shifts": ("No displacement", "Common ±8 cm", "Direction-changing ±8 cm"),
        "rmse": "Terminal time-offset RMSE (ms)",
        "supports": ("08, near-constant speed", "08, variable speed", "09, full support", "G18-C, right radar"),
        "models": ("Geometry only", "Geometry + direction centre"),
        "information": r"Projected timing information ($\mathrm{s}^{-2}$)",
        "retained": r"$I_D / I_G$",
    },
    "zh": {
        "methods": ("仅含时钟状态的逐帧 EKF", "正反向配对 EKF", "联合估计平移的逐帧 EKF"),
        "shifts": ("零位移", "共同位移 ±8 cm", "随方向换号的位移 ±8 cm"),
        "rmse": "终点时偏均方根误差（ms）",
        "supports": ("08，近恒速", "08，变速", "09，整段观测", "G18-C，右雷达"),
        "models": ("仅几何项", "几何项与方向中心项"),
        "information": r"投影后的时偏信息量（$\mathrm{s}^{-2}$）",
        "retained": r"$I_D / I_G$",
    },
}


def read_simulation():
    document = json.loads(SIMULATION.read_text(encoding="utf-8"))
    design = document["designs"]["unequal_1p2_1p8"]
    assert (document["trial_count"], document["offset_truth_ms"]) == (500, 30.0)
    assert document["primary_shared_xy_prior_std_m"] == 0.3
    rows = []
    for method in METHODS:
        zero = design["conditions"]["zero"]["statistics"][method]["rmse_ms"]
        row = [zero]
        for displacement in ("common", "direction"):
            signed_rmses = []
            for polarity in ("plus", "minus"):
                condition = design["conditions"][f"{displacement}_{polarity}"]
                assert abs(condition[f"{displacement}_shift_m"]) == 0.08
                signed_rmses.append(condition["statistics"][method]["rmse_ms"])
            pooled = design["pooled"][f"{displacement}_polarities"]["rmse_ms"][method]
            assert math.isclose(pooled, math.sqrt(sum(value**2 for value in signed_rmses) / 2))
            row.append(pooled)
        rows.append(row)
    return rows


def read_ablation():
    groups = json.loads(ABLATION.read_text(encoding="utf-8"))["groups"]
    result = []
    for support in SUPPORTS:
        group = groups[support]
        geometry = group["geometry"]["information_s_minus2"]
        direction = group["direction"]["information_s_minus2"]
        assert group["anchor_domain_audit"]["all_local_model_bounds_valid"]
        assert group["at_embedded_direction_c_m"] == 0.0
        assert 0 < direction < geometry
        assert math.isclose(direction / geometry, group["information_ratio_direction_over_geometry"])
        result.append((geometry, direction))
    return result


def save(figure, name):
    path = HERE / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=350)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def draw_simulation(language, labels, values):
    figure, axis = plt.subplots(figsize=(WIDTH_INCHES, 2.9))
    width = 0.22
    for i, (row, method, colour, hatch) in enumerate(zip(values, labels["methods"],
                                                        (GREY, BLUE, ORANGE), ("//", "", ".."))):
        positions = [condition + width * (i - 1) for condition in range(len(row))]
        bars = axis.bar(positions, row, width * 0.94, label=method, color=colour,
                        edgecolor="#fff", linewidth=0.25, hatch=hatch, zorder=3)
        axis.bar_label(bars, fmt="%.2f", padding=2.2, fontsize=8.4)
    axis.set_xticks(range(len(labels["shifts"])), labels["shifts"])
    axis.set_ylabel(labels["rmse"])
    axis.set_ylim(0, 63)
    axis.yaxis.set_major_locator(MultipleLocator(15))
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="both", length=0, pad=7)
    axis.grid(axis="y", linewidth=0.55, color=GRID, zorder=0)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.045), ncol=3, frameon=False,
                columnspacing=1.5, handlelength=1.7, borderaxespad=0)
    figure.subplots_adjust(left=0.104, right=0.987, bottom=0.15, top=0.865)
    save(figure, f"prior_common_shift_{language}")


def draw_ablation(language, labels, values):
    figure, axis = plt.subplots(figsize=(WIDTH_INCHES, 2.4))
    offsets = range(len(values))
    for row, (geometry, direction) in enumerate(values):
        axis.hlines(row, xmin=direction, xmax=geometry, color="#8b9da9", linewidth=1.3, zorder=2)
        axis.scatter(geometry, row, color=BLUE, marker="o", s=32, zorder=3)
        axis.scatter(direction, row, facecolor="white", edgecolor=ORANGE, marker="s", s=28,
                     linewidth=1.55, zorder=3)
        axis.text(1.15, row, f"{direction / geometry:.3%}", ha="center", va="center",
                  transform=axis.get_yaxis_transform())

    axis.text(1.15, 1.036, labels["retained"], ha="center", va="bottom", transform=axis.transAxes)
    axis.set_yticks(offsets, labels["supports"])
    axis.set_ylim(len(values) - 0.55, -0.55)
    axis.set_xscale("log")
    axis.set_xlim(75, 65000)
    axis.set_xlabel(labels["information"], labelpad=7)
    axis.set_xticks((100, 1000, 10000), (r"$10^2$", r"$10^3$", r"$10^4$"))
    axis.minorticks_off()
    axis.tick_params(axis="y", length=0, pad=8)
    axis.tick_params(axis="x", length=3)
    axis.spines["left"].set_visible(False)
    axis.grid(axis="x", linewidth=0.55, color=GRID, zorder=0)
    markers = [Line2D([], [], color=BLUE, marker="o", markersize=5.5, linestyle="none"),
               Line2D([], [], markerfacecolor="white", markeredgecolor=ORANGE, marker="s",
                      markersize=5.2, markeredgewidth=1.35, linestyle="none")]
    axis.legend(markers, labels["models"], loc="lower center", bbox_to_anchor=(0.48, 1.06),
                ncol=2, frameon=False, columnspacing=1.4, handletextpad=0.35, borderaxespad=0)
    figure.subplots_adjust(left=0.268 if language == "en" else 0.216,
                           right=0.845 if language == "en" else 0.85, bottom=0.23, top=0.82)
    save(figure, f"prior_direction_ablation_{language}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese-font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    arguments = parser.parse_args()
    font_manager.fontManager.addfont(arguments.chinese_font)
    chinese_family = font_manager.FontProperties(fname=arguments.chinese_font).get_name()
    simulation = read_simulation()
    ablation = read_ablation()
    for language, labels in LABELS.items():
        with matplotlib.rc_context({
            "font.family": "Times New Roman" if language == "en" else chinese_family,
            "font.size": 9,
            "axes.labelsize": 9.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "hatch.linewidth": 0.5,
            "pdf.fonttype": 42,
        }):
            draw_simulation(language, labels, simulation)
            draw_ablation(language, labels, ablation)

    for method, row in zip(METHODS, simulation):
        print(f"{method}: {', '.join(f'{value:.2f}' for value in row)} ms")
    for support, (geometry, direction) in zip(SUPPORTS, ablation):
        print(f"{support}: {geometry:.3f} to {direction:.3f} s^-2 ({direction / geometry:.3%})")


if __name__ == "__main__":
    main()
