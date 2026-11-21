"""Plot the bilingual R01 vehicle results from the archived replay data."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "airy_ars408_vehicle_r01"
FIGURES = HERE / "figures"
ARMS = ("near_constant", "variable")
MODEL_KEYS = ("geometry", "direction")
METHOD_KEYS = ("original_paired", "robust_motion_axis_paired", "robust_motion_axis_framewise")
BLUE = "#176588"
ORANGE = "#AE4A24"
GRID = "#D8E1E6"
FONT_DIR = Path("C:/Windows/Fonts")

LABELS = {
    "en": {
        "motion": ("Near-constant speed", "Variable speed"),
        "near_trips": "Near-constant speed ({} round trips)",
        "variable_trips": "Variable speed (first {} round trips)",
        "fast_trips": "Final {} fast",
        "track_legend": ("AIRY board centre", "ARS408 corner reflector"),
        "track_x": "Elapsed recorder time within each row (s)",
        "track_y": "Vehicle longitudinal coordinate (m)",
        "profile_legend": ("Translation + yaw", "Translation + yaw + direction centre"),
        "profile_x": "Candidate offset on the replay time base (ms)",
        "profile_y": "Excess Huber loss per radar frame",
        "matched_rows": "{} frames per group, balanced by direction and longitudinal position",
        "methods": ("Original\npaired", "Robust\npaired", "Robust\nframewise"),
        "metrics": ("Direction-signed ±8 cm shift", "±30 ms timestamp injection"),
        "metric_y": ("Mean absolute offset change (ms)", "Mean absolute increment error (ms)"),
    },
    "zh": {
        "motion": ("近恒速", "变速"),
        "near_trips": "近恒速（{} 趟往返）",
        "variable_trips": "变速（前 {} 趟往返）",
        "fast_trips": "末 {} 趟快推",
        "track_legend": ("AIRY 标定板中心", "ARS408 角反射器"),
        "track_x": "各行相对于窗口起点的录制时间（s）",
        "track_y": "车体纵向坐标（m）",
        "profile_legend": ("平移与偏航", "平移、偏航与方向中心项"),
        "profile_x": "回放时间基准下的候选时偏（ms）",
        "profile_y": "每雷达帧的超额 Huber 损失",
        "matched_rows": "各组 {} 帧，按方向和纵向位置配平",
        "methods": ("原始\n配对", "稳健\n配对", "稳健\n逐帧"),
        "metrics": ("随方向换号的 ±8 cm 位移", "±30 ms 时间戳注入"),
        "metric_y": ("时偏变化的绝对值均值（ms）", "时偏增量恢复的绝对误差均值（ms）"),
    },
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def panel_style(panel):
    panel.set_axisbelow(True)
    panel.grid(axis="y", color=GRID, linewidth=0.5)
    panel.tick_params(length=3, width=0.65, pad=3)


def panel_label(panel, index, label):
    panel.text(0, 1.04, f"({chr(97 + index)}) {label}", transform=panel.transAxes, va="bottom")


def save(figure, stem, language):
    path = FIGURES / f"vehicle_{stem}_{language}"
    figure.savefig(path.with_suffix(".png"), dpi=400)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def track_data():
    directory = SOURCE / "tracks"
    clips = read_json(directory / "preparation.json")["clips"]
    tracks = {}
    for arm, clip in clips.items():
        start, end = clip["radar_clip_host_s"]
        for sensor in ("lidar", "radar"):
            rows = read_csv(directory / f"{arm}_{sensor}.csv")
            coords = np.array([[float(row["receive_s"]), float(row["x_m"])] for row in rows])
            window = (coords[:, 0] >= start) & (coords[:, 0] <= end)
            tracks[arm, sensor] = coords[window].T
    return clips, tracks


def plot_tracks(language, labels, clips, tracks):
    figure, panels = plt.subplots(2, 1, figsize=(6.5, 3.8), sharex=True, sharey=True)
    track_styles = {
        "lidar": dict(color=BLUE, linewidth=1.0, linestyle="-"),
        "radar": dict(color=ORANGE, linewidth=0.0, marker=".", markersize=1.7, alpha=0.62),
    }
    ends = []
    for index, (panel, arm) in enumerate(zip(panels, ARMS)):
        origin = clips[arm]["radar_clip_host_s"][0]
        arms_in_panel = (arm,) if index == 0 else (arm, "variable_fast")
        ends.append(clips[arms_in_panel[-1]]["radar_clip_host_s"][1] - origin)
        panel_style(panel)
        for displayed_arm in arms_in_panel:
            for sensor, style in track_styles.items():
                times, positions = tracks[displayed_arm, sensor]
                panel.plot(times - origin, positions, **style)
        count = clips[arm]["trip_count"]
        title = labels["near_trips" if index == 0 else "variable_trips"].format(count)
        panel_label(panel, index, title)
        if index == 1:
            begin, end = [time - origin for time in clips["variable_fast"]["radar_clip_host_s"]]
            panel.axvspan(begin, end, color="#E8E2F1", lw=0, zorder=0)
            panel.text((begin + end) / 2, 6.68,
                       labels["fast_trips"].format(clips["variable_fast"]["trip_count"]),
                       ha="center", va="center", color="#4A3E63", fontsize=8.1)
        panel.set_ylim(1.75, 7.0)
        panel.yaxis.set_major_locator(MultipleLocator(1))
        panel.xaxis.set_major_locator(MultipleLocator(30))
    panels[-1].set_xlim(0, 10 + max(ends))
    legend = [
        Line2D([], [], **track_styles["lidar"], label=labels["track_legend"][0]),
        Line2D([], [], ls="none", marker=".", ms=6, color=ORANGE, label=labels["track_legend"][1]),
    ]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.994),
                  ncol=2, frameon=False, handlelength=2, columnspacing=2)
    figure.supxlabel(labels["track_x"], x=0.55, y=0.018)
    figure.supylabel(labels["track_y"], x=0.022, y=0.50)
    figure.subplots_adjust(left=0.103, right=0.988, bottom=0.12, top=0.831, hspace=0.39)
    save(figure, "tracks", language)


def profile_data():
    table = read_csv(SOURCE / "profile_curves.csv")
    summary = read_json(SOURCE / "profile_compare.json")["arms"]
    counts = {summary[arm]["matched_rows"] for arm in ARMS}
    if len(counts) != 1:
        raise ValueError("The two profile arms must have equal fitted row counts.")
    return table, counts.pop()


def plot_profiles(language, labels, table, matched_rows):
    figure, panels = plt.subplots(1, 2, figsize=(6.5, 2.95), sharex=True, sharey=True)
    styles = (dict(color=BLUE, linestyle="-"), dict(color=ORANGE, linestyle="--"))
    for index, (panel, arm) in enumerate(zip(panels, ARMS)):
        panel_style(panel)
        panel_label(panel, index, labels["motion"][index])
        for model, style in zip(MODEL_KEYS, styles):
            curve = sorted((float(row["offset_ms"]), float(row["mean_excess_cost"]))
                           for row in table if row["model"] == model and row["arm"] == arm)
            times, values = np.array(curve).T
            panel.plot(times, values, linewidth=1.45, **style)
        panel.set_xlim(-600, 600)
        panel.set_ylim(0, 2.0)
        panel.xaxis.set_major_locator(MultipleLocator(300))
        panel.yaxis.set_major_locator(MultipleLocator(0.5))
    legend = [Line2D([], [], linewidth=1.45, label=label, **style)
              for style, label in zip(styles, labels["profile_legend"])]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.99), ncol=2,
                  frameon=False, handlelength=2.5, columnspacing=1.8)
    figure.supxlabel(labels["profile_x"], x=0.55, y=0.105)
    figure.text(0.55, 0.025, labels["matched_rows"].format(matched_rows),
                ha="center", va="bottom", fontsize=8.3, color="#3F4B53")
    figure.supylabel(labels["profile_y"], x=0.02, y=0.53)
    figure.subplots_adjust(left=0.114, right=0.983, bottom=0.225, top=0.79, wspace=0.145)
    save(figure, "profiles", language)


def replay_data():
    responses = read_json(SOURCE / "replay_summary.json")["responses"]
    primary = {(row["arm"], row["method"]): row for row in responses if row["pair_rule"] == "primary"}
    for arm in ARMS:
        for method in METHOD_KEYS:
            row = primary[arm, method]
            if row["direction_signs_available"] != 2 or row["time_signs_available"] != 2:
                raise ValueError(f"Both perturbation signs are required for {arm}: {method}")
    return primary


def plot_perturbations(language, labels, responses):
    figure, panels = plt.subplots(1, 2, figsize=(6.5, 3.02))
    categories = np.arange(len(METHOD_KEYS))
    width = 0.34
    colors = (BLUE, ORANGE)
    for index, (panel, field) in enumerate(zip(panels, ("direction_mean_abs_ms", "time_mean_abs_ms"))):
        panel_style(panel)
        panel_label(panel, index, labels["metrics"][index])
        for group_idx, (arm, color) in enumerate(zip(ARMS, colors)):
            values = [responses[arm, method][field] for method in METHOD_KEYS]
            bars = panel.bar(categories + width * (group_idx - 0.5), values, width,
                             facecolor=color if group_idx == 0 else "white", edgecolor=color,
                             linewidth=0.9, hatch=None if group_idx == 0 else "///",
                             label=labels["motion"][group_idx])
            panel.bar_label(bars, fmt="%.1f", padding=1.8, fontsize=7.6, color="#212D33")
        panel.set(xticks=categories, xticklabels=labels["methods"], ylabel=labels["metric_y"][index])
        panel.tick_params(axis="x", length=0, pad=5)
        panel.set_ylim(0, (245, 43)[index])
        panel.yaxis.set_major_locator(MultipleLocator((50, 10)[index]))
    handles, names = panels[0].get_legend_handles_labels()
    figure.legend(handles, names, loc="upper center", bbox_to_anchor=(0.54, 0.993),
                  ncol=2, frameon=False, handlelength=1.7, columnspacing=1.9)
    figure.subplots_adjust(left=0.105, right=0.988, bottom=0.20, top=0.80, wspace=0.36)
    save(figure, "perturbations", language)


def main():
    FIGURES.mkdir(exist_ok=True)
    clips, tracks = track_data()
    curves, count = profile_data()
    responses = replay_data()
    chinese = FONT_DIR / "msyh.ttc"
    font_manager.fontManager.addfont(chinese)
    fonts = {"en": "Times New Roman", "zh": font_manager.FontProperties(fname=chinese).get_name()}
    for language, labels in LABELS.items():
        with matplotlib.rc_context({
            "font.family": fonts[language], "font.size": 9.4, "pdf.fonttype": 42,
            "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
            "axes.unicode_minus": language != "zh",
            "figure.labelsize": 9.6,
        }):
            plot_tracks(language, labels, clips, tracks)
            plot_profiles(language, labels, curves, count)
            plot_perturbations(language, labels, responses)


if __name__ == "__main__":
    main()
