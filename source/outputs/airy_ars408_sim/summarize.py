"""Check matched AIRY–ARS408 trials, tabulate their errors and plot empirical CDFs."""

import argparse
import csv
from itertools import product
from math import isfinite
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
METHOD_STYLE = {
    "raw_paired": ("#505050", ":", "原配对 EKF"),
    "axis_paired": ("#0072B2", "-.", "主轴配对 EKF"),
    "robust_axis_paired": ("#D55E00", "--", "稳健主轴配对 EKF"),
    "robust_axis_framewise": ("#00795F", "-", "同模型逐帧 EKF"),
}
MOTIONS = ("near_constant", "variable")
MOTION_LABEL = {"near_constant": "近恒速", "variable": "变速"}
OBSERVATIONS = (("complete", 0.0), ("oracle_missing", 0.0), ("tracked", 0.0), ("tracked", 0.20))
TRUTHS = (70, 160)
SEEDS = range(90, 120)
MATCH_FIELDS = (
    "pair_schedule_sha256", "prior_receive_s", "terminal_receive_s", "n_pairs",
    "n_members", "n_selected", "wrong_members",
)
METHOD_COMPARISONS = (
    ("raw_paired", "axis_paired"),
    ("raw_paired", "robust_axis_paired"),
    ("axis_paired", "robust_axis_paired"),
    ("robust_axis_paired", "robust_axis_framewise"),
)


def optional_float(row, field):
    value = row[field]
    if not value:
        return None
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} is not finite in {row}")
    return number


def load_trials(path):
    trials = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            condition = (row["motion"], (row["observation"], float(row["radar_step_m"])),
                         int(row["truth_ms"]))
            seed, method = int(row["seed"]), row["method"]
            key = (*condition, seed, method)
            if key in trials:
                raise ValueError(f"Duplicate trial: {key}")
            row["error"] = optional_float(row, "abs_error_ms")
            row["signed_error"] = optional_float(row, "signed_error_ms")
            estimate = optional_float(row, "estimate_ms")
            n_pairs = int(row["n_pairs"])
            n_members = int(row["n_members"])
            if n_members != 2 * n_pairs:
                raise ValueError(f"Unexpected number of members in {key}")
            if row["error"] is not None:
                if (n_pairs == 0 or int(row["failed_updates"]) != 0 or estimate is None or
                        row["signed_error"] is None or
                        abs(estimate - condition[-1] - row["signed_error"]) > 1e-9 or
                        abs(abs(row["signed_error"]) - row["error"]) > 1e-9):
                    raise ValueError(f"Inconsistent successful trial: {key}")
            elif ((n_pairs > 0 and int(row["failed_updates"]) == 0)
                  or estimate is not None or row["signed_error"] is not None):
                raise ValueError(f"Unexplained missing estimate: {key}")
            trials[key] = row
    expected = set(product(MOTIONS, OBSERVATIONS, TRUTHS, SEEDS, METHOD_STYLE))
    if trials.keys() != expected:
        raise ValueError(f"Unexpected trial coverage, missing {len(expected - trials.keys())} "
                         f"and extra {len(trials.keys() - expected)}")
    for key in product(MOTIONS, OBSERVATIONS, TRUTHS, SEEDS):
        signatures = {tuple(trials[*key, method][field] for field in MATCH_FIELDS)
                      for method in METHOD_STYLE}
        if len(signatures) != 1:
            raise ValueError(f"Mismatched pair schedules or endpoints: {key}")
    return trials


def table_rows(trials, np):
    summary, comparisons = [], []
    for motion, (arm, step), truth in product(MOTIONS, OBSERVATIONS, TRUTHS):
        condition = (motion, (arm, step), truth)
        identifiers = dict(motion=motion, observation=arm, radar_step_m=step, truth_ms=truth)
        for method in METHOD_STYLE:
            rows = [trials[*condition, seed, method] for seed in SEEDS]
            errors = np.array([row["error"] for row in rows if row["error"] is not None])
            signed = [row["signed_error"] for row in rows if row["signed_error"] is not None]
            counts = [int(row["n_pairs"]) for row in rows]
            scales = [optional_float(row, "local_x_data_scale_ms") for row in rows]
            scales = [scale for scale in scales if scale is not None]
            sum_spreads = [optional_float(row, "pair_speed_sum_std_mps") for row in rows]
            sum_spreads = [spread for spread in sum_spreads if spread is not None]
            wrong_count = sum(int(row["wrong_members"]) for row in rows)
            total_count = sum(int(row["n_members"]) for row in rows)
            summary.append(dict(
                **identifiers, method=method, n_seeds=len(rows), n_finite=len(errors),
                n_no_pairs=counts.count(0), n_failed_updates=sum(int(row["failed_updates"]) > 0 for row in rows),
                median_abs_error_ms=float(np.median(errors)) if len(errors) else None,
                p90_abs_error_ms=float(np.quantile(errors, 0.90)) if len(errors) else None,
                mean_abs_error_ms=float(np.mean(errors)) if len(errors) else None,
                sd_abs_error_ms=float(np.std(errors, ddof=1)) if len(errors) > 1 else None,
                rmse_ms=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                mean_signed_error_ms=float(np.mean(signed)) if len(signed) else None,
                median_n_pairs=float(np.median(counts)), min_n_pairs=min(counts), max_n_pairs=max(counts),
                median_pair_speed_sum_std_mps=float(np.median(sum_spreads)) if sum_spreads else None,
                n_finite_local_x_scales=len(scales),
                median_local_x_scale_ms=float(np.median(scales)) if scales else None,
                wrong_member_fraction=wrong_count / total_count if total_count else None,
            ))
        for baseline, method in METHOD_COMPARISONS:
            both = [(trials[*condition, seed, baseline]["error"], trials[*condition, seed, method]["error"])
                    for seed in SEEDS]
            differences = np.array([new - old for old, new in both if old is not None and new is not None])
            improved = int(np.sum(differences < -1e-9))
            worsened = int(np.sum(differences > 1e-9))
            comparisons.append(dict(
                **identifiers, baseline=baseline, method=method, n_total=len(both),
                n_both_finite=len(differences), n_improved=improved, n_worsened=worsened,
                n_tied=len(differences) - improved - worsened,
                median_paired_abs_error_difference_ms=float(np.median(differences)) if len(differences) else None,
            ))
    return summary, comparisons


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_ecdf(trials, font_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams.update({
        "font.family": font_name, "font.size": 10, "pdf.fonttype": 42, "axes.unicode_minus": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    main = {}
    for motion, truth, method in product(MOTIONS, TRUTHS, METHOD_STYLE):
        errors = [trials[motion, ("tracked", 0.0), truth, seed, method]["error"] for seed in SEEDS]
        main[motion, truth, method] = np.sort([error for error in errors if error is not None])
    right = 1.03 * max(float(values[-1]) for values in main.values() if len(values))
    fig, panels = plt.subplots(2, 2, figsize=(9.1, 6.6), sharex=True, sharey=True)
    for panel, (motion, truth) in zip(panels.flat, product(MOTIONS, TRUTHS)):
        for method, (color, style, _) in METHOD_STYLE.items():
            values = main[motion, truth, method]
            xs = np.r_[0, values, right]
            ys = np.r_[0, np.arange(1, len(values) + 1) / len(SEEDS), len(values) / len(SEEDS)]
            panel.step(xs, ys, where="post", color=color, linestyle=style, linewidth=2)
        panel.set_title(f"{MOTION_LABEL[motion]}，设定时偏 +{truth} ms", loc="left", fontweight="bold")
        panel.set(xlim=(0, right), ylim=(0, 1.03), yticks=np.linspace(0, 1, 5))
        panel.xaxis.set_major_locator(MaxNLocator(6))
        panel.grid(visible=True, color="#E0E4E8", linewidth=0.7)
        panel.set_axisbelow(True)

    handles = [Line2D([0], [0], color=color, linestyle=style, linewidth=2, label=label)
               for color, style, label in METHOD_STYLE.values()]
    fig.suptitle("AIRY 与 ARS408 仿真时偏误差", x=0.105, ha="left", y=0.982, fontsize=15, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.925), ncol=4, frameon=False)
    fig.supxlabel("终点绝对时偏误差（ms）", x=0.54, y=0.058)
    fig.supylabel("累计试验比例", x=0.026, y=0.48)
    fig.text(0.104, 0.020, "3–6 m 往返 8 趟，跟踪观测，四种方法使用相同配对成员。每组 30 次，缺失估计计入分母。",
             fontsize=9, color="#505861")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.14, top=0.823, wspace=0.125, hspace=0.24)
    fig.savefig(HERE / "timing_error_ecdf.png", dpi=400)
    fig.savefig(HERE / "timing_error_ecdf.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese-font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent.parent / "MultiSensor_TimeSync" / ".deps" / ".mplconfig"))
    import numpy as np

    trials = load_trials(HERE / "trials.csv")
    summary, comparisons = table_rows(trials, np)
    write_rows(HERE / "summary.csv", summary)
    write_rows(HERE / "paired_comparisons.csv", comparisons)
    plot_ecdf(trials, args.chinese_font)
    print(f"Validated {len(trials)} trials, wrote {len(summary)} summaries and {len(comparisons)} comparisons.")


if __name__ == "__main__":
    main()
