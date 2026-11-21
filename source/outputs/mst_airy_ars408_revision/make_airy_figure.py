"""Render the manuscript's bilingual AIRY–ARS408 error distributions."""

import argparse
from itertools import product
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SIMULATION = HERE.parent / "airy_ars408_sim"
sys.path.insert(0, str(SIMULATION))
from summarize import METHOD_STYLE, MOTIONS, SEEDS, TRUTHS, load_trials


LOCALIZED = {
    "en": {
        "methods": ("Original paired", "Motion-axis paired", "Robust paired", "Robust framewise"),
        "motions": ("Near-constant speed", "Variable speed"),
        "x": "Absolute terminal offset error (ms)",
        "y": "Cumulative fraction of trials",
    },
    "zh": {
        "methods": ("原配对", "运动主轴配对", "稳健主轴配对", "稳健主轴逐帧"),
        "motions": ("近恒速", "变速"),
        "x": "终点绝对时偏误差（ms）",
        "y": "累计试验比例",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chinese-font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    arguments = parser.parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR", str(HERE.parent.parent / "MultiSensor_TimeSync" / ".deps" / ".mplconfig")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import numpy as np

    font_manager.fontManager.addfont(arguments.chinese_font)
    chinese_name = font_manager.FontProperties(fname=arguments.chinese_font).get_name()
    fonts = {"en": "Times New Roman", "zh": chinese_name}
    trials = load_trials(SIMULATION / "trials.csv")
    errors = {}
    for motion, truth, method in product(MOTIONS, TRUTHS, METHOD_STYLE):
        values = [trials[motion, ("tracked", 0.0), truth, seed, method]["error"] for seed in SEEDS]
        errors[motion, truth, method] = np.sort([value for value in values if value is not None])
    right = 1.02 * max(values[-1] for values in errors.values() if len(values))

    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    for language, labels in LOCALIZED.items():
        with matplotlib.rc_context({
            "font.family": fonts[language],
            "font.size": 8.7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
        }):
            figure, panels = plt.subplots(2, 2, figsize=(6.60, 4.35), sharex=True, sharey=True)
            for index, (panel, (motion, truth)) in enumerate(zip(panels.flat, product(MOTIONS, TRUTHS))):
                for method, (color, style, _) in METHOD_STYLE.items():
                    values = errors[motion, truth, method]
                    xs = np.r_[0, values, right]
                    ys = np.r_[0, np.arange(1, len(values) + 1), len(values)] / len(SEEDS)
                    panel.step(xs, ys, where="post", color=color, linestyle=style, linewidth=1.35)
                motion_label = labels["motions"][MOTIONS.index(motion)]
                panel.set_title(f"({chr(97 + index)}) {motion_label}, +{truth} ms", loc="left", pad=5)
                panel.set(xlim=(0, right), ylim=(0, 1.04))
                panel.xaxis.set_major_locator(MultipleLocator(100))
                panel.yaxis.set_major_locator(MultipleLocator(0.25))
                panel.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                panel.tick_params(length=3, width=0.6)
                panel.grid(visible=True, color="#DDE2E5", linewidth=0.45)
                panel.set_axisbelow(True)

            handles = [
                Line2D([0], [0], color=color, linestyle=style, linewidth=1.4, label=label)
                for (color, style, _), label in zip(METHOD_STYLE.values(), labels["methods"])
            ]
            figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.991),
                          ncol=4, frameon=False, handlelength=2.5, columnspacing=1.1)
            figure.supxlabel(labels["x"], x=0.54, y=0.028, fontsize=9.1)
            figure.supylabel(labels["y"], x=0.023, y=0.51, fontsize=9.1)
            figure.subplots_adjust(left=0.105, right=0.985, bottom=0.124, top=0.875,
                                   wspace=0.13, hspace=0.32)
            destination = figures / f"airy_error_{language}"
            figure.savefig(destination.with_suffix(".png"), dpi=350)
            figure.savefig(destination.with_suffix(".pdf"))
            plt.close(figure)

    for condition, values in errors.items():
        if len(values) < len(SEEDS):
            print(f"{condition}: {len(values)}/{len(SEEDS)} finite errors")
    print(f"Validated {len(trials)} records. Both versions plotted through {right:.1f} ms.")


if __name__ == "__main__":
    main()
