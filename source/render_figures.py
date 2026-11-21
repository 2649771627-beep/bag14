"""Render the English manuscript figures from this archive's numeric outputs."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def load_module(name, file):
    specification = spec_from_file_location(name, file)
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def render_field(field, destination):
    module = load_module("field_plots", ROOT / "outputs/mst_airy_vehicle_revision/make_vehicle_figures.py")
    module.FIGURES = destination
    module.SOURCE = field
    labels = module.LABELS["en"]
    with matplotlib.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9.4, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.65,
        "figure.labelsize": 9.6,
    }):
        clips, tracks = module.track_data()
        curves, count = module.profile_data()
        module.plot_tracks("en", labels, clips, tracks)
        module.plot_profiles("en", labels, curves, count)
        module.plot_perturbations("en", labels, module.replay_data())


def render_prior(augmented, destination):
    module = load_module("prior_plots", ROOT / "outputs/mst_airy_vehicle_revision/make_prior_figures.py")
    module.SIMULATION = augmented

    def save_english(figure, name):
        figure.savefig(destination / f"{name}.png", dpi=350)
        figure.savefig(destination / f"{name}.pdf")
        plt.close(figure)

    module.save = save_english
    with matplotlib.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.labelsize": 9.4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "hatch.linewidth": 0.5, "pdf.fonttype": 42,
    }):
        labels = module.LABELS["en"]
        module.draw_simulation("en", labels, module.read_simulation())
        module.draw_ablation("en", labels, module.read_ablation())


def render_simulation(trials, destination):
    path = ROOT / "outputs/mst_airy_ars408_revision/make_airy_figure.py"
    # Execute the published English plotting branch with local input and output paths.
    source = path.read_text(encoding="utf-8")
    changes = {
        'SIMULATION / "trials.csv"': 'arguments.trials',
        'figures = HERE / "figures"': 'figures = arguments.figure_dir',
        'figures.mkdir(exist_ok=True)': 'figures.mkdir(parents=True, exist_ok=True)',
        '    font_manager.fontManager.addfont(arguments.chinese_font)\n'
        '    chinese_name = font_manager.FontProperties(fname=arguments.chinese_font).get_name()\n'
        '    fonts = {"en": "Times New Roman", "zh": chinese_name}':
        '    fonts = {"en": "DejaVu Serif"}',
        'for language, labels in LOCALIZED.items():': 'for language, labels in (("en", LOCALIZED["en"]),):',
        '    arguments = parser.parse_args()':
        '    parser.add_argument("--trials", type=Path, required=True)\n'
        '    parser.add_argument("--figure-dir", type=Path, required=True)\n'
        '    arguments = parser.parse_args()',
    }
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    previous_argv = sys.argv
    sys.argv = [str(path), "--trials", str(trials), "--figure-dir", str(destination)]
    try:
        exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    finally:
        sys.argv = previous_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    render_simulation(args.simulation, args.figures)
    render_field(args.field, args.figures)
    render_prior(args.augmented, args.figures)


if __name__ == "__main__":
    main()
