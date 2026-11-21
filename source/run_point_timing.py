"""Regenerate and check the AIRY fixed-pair manuscript evidence."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
from matplotlib.image import imread
import numpy as np

import verify_results


ROOT = Path(__file__).resolve().parent
POINT = ROOT / "outputs/airy_ars408_point_timing"
FIELD = ROOT / "outputs/airy_ars408_vehicle_r01"
EXPECTED = ROOT / "expected/point_timing"
sys.path.insert(0, str(POINT))

import compare_timing
from paper_update.figures import make_point_timing_figures as figures


ARMS = {
    "near_constant": "Near-constant",
    "variable": "Primary variable speed",
    "variable_fast": "Faster segment",
}
METHODS = {
    "original_paired": "Original paired",
    "motion_axis_paired": "Motion-axis paired",
    "robust_motion_axis_paired": "Robust motion-axis paired",
    "robust_motion_axis_framewise": "Same-model robust framewise",
}
PRIMARY = ("original_paired", "robust_motion_axis_paired")
TIMINGS = ("frame_header", "board_point")


def save_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_cell(record, name, key, signed=False):
    fmt = "+.2f" if signed else ".2f"
    return " → ".join(format(record[timing][name][key], fmt).replace("-", "−") for timing in TIMINGS)


def write_manuscript_tables(output):
    data = json.loads((output / "comparison_summary.json").read_text(encoding="utf-8"))
    table18, table_s3, table_s4 = [], [], []
    for key, motion in ARMS.items():
        arm = data["arms"][key]
        n = arm["schedule"]["n_pairs"]
        for method, label in METHODS.items():
            record = arm["methods"][method]
            prefix = {"Motion": motion, "EKF": label}
            cells = {
                "Terminal offset (ms)": pair_cell(record, "ekf", "estimate_ms", signed=True),
                "Leave-one range (ms)": pair_cell(record, "leave_out", "range_ms"),
            }
            support = f"{arm['schedule']['n_members']} frames" if method.endswith("framewise") else f"{n} pairs"
            table_s3.append({**prefix, "Support": support, **cells,
                             "Maximum change (ms)": pair_cell(record, "leave_out", "max_abs_change_ms")})
            if method in PRIMARY:
                main_label = "Original paired" if method == PRIMARY[0] else "Direction centre + Huber"
                table18.append({"Motion (pairs)": f"{motion} ({n})", "EKF": main_label, **cells})
            if "profile" in record[TIMINGS[0]]:
                grid = [record[timing]["profile"] for timing in TIMINGS]
                minima = [(f"{row['best_ms']:+.0f}".replace("-", "−")) + ("*" if row["grid_min_at_boundary"] else "")
                          for row in grid]
                losses = [f"{row['min_mean_cost']:.4f}" for row in grid]
                table_s4.append({"Motion": motion, "Static pair model": label,
                                 "Minimum (ms)": " → ".join(minima), "Mean loss per pair": " → ".join(losses)})
    save_rows(output / "table18.csv", table18)
    save_rows(output / "table_s3.csv", table_s3)
    save_rows(output / "table_s4.csv", table_s4)


def verify_replay(output, chart_files):
    numerics = {}
    for name in ("comparison_summary.json", "replay_comparison.csv", "stability.csv", "cost_curves.csv"):
        compare = verify_results.compare_csv if name.endswith(".csv") else verify_results.compare_json
        numerics[name] = compare(EXPECTED / name, output / name)
    for name, count in (("table18.csv", 6), ("table_s3.csv", 12), ("table_s4.csv", 9)):
        with (output / name).open(encoding="utf-8-sig", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == count
    frozen = json.loads((EXPECTED / "figure_reference.json").read_text(encoding="utf-8"))
    rendered = {}
    for image in chart_files:
        pixels = imread(image)
        ref = frozen[image.name]
        assert list(pixels.shape[:2]) == ref["pixels_height_width"] and np.std(pixels) > 0.01
        digest = sha256(image.read_bytes()).hexdigest()
        rendered[image.name] = {"sha256": digest, "manuscript_source_sha256": ref["sha256"],
                                "byte_identical_to_manuscript_source": digest == ref["sha256"]}
    report = {"passed": True, "numeric_checks": numerics, "figures_rendered": rendered,
              "generated_tables": ["table18.csv", "table_s3.csv", "table_s4.csv"],
              "python_version": sys.version.split()[0], "numpy_version": np.__version__,
              "matplotlib_version": matplotlib.__version__}
    (output / "point_timing_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified all {len(numerics)} AIRY point-timing numerical records and rendered {len(rendered)} figures",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cjk-font", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    compare_timing.run(args.tracks, args.output)
    write_manuscript_tables(args.output)
    rendered = figures.render(args.output, args.output / "figures", args.cjk_font)
    verify_replay(args.output, rendered)


if __name__ == "__main__":
    main()
