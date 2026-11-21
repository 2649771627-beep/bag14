"""Verify schedules, offsets, fit diagnostics, uncertainty scales and figure outputs."""

import csv
import json
from math import isclose, isfinite
from pathlib import Path
import re
import sys

import matplotlib
from matplotlib.image import imread
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "generated"
FIELD = ROOT / "outputs/airy_ars408_vehicle_r01"
CONTROLS = ROOT / "outputs/mst_airy_submission"
FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z")
EXCLUDE_JSON = {"source_sha256"}  # Prepared CSV byte hashes can depend on float formatting.


def compare(expected, actual, location, errors):
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"Different fields at {location}: {expected.keys() ^ actual.keys()}")
        for key in expected.keys() - EXCLUDE_JSON:
            compare(expected[key], actual[key], f"{location}.{key}", errors)
    elif isinstance(expected, list):
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare(left, right, f"{location}[{index}]", errors)
    elif isinstance(expected, float):
        if not isfinite(actual) or not isclose(expected, actual, abs_tol=1e-6, rel_tol=1e-8):
            raise AssertionError(f"{location}: expected {expected}, got {actual}")
        errors[0] = max(errors[0], abs(expected - actual))
        errors[1] += 1
    elif expected != actual:
        raise AssertionError(f"{location}: expected {expected!r}, got {actual!r}")


def compare_json(expected, actual):
    errors = [0.0, 0]
    old, new = [json.loads(path.read_text(encoding="utf-8-sig")) for path in (expected, actual)]
    compare(old, new, actual.name, errors)
    return {"floating_fields_compared": errors[1], "max_absolute_difference": errors[0]}


def csv_value(field, value):
    # Preserve source IDs, nanosecond epochs, sequence keys and schedule hashes exactly.
    if FLOAT_PATTERN.fullmatch(value) and "." in value and not field.endswith("_ns"):
        return float(value)
    if FLOAT_PATTERN.fullmatch(value) and "e" in value.lower():
        return float(value)
    return value


def compare_csv(expected, actual):
    errors = [0.0, 0]
    rows = 0
    with expected.open(encoding="utf-8-sig", newline="") as old:
        with actual.open(encoding="utf-8-sig", newline="") as new:
            expected_rows, actual_rows = csv.DictReader(old), csv.DictReader(new)
            if expected_rows.fieldnames != actual_rows.fieldnames:
                raise AssertionError(f"Different CSV fields in {actual.name}")
            for rows, (left, right) in enumerate(zip(expected_rows, actual_rows, strict=True), 1):
                for column in expected_rows.fieldnames:
                    a = csv_value(column, left[column])
                    b = csv_value(column, right[column])
                    compare(a, b, f"{actual.name}[{rows}].{column}", errors)
    return {"rows": rows, "floating_fields_compared": errors[1], "max_absolute_difference": errors[0]}


def main():
    generated_field = RESULTS / "field"
    checks = [
        (ROOT / "expected/airy_ars408_sim" / file, RESULTS / "simulation" / file)
        for file in ("trials.csv", "summary.csv", "paired_comparisons.csv")
    ]
    checks.extend((FIELD / file, generated_field / file) for file in
                  ("pairs.csv", "profile_compare.json", "profile_curves.csv",
                   "replay_trials.csv", "replay_summary.json"))
    checks.extend((path, generated_field / "tracks" / path.name)
                  for path in sorted((FIELD / "tracks").iterdir()) if path.suffix in (".csv", ".json"))
    checks.extend((CONTROLS / group / file, RESULTS / group / file)
                  for group, files in (
                      ("field_stability", ("field_stability.json",)),
                      ("factorial", ("factorial_trials.csv", "factorial_summary.csv", "factorial_summary.json")),
                  ) for file in files)

    checks.extend((ROOT / "expected/diagnostic_supplement" / file,
                   RESULTS / "diagnostic_supplement" / file) for file in (
                       "strict_opposite_trials.csv", "strict_opposite_summary.csv",
                       "simulation_scale_trials.csv", "simulation_scale_summary.csv",
                       "diagnostic_supplement.json",
                   ))
    filename = "augmented_baseline_results.json"
    checks.append((ROOT / "outputs/mst_q2_20261118/replay_extension" / filename,
                   RESULTS / "augmented" / filename))
    results = {}
    for expected, actual in checks:
        method = compare_csv if actual.suffix == ".csv" else compare_json
        name = actual.relative_to(RESULTS).as_posix()
        results[name] = method(expected, actual)
        print(f"Verified {name}", flush=True)
    figures = RESULTS / "figures"
    required_figures = [figures / (stem + "_en.png") for stem in
                        ("airy_error", "vehicle_tracks", "vehicle_profiles", "vehicle_perturbations",
                         "prior_common_shift", "prior_direction_ablation")]
    required_figures.append(RESULTS / "factorial/figures/factorial_en.png")
    if (CONTROLS / "field_stability/field_stability.png").is_file():
        required_figures.append(RESULTS / "field_stability/field_stability.png")
    for image in required_figures:
        if image.stat().st_size < 10000:
            raise AssertionError(f"Unexpected image size: {image.name}")
        pixels = imread(image)
        if pixels.shape[0] < 300 or float(np.std(pixels)) <= 0.01:
            raise AssertionError(f"Unexpected figure content: {image.name}")
    reported = {
        "passed": True,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "numeric_checks": results,
        "figures_rendered": [path.relative_to(RESULTS).as_posix() for path in required_figures],
    }
    (RESULTS / "verification.json").write_text(json.dumps(reported, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
