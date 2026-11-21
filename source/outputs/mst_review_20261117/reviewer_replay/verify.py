"""Check regenerated main evidence against the archived numerical outputs."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
NUMERIC_ENDPOINT_COLUMNS = (
    "receive_s",
    "offset_ms",
    "spatial_response_ms",
    "injection_recovery_error_ms",
)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare_numeric_tree(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            compare_numeric_tree(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected, strict=True):
            compare_numeric_tree(first, second)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)
    else:
        assert actual == expected, (actual, expected)


def check_recorded(output):
    expected_directory = ROOT / "expected/recorded"
    summary = load_json(output / "summary.json")
    expected = load_json(expected_directory / "summary.json")
    assert len(summary["rows"]) == 8
    compare_numeric_tree(summary["rows"], expected["rows"])
    compare_numeric_tree(summary["original_profile_config"], expected["original_profile_config"])
    assert summary["innovation2_changed"] is False

    count = 0
    maximum = 0.0
    with (output / "endpoints.csv").open(encoding="utf-8", newline="") as actual_handle:
        with (expected_directory / "endpoints.csv").open(encoding="utf-8", newline="") as expected_handle:
            actual_reader = csv.DictReader(actual_handle)
            expected_reader = csv.DictReader(expected_handle)
            assert actual_reader.fieldnames == expected_reader.fieldnames
            for actual, expected in zip(actual_reader, expected_reader, strict=True):
                count += 1
                for key, value in expected.items():
                    if key in NUMERIC_ENDPOINT_COLUMNS and value:
                        difference = abs(float(actual[key]) - float(value))
                        maximum = max(maximum, difference)
                        assert difference <= 1e-8, (count, key, difference)
                    else:
                        assert actual[key] == value, (count, key, actual[key], value)
    assert count == 4440
    return {"summary_rows": len(summary["rows"]), "endpoint_rows": count, "max_numeric_delta": maximum}


def check_simulation(output):
    archive = ROOT / "expected/simulation"
    actual = load_json(output / "results.json")
    expected = load_json(archive / "results.json")
    for key in (
        "trial_count", "seed", "bootstrap_seed", "bootstrap_repetitions",
        "offset_truth_ms", "drift_truth_sps", "motion_designs", "config",
    ):
        compare_numeric_tree(actual[key], expected[key])
    compare_numeric_tree(
        load_json(output / "terminal_affine_verification.json"),
        load_json(archive / "terminal_affine_verification.json"),
    )
    maximum = 0.0
    with np.load(output / "signed_endpoint_errors_ms.npz", allow_pickle=False) as actual_errors:
        with np.load(archive / "signed_endpoint_errors_ms.npz", allow_pickle=False) as expected_errors:
            assert actual_errors.files == expected_errors.files
            for name in expected_errors.files:
                difference = np.max(np.abs(actual_errors[name] - expected_errors[name]))
                maximum = max(maximum, float(difference))
                assert difference <= 1e-8, (name, difference)
            arrays = len(expected_errors.files)
    pooled = json.loads(subprocess.check_output(
        [sys.executable, "-I", "-B", str(ROOT / "simulation/check_gain.py"),
         str(output / "signed_endpoint_errors_ms.npz")],
        text=True,
    ))
    compare_numeric_tree(pooled, load_json(archive / "pooled_simulation_gain.json"))
    return {
        "trials_per_design": actual["trial_count"],
        "compared_arrays": arrays,
        "max_error_delta_ms": maximum,
        "pooled_simulation_gain": pooled,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-output", type=Path, default=ROOT / "replayed_08_09")
    parser.add_argument("--simulation-output", type=Path, default=ROOT / "simulation")
    args = parser.parse_args()
    print(json.dumps({
        "recorded": check_recorded(args.recorded_output),
        "simulation": check_simulation(args.simulation_output),
    }, indent=2))


if __name__ == "__main__":
    main()
