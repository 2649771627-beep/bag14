"""Reproduce the reindexed Radar Ghost summaries and figures from archived results."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED = ROOT / "expected/public_evidence"
POLICIES = ("original_130ms", "nominal_100ms", "native_frames")
METHODS = ("difference", "raw_completed_pairs")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sampling_rows(data):
    rows = []
    for stream in data:
        row = {field: stream[field] for field in ("recording", "sequence", "radar")}
        for policy in POLICIES:
            pairs = stream["pairs"][policy]
            endpoints = {member for pair in pairs for member in pair}
            if len(endpoints) != 2 * len(pairs):
                raise ValueError(f"Reused endpoints in {row['recording']} / {row['radar']} / {policy}")
            row[policy + "_pairs"] = len(pairs)
        row["native_observations"] = 2 * row["native_frames_pairs"]
        rows.append(row)
    return rows


def evidence_curves(result):
    pairs = result["frozen"]["pairs"]
    members = {str(member["event_id"]): member for member in result["frozen"]["members"]}
    ids = [str(pair["late_id"]) for pair in pairs]
    rows = []
    for method in METHODS:
        for case, injections in result["replays"].items():
            for injection, replay in injections.items():
                source = replay["outputs"][method]
                nominal = result["replays"][case]["0"]["outputs"][method]
                baseline = result["replays"]["nominal"]["0"]["outputs"][method]
                for event_id in ids:
                    estimate = source[event_id]["offset_s"] * 1000
                    rows.append({
                        "event_id": event_id, "endpoint_s": members[event_id]["receive_s"],
                        "method": method, "spatial_case": case, "injection_ms": int(injection),
                        "offset_ms": estimate,
                        "spatial_change_ms": estimate - baseline[event_id]["offset_s"] * 1000 if injection == "0" else "",
                        "increment_error_ms": estimate - nominal[event_id]["offset_s"] * 1000 - int(injection),
                    })
    return rows


def rms(values):
    return math.sqrt(math.fsum(value * value for value in values) / len(values))


def check_metrics(result, curves):
    """Pool both signs on identical endpoints and validate the original Table 8 metrics."""
    checks = []
    for method in METHODS:
        def displacement(prefix):
            return rms([row["spatial_change_ms"] for row in curves
                        if row["method"] == method and row["injection_ms"] == 0
                        and row["spatial_case"].startswith(prefix)])

        common = displacement("common_")
        direction = displacement("direction_")
        increment = rms([row["increment_error_ms"] for row in curves
                         if row["method"] == method and row["injection_ms"]
                         and row["spatial_case"] == "nominal"])
        expected = (
            result["pooled_displacement_change_rmse_ms"]["common"][method],
            result["pooled_displacement_change_rmse_ms"]["direction"][method],
            result["metrics"]["nominal"][method]["recovery_pooled_rmse_ms"],
        )
        if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
               for a, b in zip((common, direction, increment), expected)):
            raise ValueError(f"Frozen G01-C/left metrics differ for {method}")
        checks.append({"recording": "G01-C / left", "pairs": len(result["frozen"]["pairs"]),
                       "observations": len(result["frozen"]["members"]), "method": method,
                       "common_rms_ms": f"{common:.3f}", "direction_rms_ms": f"{direction:.3f}",
                       "increment_rmse_ms": f"{increment:.3f}"})
    return checks


def run(output):
    from public_evidence_render import render

    output.mkdir(parents=True, exist_ok=True)
    sampling = sampling_rows(read(EXPECTED / "sampling_pairs.json"))
    sampling_tables = {
        "table5_public.csv": [row for row in sampling if row["native_frames_pairs"] >= 15],
        "table_s5.csv": [row for row in sampling if row["native_frames_pairs"] <= 1],
    }
    if len(sampling_tables["table5_public.csv"]) != 1 or len(sampling_tables["table_s5.csv"]) != 4:
        raise ValueError("Radar Ghost evidence split differs from the frozen stream summary")
    for name, rows in sampling_tables.items():
        write_csv(output / name, rows)
    result = read(EXPECTED / "g01_c_left_replay.json")
    curves = evidence_curves(result)
    checks = check_metrics(result, curves)
    write_csv(output / "table8_g01_c_left.csv", checks)
    write_csv(output / "g01_c_left_curves.csv", curves)
    images = render(result, output)
    figure_hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in images}
    reference = read(EXPECTED / "supplement_figure_hashes.json")
    report = {"passed": True, "sampling_streams_verified": len(sampling),
              "table5_public_rows": len(sampling_tables["table5_public.csv"]),
              "table_s5_rows": len(sampling_tables["table_s5.csv"]),
              "table8_g01_methods_verified": len(checks), "curve_rows": len(curves),
              "figures_rendered": sorted(figure_hashes),
              "png_matches_archived": {name: figure_hashes[name] == reference[name] for name in figure_hashes},
              "png_sha256": figure_hashes}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
