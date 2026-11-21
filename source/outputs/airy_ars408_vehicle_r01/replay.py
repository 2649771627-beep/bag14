"""Matched-observation replay of the R01 AIRY and ARS408 vehicle tracks.

CSV times must already share the explicitly selected clock convention and are
expressed in seconds relative to the common origin. Positions use vehicle XY.
Only analysis replay receives artificial spatial shifts or timestamp increments.
The unmodified recording has no independently measured clock offset.
"""

import argparse
import csv
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
VEHICLE = OUTPUTS.parent / "MultiSensor_TimeSync"
ARCHIVE = OUTPUTS / "mst_review_20261117/reviewer_replay/bundle/core"
sys.path[:0] = [str(VEHICLE), str(ARCHIVE), str(OUTPUTS / "direction_center_ekf_4_10m")]

from experiments.paired_ekf import match_pairs, select_observations
from observability import paired_information
from src.ekf_estimator.center_filter import DirectionCenterPairedEKF, reference_axis
from src.ekf_estimator.paired import PairedTemporalEKF
from src.temporal_model.vehicle import TargetReference, UnsupportedReference, VehicleObservationModel


CONFIG_PATH = ARCHIVE / "inputs/filter_config.json"
ARMS = ("near_constant", "variable", "variable_fast")
METHODS = ("original_paired", "motion_axis_paired", "robust_motion_axis_paired",
           "robust_motion_axis_framewise")
CASES = (
    ("nominal", 0.0, False, 0.0),
    ("common_minus_8cm", -0.08, False, 0.0),
    ("common_plus_8cm", 0.08, False, 0.0),
    ("direction_minus_8cm", -0.08, True, 0.0),
    ("direction_plus_8cm", 0.08, True, 0.0),
    ("time_minus_30ms", 0.0, False, -30.0),
    ("time_plus_30ms", 0.0, False, 30.0),
)


def read_numeric(path, columns):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {path.name}: {columns}")
        rows = np.asarray([[float(row[key]) for key in columns] for row in reader], dtype=float)
    rows = rows.reshape(-1, len(columns))
    if not np.isfinite(rows).all():
        raise ValueError(f"Nonfinite values in {path.name}")
    return rows


def load_arm(tracks, arm, cfg):
    lidar = read_numeric(tracks / f"{arm}_lidar.csv", ("t_s", "x_m", "y_m"))
    radar = read_numeric(tracks / f"{arm}_radar.csv", ("reported_s", "receive_s", "x_m", "y_m"))
    reference = TargetReference([dict(t=t, x=x, y=y, z=0.0) for t, x, y in lidar],
                                cfg["reference_gap_s"], cfg["reference_smooth_half_window_s"])
    model = VehicleObservationModel((np.eye(4), np.eye(4), np.eye(3), np.zeros(4)))
    axis, energy = reference_axis(reference)
    rows = [dict(t=t, bag_record_ns=round(receive * 1e9), x=x, y=y)
            for t, receive, x, y in radar]
    events, positions, slopes = select_observations(rows, "radar", 0, cfg, reference, model)
    return reference, model, axis, energy, events, positions, slopes.reshape(-1, 2), len(radar)


def comparison(method, events, pairs, directions, axis, reference, model, cfg, prior, terminal,
               *, predict_to_terminal=False):
    if not pairs:
        return dict(estimate_ms=None, candidate_offset_ms=None, offset_posterior_std_ms=None,
                    center_m=None, offset_center_correlation=None, failed_updates=0,
                    valid_updates=0, downweighted_updates=0, output_status="no_eligible_pairs")
    args = (cfg["sensors"], cfg["filter"], reference, model)
    if method == "original_paired":
        ekf = PairedTemporalEKF(*args)
    else:
        ekf = DirectionCenterPairedEKF(*args, axis=axis, robust=method.startswith("robust"),
                                       center_std_m=0.15)
    ekf.predict_to(prior)
    framewise = method.endswith("framewise")
    updates = [(i, None) for i in sorted(i for pair in pairs for i in pair)] if framewise else pairs
    geom = cfg["sensors"]["radar"]["geometry_sigma"]
    failures = downweighted = 0
    last = None
    for i, j in updates:
        try:
            if framewise:
                last = ekf.step(events[i], directions[i])
            elif method == "original_paired":
                last = ekf.step_pair(events[i], events[j], geom)
            else:
                last = ekf.step_pair(events[i], events[j], geom, directions[i], directions[j])
            downweighted += int(last.get("innovation_weight", 1.0) < 1.0 - 1e-12)
        except UnsupportedReference:
            failures += 1
    if predict_to_terminal and ekf.last_receive_s < terminal:
        ekf.predict_to(terminal)
    assert np.isclose(ekf.last_receive_s, terminal, rtol=0, atol=1e-9)
    candidate = 1000.0 * float(ekf.x[0])
    available = last is not None and failures == 0
    return dict(estimate_ms=candidate if available else None, candidate_offset_ms=candidate,
                offset_posterior_std_ms=1000.0 * float(np.sqrt(ekf.P[0, 0])) if available else None,
                center_m=last.get("center_m") if available else None,
                offset_center_correlation=last.get("offset_center_corr") if available else None,
                failed_updates=failures, valid_updates=len(updates) - failures,
                downweighted_updates=downweighted,
                output_status="computed" if available else "unsupported_reference_update")


def run_arm(tracks, arm, cfg):
    reference, model, axis, energy, events, positions, slopes, radar_count = load_arm(tracks, arm, cfg)
    sensitivity = deepcopy(cfg)
    sensitivity["sensors"]["radar"]["min_slope"] = 0.25
    directions = -np.sign(slopes @ axis)
    coverage = []
    pair_rows = []
    trials = []
    for label, settings in (("primary", cfg), ("min_speed_0p25_sensitivity", sensitivity)):
        pairs = match_pairs(events, positions, slopes, "radar", settings)
        members = sorted(i for pair in pairs for i in pair)
        assert len(set(members)) == len(members)
        assert not pairs or members[-1] == pairs[-1][1]
        prior = events[members[0]].receive_s if pairs else None
        terminal = events[members[-1]].receive_s if pairs else None
        pair_times = np.asarray([(events[i].receive_s, events[j].receive_s) for i, j in pairs], dtype="<f8")
        sums = [float(abs(slopes[i] @ axis) + abs(slopes[j] @ axis)) for i, j in pairs]
        info = paired_information(events, pairs, slopes, settings)
        summary = dict(arm=arm, pair_rule=label, min_speed_mps=settings["sensors"]["radar"]["min_slope"],
                       n_lidar_reference_knots=len(reference.times), n_radar_input=radar_count,
                       n_reference_supported_decimated_events=len(events),
                       n_events_above_pair_speed=int(np.sum(np.linalg.norm(slopes, axis=1) >= settings["sensors"]["radar"]["min_slope"])),
                       n_pairs=len(pairs), n_members=len(members), prior_receive_s=prior, terminal_receive_s=terminal,
                       pair_schedule_sha256=sha256(pair_times.tobytes()).hexdigest(),
                       motion_axis_x=float(axis[0]), motion_axis_y=float(axis[1]), axis_energy_fraction=energy,
                       speed_sum_mean_mps=float(np.mean(sums)) if pairs else None,
                       speed_sum_std_mps=float(np.std(sums)) if pairs else None,
                       n_same_axis_direction_pairs=sum(int(directions[i] == directions[j]) for i, j in pairs),
                       **info)
        coverage.append(summary)
        for k, (i, j) in enumerate(pairs, 1):
            pair_rows.append(dict(arm=arm, pair_rule=label, pair_number=k,
                                  early_id=i, late_id=j, early_receive_s=events[i].receive_s,
                                  late_receive_s=events[j].receive_s, early_reported_s=events[i].reported_s,
                                  late_reported_s=events[j].reported_s, early_direction=int(directions[i]),
                                  late_direction=int(directions[j]), early_reference_x_m=positions[i][0],
                                  late_reference_x_m=positions[j][0], speed_sum_mps=sums[k - 1]))
        outputs = {}
        for case, amplitude, signed, injection_ms in CASES:
            injected = events if case == "nominal" else [
                replace(e, reported_s=e.reported_s + injection_ms / 1000.0,
                        value=e.value + amplitude * (directions[e.id] if signed else 1.0) * axis)
                for e in events]
            for method in METHODS:
                result = comparison(method, injected, pairs, directions, axis, reference, model,
                                    settings, prior, terminal)
                nominal = outputs.get((method, "nominal"))
                response = None if nominal is None or result["estimate_ms"] is None else result["estimate_ms"] - nominal
                outputs[method, case] = result["estimate_ms"]
                trials.append(dict(arm=arm, pair_rule=label, method=method, case=case,
                                   n_pairs=len(pairs), n_members=len(members), prior_receive_s=prior,
                                   terminal_receive_s=terminal, pair_schedule_sha256=summary["pair_schedule_sha256"],
                                   injected_space_m=amplitude, space_changes_sign_by_direction=signed,
                                   injected_time_ms=injection_ms, **result, response_from_nominal_ms=response,
                                   injected_increment_recovery_error_ms=response - injection_ms
                                   if injection_ms and response is not None else None))
    return coverage, pair_rows, trials


def write_csv(path, rows, default_fields=()):
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(rows[0]) if rows else default_fields
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_responses(trials):
    grouped = {}
    for row in trials:
        key = (row["arm"], row["pair_rule"], row["method"])
        grouped.setdefault(key, {})[row["case"]] = row
    results = []
    for (arm, rule, method), cases in grouped.items():
        nominal = cases["nominal"]
        row = dict(arm=arm, pair_rule=rule, method=method, n_pairs=nominal["n_pairs"],
                   candidate_offset_ms=nominal["estimate_ms"], nominal_status=nominal["output_status"])
        for prefix, metric in (("common", "response_from_nominal_ms"),
                               ("direction", "response_from_nominal_ms"),
                               ("time", "injected_increment_recovery_error_ms")):
            values = [trial[metric] for case, trial in cases.items() if case.startswith(prefix + "_")
                      and trial[metric] is not None]
            row[prefix + "_signs_available"] = len(values)
            row[prefix + "_mean_abs_ms"] = float(np.mean(np.abs(values))) if values else None
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=HERE / "tracks")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    coverage, pairs, trials = [], [], []
    for arm in ARMS:
        if arm == "variable_fast" and not (args.tracks / f"{arm}_radar.csv").exists():
            continue
        c, p, t = run_arm(args.tracks, arm, cfg)
        coverage.extend(c)
        pairs.extend(p)
        trials.extend(t)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "replay_trials.csv", trials)
    write_csv(args.output / "pairs.csv", pairs, ("arm", "pair_rule", "pair_number", "early_id", "late_id"))
    summary = dict(recorded_offset_truth_available=False,
                   paired_local_information_axis="vehicle_x_paper_diagnostic",
                   comparison_epoch="last completed pair in each arm and pair-rule branch",
                   filter_config_sha256=sha256(CONFIG_PATH.read_bytes()).hexdigest(), coverage=coverage,
                   responses=summarize_responses(trials), trials=trials)
    (args.output / "replay_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for row in coverage:
        print(f"{row['arm']} {row['pair_rule']}: {row['n_pairs']} pairs, "
              f"{row['n_members']}/{row['n_reference_supported_decimated_events']} eligible members")


if __name__ == "__main__":
    main()
