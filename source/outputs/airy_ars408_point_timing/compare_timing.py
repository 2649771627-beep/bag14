"""Change only the AIRY observation epochs on the frozen R01 pair schedules."""

from collections import defaultdict
from copy import copy
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
FIELD = HERE.parent / "airy_ars408_vehicle_r01"
sys.path[:0] = [str(FIELD), str(HERE.parent / "mst_airy_submission/factorial")]

import replay
from profile_compare import profile
from run_factorial import archive_group, archive_tables, read_csv, validate_against_archive


TIMINGS = ("frame_header", "board_point")
PROFILE_METHODS = tuple(method for method in replay.METHODS if method.endswith("paired"))
GRID_S = np.linspace(-0.60, 0.60, 1201)


def read_board_times():
    samples = defaultdict(dict)
    for row in read_csv(HERE / "board_point_support.csv"):
        arm = row.pop("recording")
        receipt = int(row["receive_ns"])
        assert receipt not in samples[arm]
        samples[arm][receipt] = row
    checks = {}
    for arm, observations in samples.items():
        old_rows = read_csv(FIELD / "raw" / f"{arm}_lidar.csv")
        assert len(old_rows) == len(observations)
        for old in old_rows:
            new = observations[int(old["receive_ns"])]
            for key in ("stamp_ns", "n_points"):
                assert int(old[key]) == int(new[key])
            for key in ("x_m", "y_m", "z_m", "point_dt_s"):
                assert np.isclose(float(old[key]), float(new[key]), rtol=0, atol=1e-10)
        checks[arm] = len(observations)
    assert checks == {"near_constant": 3303, "variable": 3451}
    return samples, checks


def board_reference(arm, baseline, board_samples, settings, tracks):
    rows = read_csv(tracks / f"{arm}_lidar.csv")
    offsets = np.array([float(board_samples[int(row["raw_receive_ns"])]["point_dt_s"]) for row in rows])
    assert np.all((offsets >= 0) & (offsets < 0.11))
    times = baseline.times + offsets
    intervals = np.diff(times)
    assert np.all(intervals > 0)
    assert np.array_equal(np.diff(baseline.times) > baseline.max_gap, intervals > baseline.max_gap)
    nodes = [dict(t=t, x=float(row["x_m"]), y=float(row["y_m"]), z=0.0) for t, row in zip(times, rows)]
    reference = replay.TargetReference(nodes, baseline.max_gap, settings["reference_smooth_half_window_s"])
    audit = {"knots": len(rows), "offset_min_median_max_ms": list(1000 * np.quantile(offsets, (0, .5, 1))),
             "new_knot_gap_min_max_ms": [1000 * intervals.min(), 1000 * intervals.max()],
             "new_knot_gaps_under_20ms": int(np.sum(intervals < .02)),
             "new_knot_gaps_over_180ms": int(np.sum(intervals > .18))}
    return reference, offsets, audit


def pair_profile(reference, events, pairs, directions, axis, config, method):
    early, late = np.asarray(pairs).T
    t = np.array([event.reported_s for event in events])
    z = np.array([event.value for event in events])
    settings = config["sensors"]["radar"]
    noise = np.sqrt(2*np.square(settings["sigma"]) + np.square(settings["geometry_sigma"]))
    shift = ((directions[late] - directions[early])[:, None] * axis / noise).reshape(-1, 1)
    measured_difference = z[late] - z[early]

    def cost(offset):
        p, _, valid = profile.reference_many(reference, np.r_[t[early], t[late]] - offset)
        if not np.all(valid):
            return np.inf
        predicted_difference = p[len(pairs):, :2] - p[:len(pairs), :2]
        residual = ((measured_difference - predicted_difference) / noise).ravel()
        if method == "robust_motion_axis_paired":
            return profile.robust_profile(residual, shift, delta=1.5)[0] / len(pairs)
        if method == "motion_axis_paired":
            residual -= shift[:, 0] * float(np.linalg.lstsq(shift, residual, rcond=None)[0][0])
        return float(residual @ residual) / (2 * len(pairs))

    return cost


def fit_curve(cost):
    values = np.array([cost(offset) for offset in GRID_S])
    if not np.all(np.isfinite(values)):
        raise ValueError("A frozen endpoint lost support on the common profile grid")
    k = int(np.argmin(values))
    local = {"best_ms": float(GRID_S[k]*1000), "min_mean_cost": float(values[k]),
             "grid_min_at_boundary": k in (0, len(GRID_S)-1)}
    return values, local


def export_stability(arm, references, events, groups, pairs, directions, axis, model, settings, schedule):
    rows = []
    for omitted, subset in groups:
        for timing, reference in references.items():
            for method in replay.METHODS:
                trial = replay.comparison(method, events, subset, directions, axis, reference, model, settings,
                                          schedule["prior_receive_s"], schedule["terminal_receive_s"],
                                          predict_to_terminal=True)
                rows.append(dict(arm=arm, method=method, timing=timing, omitted_trip=omitted, n_pairs=len(subset),
                                 n_pairs_removed=len(pairs)-len(subset), estimate_ms=trial["estimate_ms"],
                                 status=trial["output_status"], failed_updates=trial["failed_updates"],
                                 posterior_std_ms=trial["offset_posterior_std_ms"],
                                 prior_receive_s=schedule["prior_receive_s"], terminal_receive_s=schedule["terminal_receive_s"]))
    return rows


def stability_summary(rows, arm, method, timing, nominal):
    relevant = [row for row in rows if row["arm"] == arm and row["method"] == method and row["timing"] == timing
                and row["n_pairs_removed"] > 0]
    available = [row["estimate_ms"] for row in relevant if row["estimate_ms"] is not None]
    return {"informative_omissions": len(relevant), "computed": len(available),
            "min_ms": min(available) if available else None, "max_ms": max(available) if available else None,
            "range_ms": max(available)-min(available) if available else None,
            "median_abs_change_ms": float(np.median(np.abs(np.array(available)-nominal))) if available else None,
            "max_abs_change_ms": float(np.max(np.abs(np.array(available)-nominal))) if available else None}


def run(tracks, out):
    board_samples, extraction_checks = read_board_times()
    config = json.loads(replay.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    preparation = json.loads((tracks / "preparation.json").read_text(encoding="utf-8-sig"))
    pair_rows, originals = archive_tables(FIELD)
    arms = {}
    for arm in replay.ARMS:
        old, model, axis, _, events, _, slopes, _ = replay.load_arm(tracks, arm, config)
        source = "variable" if arm == "variable_fast" else arm
        new, offsets, timing_audit = board_reference(arm, old, board_samples[source], config, tracks)
        directions = -np.sign(slopes @ axis)
        pairs, schedule = archive_group(arm, "primary", pair_rows[arm, "primary"], events, directions, replay, originals)
        arms[arm] = old, new, offsets, timing_audit, model, axis, events, directions, pairs, schedule

    constant = float(np.median(np.concatenate([arms[arm][2] for arm in ("near_constant", "variable")])))
    summary = {"raw_roundtrip_track_matches": extraction_checks, "common_shift_ms": 1000 * constant,
               "clock_anchor_ns": preparation["clock_mapping"]["device_to_host_offset_ns"], "arms": {}}
    trial_rows, stability_rows, curve_rows = [], [], []
    for arm, (old, new, _, timing_audit, model, axis, events, directions, pairs, schedule) in arms.items():
        references = dict(zip(TIMINGS, (old, new)))
        result = dict(schedule=schedule, timing=timing_audit, methods={})
        summary["arms"][arm] = result
        pad = preparation["clip_edge_pad_s"]
        groups = []
        for i, trip in enumerate(preparation["clips"][arm]["trips"], start=1):
            begin = trip["near_departure_s"] - pad
            end = trip["near_return_s"] + pad
            subset = [pair for pair in pairs if not any(begin <= events[j].receive_s <= end for j in pair)]
            groups.append((i, subset))
        arm_stability = export_stability(arm, references, events, groups, pairs, directions, axis, model, config, schedule)
        stability_rows.extend(arm_stability)
        result["trip_pairs_removed"] = [len(pairs)-len(subset) for _, subset in groups]

        uniform = copy(old)
        uniform.times = old.times + constant
        for method in replay.METHODS:
            method_result = {}
            for timing, ref in references.items():
                trial = replay.comparison(method, events, pairs, directions, axis, ref, model, config,
                                          schedule["prior_receive_s"], schedule["terminal_receive_s"])
                if timing == "frame_header":
                    validate_against_archive(trial, originals[arm, "primary", method, "nominal"])
                if trial["estimate_ms"] is None:
                    raise ValueError(f"Full frozen replay failed: {arm}, {method}, {timing}")
                trial_rows.append(dict(arm=arm, method=method, timing=timing, **schedule_fields(schedule), **trial))
                stability = stability_summary(arm_stability, arm, method, timing, trial["estimate_ms"])
                method_result[timing] = {"ekf": trial, "leave_out": stability}
                if method in PROFILE_METHODS:
                    cost = pair_profile(ref, events, pairs, directions, axis, config, method)
                    values, diagnostics = fit_curve(cost)
                    method_result[timing]["profile"] = diagnostics
                    curve_rows += [dict(arm=arm, method=method, timing=timing, offset_ms=1000*x, mean_cost=y)
                                   for x, y in zip(GRID_S, values)]
            if method in PROFILE_METHODS:
                baseline = method_result["frame_header"]["profile"]
                mapped_best_s = baseline["best_ms"]/1000-constant
                control_cost = pair_profile(uniform, events, pairs, directions, axis, config, method)
                assert np.isclose(control_cost(mapped_best_s), baseline["min_mean_cost"], rtol=1e-9, atol=1e-9)
                method_result["uniform_control_profile_min_ms"] = 1000*mapped_best_s
            result["methods"][method] = method_result

    replay.write_csv(out / "replay_comparison.csv", trial_rows)
    replay.write_csv(out / "stability.csv", stability_rows)
    replay.write_csv(out / "cost_curves.csv", curve_rows)
    (out / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def schedule_fields(schedule):
    return {key: schedule[key] for key in ("n_pairs", "n_members", "prior_receive_s", "terminal_receive_s", "pair_schedule_sha256")}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.tracks, args.output)
