"""Reproduce split, traversal removal, timestamp-origin and radar-association checks for R01."""

import argparse
from copy import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import numpy as np


DEFAULT_RECORDINGS = Path(__file__).resolve().parents[2] / "airy_ars408_vehicle_r01"
# Fixed engineering scales for 0.20 m radar coordinates and walking-speed motion.
TRACKER = {
    "longitudinal_range_m": [1.75, 6.35],
    "max_abs_lateral_m": 0.48,
    "rcs_step_db": 6.0,
    "rcs_penalty_cap": 3.0,
    "longitudinal_step_m": 0.20,
    "longitudinal_rate_mps": 1.50,
    "lateral_step_m": 0.20,
    "lateral_rate_mps": 0.50,
    "restart_penalty": 8.0,
    "max_continuity_gap_s": 0.50,
}


def radar_path(frames):
    """Find a low-cost path through the complete bag, using radar fields only."""
    result = {}
    emissions = []
    parents = []
    costs = None
    previous_xy = None
    last_receive = None
    low, high = TRACKER["longitudinal_range_m"]
    for frame in frames:
        candidates = [candidate for candidate in frame.candidates
                      if low <= candidate[0] <= high
                      and abs(candidate[1]) <= TRACKER["max_abs_lateral_m"]]
        if not candidates:
            continue
        coordinates = np.array([(c[0], c[1]) for c in candidates])
        db = np.array([c[3] for c in candidates])
        rcs_penalty = np.zeros(len(db))
        if np.any(np.isfinite(db)):
            rcs_penalty = np.minimum((db.max() - db) / TRACKER["rcs_step_db"],
                                     TRACKER["rcs_penalty_cap"])
        rcs_penalty += 0.25 * (coordinates[:, 1] / TRACKER["max_abs_lateral_m"]) ** 2
        dt = (frame.receipt - last_receive) * 1e-9 if last_receive is not None else 0.0
        if costs is None:
            costs = rcs_penalty
            back = np.full(len(candidates), -1, dtype=int)
        else:
            scales = np.array([TRACKER["longitudinal_step_m"] + TRACKER["longitudinal_rate_mps"] * dt,
                               TRACKER["lateral_step_m"] + TRACKER["lateral_rate_mps"] * dt])
            squared_step = np.sum(((previous_xy[:, None] - coordinates) / scales) ** 2, axis=2)
            reset = 0.0 if dt > TRACKER["max_continuity_gap_s"] else TRACKER["restart_penalty"]
            transitions = costs[:, None] + np.minimum(squared_step, reset)
            back = np.argmin(transitions, axis=0)
            costs = rcs_penalty + transitions[back, np.arange(len(candidates))]
            costs -= costs.min()
        emissions.append((frame, candidates))
        parents.append(back)
        previous_xy = coordinates
        last_receive = frame.receipt
    if costs is None:
        raise ValueError("Radar corridor has no candidates.")
    j = int(np.argmin(costs))
    for (frame, candidates), back in zip(reversed(emissions), reversed(parents)):
        result[frame.receipt] = (frame, candidates[j])
        j = int(back[j])
    return result


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def alternate_tracks(recordings, tracks, out, preparation, arms, read_radar):
    """Produce radar-only choices before the recorded LiDAR-dependent clipping."""
    out.mkdir(parents=True, exist_ok=True)
    origin = preparation["clock_mapping"]["origin_host_ns"]
    all_paths = {}
    coverage = {}
    for arm in arms:
        frames = read_radar(recordings / "raw" / f"{arm}_radar.csv")
        chosen = radar_path(frames)
        begin, end = preparation["clips"][arm]["radar_clip_host_s"]
        complete = [f for f in frames if begin <= (f.receipt - origin) / 1e9 <= end]
        used = {f.receipt: chosen[f.receipt] for f in complete if f.receipt in chosen}
        all_paths[arm] = used
        previous = {int(row["raw_receive_ns"]): row for row
                    in read_rows(tracks / f"{arm}_radar.csv")}
        shared = set(used).intersection(previous)
        n_same_object = sum(used[i][1][2] == previous[i]["object_id"] for i in shared)
        n_same_coordinates = sum(np.allclose(used[i][1][:2],
                                             [float(previous[i]["x_m"]), float(previous[i]["y_m"])],
                                             rtol=0.0, atol=1e-9) for i in shared)
        delta_x = [abs(used[i][1][0] - float(previous[i]["x_m"])) for i in shared]
        coverage[arm] = {
            "recording_native_frames": len(frames),
            "recording_candidate_rows": sum(len(f.candidates) for f in frames),
            "native_clip_frames": len(complete),
            "lidar_gated_frames": len(previous),
            "radar_only_frames": len(used),
            "shared_frames": len(shared),
            "same_object_id_on_shared": n_same_object,
            "same_xy_on_shared": n_same_coordinates,
            "median_abs_x_difference_m": float(np.median(delta_x)),
            "p95_abs_x_difference_m": float(np.quantile(delta_x, 0.95)),
            "first_receive_ns": min(used),
            "last_receive_ns": max(used),
            "largest_inter_association_gap_s": float(np.diff([begin, *((i-origin) / 1e9 for i in used), end]).max()),
        }
        shutil.copyfile(tracks / f"{arm}_lidar.csv", out / f"{arm}_lidar.csv")
        with (out / f"{arm}_radar.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("reported_s", "receive_s", "x_m", "y_m", "raw_stamp_ns",
                             "raw_receive_ns", "object_id", "rcs", "n_candidates"))
            for receipt, (frame, (x, y, object_id, rcs)) in used.items():
                writer.writerow(((frame.stamp-origin) / 1e9, (receipt-origin) / 1e9, x, y,
                                 frame.stamp, receipt, object_id, rcs, len(frame.candidates)))
    return all_paths, coverage


def selected_fits(problem, modes, blocks=False):
    fits = {}
    for mode in modes:
        b, grid, costs = problem.optimize(mode)
        assert b is not None and np.isfinite(costs).all()
        minimum, eta, residual, _, _, solver = problem.cost(b, mode, details=True)
        fit = {
            "candidate_offset_ms": float(b * 1000),
            "mean_huber_cost_per_frame": float(minimum / problem.mask.sum()),
            "fitted_nuisance": [float(x) for x in eta],
            "median_abs_whitened_residual": float(np.median(np.abs(residual))),
            "solver_converged": solver["converged"],
        }
        if blocks:
            receive = problem.r[problem.mask]
            bins = np.floor(receive - receive.min()).astype(int)
            splits = []
            for parity in (0, 1):
                mask = problem.mask.copy()
                mask[problem.mask] = (bins % 2) == parity
                split_b = problem.optimize(mode, mask)[0]
                assert split_b is not None
                splits.append({"frames": int(mask.sum()), "candidate_offset_ms": float(split_b * 1000)})
            fit["alternate_one_second_bins"] = splits
            fit["absolute_split_difference_ms"] = abs(splits[0]["candidate_offset_ms"]
                                                      - splits[1]["candidate_offset_ms"])
        fits[mode] = fit
    return fits


def summary(problem, fixed_information, modes, blocks=False):
    info = fixed_information(problem)
    return {
        "frames": int(problem.mask.sum()),
        "directions_negative_positive": [int(np.sum(problem.mask & (problem.direction == q)))
                                         for q in (-1, 1)],
        "axis_xy": problem.axis[:2].tolist(),
        "information_s_minus2": {m: info[m]["projected_information_s_minus2"] for m in modes},
        "fits": selected_fits(problem, modes, blocks),
    }


def leave_one_traversal(preparation, loaded, arms, matched_support, subset, geometry, information):
    rows = []
    pad = preparation["clip_edge_pad_s"]
    for removed_arm in arms:
        for i, trip in enumerate(preparation["clips"][removed_arm]["trips"], start=1):
            problem, *rest = loaded[removed_arm]
            retained = copy(problem)
            within = ((problem.r >= trip["near_departure_s"] - pad)
                      & (problem.r <= trip["near_return_s"] + pad))
            retained.mask = problem.mask & ~within
            held_out = {**loaded, removed_arm: (retained, *rest)}
            indices, _ = matched_support(held_out)
            row = {
                "removed_arm": removed_arm,
                "removed_cycle": i,
                "removed_eligible_observations": int(np.sum(within & problem.mask)),
                "matched_frames_per_arm": len(indices[removed_arm]),
            }
            for arm in arms:
                matched = subset(held_out[arm][0], geometry, indices[arm])
                row[arm] = summary(matched, information, ("geometry", "direction"))
            row["direction_information_ratio_variable_over_near"] = (
                row["variable"]["information_s_minus2"]["direction"]
                / row["near_constant"]["information_s_minus2"]["direction"])
            rows.append(row)
    return rows


def common_schedule(loaded, alternatives, arms, matched_support, subset, geometry, profile):
    """Change only radar target coordinates on a common frozen receive schedule."""
    reusable = {}
    for arm in arms:
        problem, positions, slopes, count = loaded[arm]
        available = set(round(event.receive_s * 1e9) for event in problem.events).intersection(alternatives[arm])
        intersected = copy(problem)
        intersected.mask = problem.mask & np.array([round(e.receive_s * 1e9) in available
                                                    for e in problem.events])
        reusable[arm] = (intersected, positions, slopes, count)
    indices, strata = matched_support(reusable)
    joint = {}
    for arm in arms:
        reference_problem = subset(reusable[arm][0], geometry, indices[arm])
        events = [replace(event, value=np.array(alternatives[arm][round(event.receive_s * 1e9)][:2]))
                  if reference_problem.mask[i] else event
                  for i, event in enumerate(reference_problem.events)]
        alternate_problem = profile.ProfileProblem(events, reference_problem.ref, geometry,
                                                   reference_problem.sensor, reference_problem.sigma,
                                                   support=reference_problem.support)
        assert np.array_equal(reference_problem.t, alternate_problem.t)
        assert np.array_equal(reference_problem.r, alternate_problem.r)
        assert np.array_equal(reference_problem.mask, alternate_problem.mask)
        assert np.array_equal(reference_problem.direction, alternate_problem.direction)
        joint[arm] = {"lidar_gated": reference_problem, "radar_only": alternate_problem}
    return joint, strata


def verify_anchor(problem, fits, delta_ms, modes, geometry, profile):
    """Check the translated objective on the same frames, direction labels and axis."""
    delta = delta_ms / 1000
    shifted_reference = copy(problem.ref)
    shifted_reference.times = problem.ref.times + delta
    shifted = profile.ProfileProblem(problem.events, shifted_reference, geometry, problem.sensor,
                                     problem.sigma, support=problem.support)
    low, high = profile.CONFIG["offset_search_s"]
    grid = np.linspace(low, high, 121)
    result = {}
    for mode in modes:
        p05_fit_ms = fits[mode]["candidate_offset_ms"]
        p05_fit = p05_fit_ms / 1000
        translated = np.r_[grid, p05_fit]
        differences = [abs(float(problem.cost(x, mode) - shifted.cost(x-delta, mode)))
                       for x in translated]
        max_error = max(differences)
        assert np.isfinite(max_error) and max_error <= 1e-6
        result[mode] = {
            "p05_candidate_ms": p05_fit_ms,
            "p50_equivalent_candidate_ms": p05_fit_ms - delta_ms,
            "maximum_absolute_huber_cost_difference": max_error,
        }
    return result


def render_plot(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    near, variable = "near_constant", "variable"
    markers = {"geometry": "o", "direction": "s"}
    colors = {near: "#165B78", variable: "#A6522D"}
    display = {near: "Near-constant", variable: "Variable"}
    values = report["leave_one_traversal"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), layout="constrained")
    a = axes[0]
    baseline_ratio = report["original"]["direction_information_ratio_variable_over_near"]
    a.axhline(baseline_ratio, color="#664E78", ls="--", lw=1.3, label="All traversals")
    x = np.arange(1, len(values) + 1)
    for arm in (near, variable):
        rows = [i for i, row in enumerate(values) if row["removed_arm"] == arm]
        a.scatter(x[rows], [values[i]["direction_information_ratio_variable_over_near"] for i in rows],
                  s=26, color=colors[arm], label=f"Remove {display[arm].lower()} traversal")
    a.set(xticks=x, ylabel="Direction-model information ratio\n(variable / near-constant)",
          xlabel="Omitted traversal")
    a.set_ylim(3.0, 5.8)
    a.legend(loc="upper right", fontsize=8, frameon=False)
    a.set_title("(a) Traversal omission", loc="left")
    a.grid(axis="y", alpha=0.23)
    b = axes[1]
    variants = report["association_common_schedule"]["arms"]
    for j, arm in enumerate((near, variable)):
        for k, mode in enumerate(markers):
            y = j * 2.6 + k * 1.0
            gated = variants[arm]["lidar_gated"]["fits"][mode]["candidate_offset_ms"]
            radar = variants[arm]["radar_only"]["fits"][mode]["candidate_offset_ms"]
            b.plot([gated, radar], [y, y], color=colors[arm], lw=1.1)
            b.scatter([gated], [y], marker=markers[mode], c=colors[arm], s=32,
                      label="LiDAR-gated" if j == k == 0 else None)
            b.scatter([radar], [y], marker=markers[mode], facecolors="white", edgecolors=colors[arm], s=32,
                      label="Radar-only ranking" if j == k == 0 else None)
    b.set(yticks=[0, 1, 2.6, 3.6],
          yticklabels=["Near / geometry", "Near / +direction",
                       "Variable / geometry", "Variable / +direction"],
          xlabel="Profile minimum under p05 anchor (ms)", ylim=(-0.55, 4.8))
    b.legend(loc="upper right", fontsize=8, frameon=False)
    b.set_title("(b) Association on the same frames", loc="left")
    b.grid(axis="x", alpha=0.23)
    fig.savefig(output, dpi=250)
    plt.close(fig)


def run(recordings, out):
    sys.path.insert(0, str(recordings.resolve()))
    from prepare import read_radar
    from profile_compare import (ARMS, CORE, GEOMETRY, MODES, fixed_information,
                                 load_arm, matched_support, profile, speed_stats, subset)

    recordings = recordings.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tracks = recordings / "tracks"
    preparation = json.loads((tracks / "preparation.json").read_text(encoding="utf-8"))
    cfg = json.loads((CORE / "inputs" / "filter_config.json").read_text(encoding="utf-8-sig"))
    loaded = {arm: load_arm(tracks, arm, cfg) for arm in ARMS}
    selection, _ = matched_support(loaded)
    matched = {arm: subset(loaded[arm][0], GEOMETRY, selection[arm]) for arm in ARMS}
    original = {arm: summary(matched[arm], fixed_information, MODES, blocks=True) for arm in ARMS}
    for arm in ARMS:
        original[arm]["radar_input_rows"] = loaded[arm][3]
        original[arm]["decimated_rows"] = len(loaded[arm][0].events)
        original[arm]["whole_search_support_rows"] = int(loaded[arm][0].mask.sum())
        original[arm]["speed_stats"] = speed_stats(loaded[arm][0], matched[arm], loaded[arm][2])
    original_ratio = (original["variable"]["information_s_minus2"]["direction"]
                      / original["near_constant"]["information_s_minus2"]["direction"])
    delta_ms = preparation["clock_mapping"]["pooled_median_anchor_difference_ms"]
    anchor = {arm: verify_anchor(matched[arm], original[arm]["fits"], delta_ms, MODES, GEOMETRY,
                                 profile) for arm in ARMS}
    alt_path = out / "alternate_tracks"
    alternatives, coverage = alternate_tracks(recordings, tracks, alt_path, preparation, ARMS, read_radar)
    origin = preparation["clock_mapping"]["origin_host_ns"]
    alternative_coordinates = {arm: {receipt-origin: c for receipt, (_, c) in choices.items()}
                               for arm, choices in alternatives.items()}
    joint, strata = common_schedule(loaded, alternative_coordinates, ARMS, matched_support,
                                    subset, GEOMETRY, profile)
    comparisons = {}
    alternate_loaded = {arm: load_arm(alt_path, arm, cfg) for arm in ARMS}
    alt_selection, _ = matched_support(alternate_loaded)
    alt_independent = {arm: summary(subset(alternate_loaded[arm][0], GEOMETRY, alt_selection[arm]),
                                    fixed_information, MODES) for arm in ARMS}
    for arm in ARMS:
        comparisons[arm] = {name: summary(problem, fixed_information, MODES)
                            for name, problem in joint[arm].items()}
        assert comparisons[arm]["lidar_gated"]["information_s_minus2"] == (
            comparisons[arm]["radar_only"]["information_s_minus2"])
        mask = joint[arm]["lidar_gated"].mask
        comparisons[arm]["receives_relative_ns"] = [round(x * 1e9) for x in joint[arm]["lidar_gated"].r[mask]]
        comparisons[arm]["same_xy_frames"] = int(np.sum(np.all(
            joint[arm]["lidar_gated"].z[mask] == joint[arm]["radar_only"].z[mask], axis=1)))
        alt_independent[arm]["radar_input_rows"] = alternate_loaded[arm][3]
        alt_independent[arm]["decimated_rows"] = len(alternate_loaded[arm][0].events)
        alt_independent[arm]["whole_search_support_rows"] = int(alternate_loaded[arm][0].mask.sum())
    removals = leave_one_traversal(preparation, loaded, ARMS, matched_support, subset,
                                   GEOMETRY, fixed_information)
    report = {
        "protocol": {
            "measurement": "Derived AIRY frames and ARS408 candidates from parked-vehicle R01.",
            "association": "Offline radar-only path ranking over each full bag. Afterwards use the same LiDAR-derived clip boundaries and fixed LiDAR reference for evaluation.",
            "tracker_settings": TRACKER,
            "common_schedule": "Intersect radar frame availability on the original decimated schedule, then repeat the original distance-and-direction balancing. Compare both candidate choices on identical receive and measurement epochs, axis, reference and direction labels.",
            "alternate_own_schedule": "Separately decimate and balance the radar-only path with the same rules to show potential changes of support.",
            "traversal_omission": "Remove one round-trip's eligible radar observations from one bag at a time, extend each recorded boundary by the original 0.65 s pad, and rematch. The two LiDAR references and axes remain fixed.",
            "clock_anchor": "Shift every mapped AIRY timestamp by p50 minus p05. Compare the same observations at candidate offsets shifted by the negative of that difference and translate the offset grid accordingly.",
            "scope": "Conditional profile candidates and standardized costs. Matched frames from two bags share trajectory samples.",
        },
        "original": {"arms": original, "direction_information_ratio_variable_over_near": original_ratio},
        "anchor": {"p50_minus_p05_lidar_time_ms": delta_ms, "arms": anchor},
        "association_coverage": coverage,
        "association_common_schedule": {"arms": comparisons, "strata": strata},
        "association_alternate_own_schedule": alt_independent,
        "leave_one_traversal": removals,
    }
    (out / "field_stability.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    render_plot(report, out / "field_stability.png")
    print(json.dumps({
        "original_matched_per_arm": [original[arm]["frames"] for arm in ARMS],
        "common_schedule_matched_per_arm": [comparisons[arm]["lidar_gated"]["frames"] for arm in ARMS],
        "coverage": coverage,
        "origin_translation_ms": delta_ms,
        "whole_trip_omissions": len(removals),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.recordings, args.out)
