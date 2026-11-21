"""Compare constant-offset profiles on direction- and position-balanced R01 support.

Both inputs use the common host-relative time convention prepared for replay.
Offsets and information here are conditional model diagnostics on recorded data.
"""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
CORE = OUTPUTS / "mst_review_20261117/reviewer_replay/bundle/core"
sys.path[:0] = [str(OUTPUTS.parent / "MultiSensor_TimeSync"), str(CORE)]

import profile_estimator as profile
from experiments.paired_ekf import select_observations
from speed_groups import fixed_information, subset
from src.temporal_model.vehicle import TargetReference, VehicleObservationModel


ARMS = ("near_constant", "variable")
MODES = ("geometry", "direction")
POSITION_EDGES = np.linspace(2.65, 5.45, 8)
GEOMETRY = (np.eye(4), np.eye(4), np.eye(3), np.zeros(4))


def load_arm(directory, arm, config):
    with (directory / f"{arm}_lidar.csv").open(encoding="utf-8-sig", newline="") as handle:
        track = [
            dict(t=float(row["t_s"]), x=float(row["x_m"]), y=float(row["y_m"]), z=0.0)
            for row in csv.DictReader(handle)
        ]
    with (directory / f"{arm}_radar.csv").open(encoding="utf-8-sig", newline="") as handle:
        radar = [
            dict(t=float(row["reported_s"]),
                 bag_record_ns=round(float(row["receive_s"]) * 1e9),
                 x=float(row["x_m"]), y=float(row["y_m"]))
            for row in csv.DictReader(handle)
        ]
    reference = TargetReference(track, config["reference_gap_s"],
                                config["reference_smooth_half_window_s"])
    model = VehicleObservationModel(GEOMETRY)
    events, _, _ = select_observations(radar, "radar", 0, config, reference, model)
    problem = profile.ProfileProblem(events, reference, GEOMETRY, "radar",
                                     config["sensors"]["radar"]["sigma"])
    positions, velocities, valid = profile.reference_many(reference, problem.t)
    assert np.all(~problem.mask | valid)
    return problem, positions[:, 0], velocities @ problem.axis, len(radar)


def matched_support(loaded):
    """Keep equal counts in each position/direction bin without using radar XY."""
    problems = {label: value[0] for label, value in loaded.items()}
    selected = {label: [] for label in loaded}
    strata = []
    for direction in (-1, 1):
        for low, high in zip(POSITION_EDGES[:-1], POSITION_EDGES[1:]):
            candidates = {
                label: np.flatnonzero(
                    problems[label].mask & (problems[label].direction == direction)
                    & (value[1] >= low) & (value[1] < high)
                ) for label, value in loaded.items()
            }
            count = min(map(len, candidates.values()))
            chosen = {}
            for label, eligible in candidates.items():
                # Equal-spaced longitudinal ranks cover the sub-bin on both arms.
                order = np.argsort(loaded[label][1][eligible], kind="stable")
                indices = eligible[order[np.linspace(0, len(eligible) - 1, count).astype(int)]]
                assert len(indices) == len(np.unique(indices)) == count
                selected[label].extend(indices)
                chosen[label] = loaded[label][1][indices]
            mismatch = np.abs(chosen[ARMS[0]] - chosen[ARMS[1]])
            strata.append({
                "direction": direction, "vehicle_x_bin_m": [float(low), float(high)],
                "available": {label: len(eligible) for label, eligible in candidates.items()},
                "selected_in_each": count,
                "median_rankwise_abs_x_difference_m": float(np.median(mismatch)) if count else None,
            })
    if not selected[ARMS[0]]:
        raise ValueError("The arms share no direction/longitudinal positions on whole-search support.")
    assert len(selected[ARMS[0]]) == len(selected[ARMS[1]])
    return {label: np.array(sorted(indices), dtype=int) for label, indices in selected.items()}, strata


def speed_stats(full, matched, axis_velocity):
    """Measure within-leg spread using consecutive signs of the full eligible observations."""
    interior = np.flatnonzero(
        full.mask & (np.abs(axis_velocity) > 0.0)
        & np.isin(full.direction, [-1., 1.])
    )
    transition = np.flatnonzero(np.diff(full.direction[interior]) != 0.0) + 1
    legs = np.split(interior, transition)
    rates = np.abs(axis_velocity[matched.mask])
    directions = matched.direction[matched.mask]
    records = []
    for direction in (-1, 1):
        velocity = rates[directions == direction]
        spreads = []
        for leg in legs:
            if len(leg) == 0 or full.direction[leg[0]] != direction:
                continue
            retained = np.abs(axis_velocity[leg[matched.mask[leg]]])
            if len(retained) >= 8:
                spreads.append(float(np.diff(np.quantile(retained, [.1, .9]))[0]))
        records.append({
            "sign": direction, "sample_count": len(velocity),
            "mean_axis_speed_mps": float(velocity.mean()) if len(velocity) else None,
            "pooled_axis_speed_q10_q90_mps": np.quantile(velocity, [.1, .9]).tolist()
            if len(velocity) else [],
            "legs_with_eight_matched_rows": len(spreads),
            "median_within_leg_q90_minus_q10_mps": float(np.median(spreads))
            if spreads else None,
        })
    return records


def evaluate(problem):
    info = fixed_information(problem)
    fits = {}
    curves = {}
    for mode in MODES:
        information = info[mode]["projected_information_s_minus2"]
        info[mode]["local_linearized_scale_ms"] = float(1000.0 / np.sqrt(information)) if information else None
        offset, grid, costs = problem.optimize(mode)
        assert offset is not None and np.isfinite(costs).all()
        cost, nuisance, _, _, _, inner = problem.cost(offset, mode, details=True)
        fits[mode] = {
            "candidate_offset_ms": float(offset * 1000),
            "mean_huber_cost_per_radar_frame": float(cost / problem.mask.sum()),
            "fitted_translation_xy_m": nuisance[:2].tolist(),
            "fitted_yaw_deg": float(np.degrees(nuisance[2])),
            "fitted_direction_center_m": float(nuisance[3]) if mode == "direction" else None,
            "inner_solver": inner,
        }
        curves[mode] = (grid, costs, cost)
    info["fraction_remaining_after_direction_vs_geometry"] = (
        info["direction"]["projected_information_s_minus2"]
        / info["geometry"]["projected_information_s_minus2"]
        if info["geometry"]["projected_information_s_minus2"] else None
    )
    return info, fits, curves


def render(curves, count, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
    for axis, mode in zip(axes, MODES):
        for arm, label, style in zip(ARMS, ("Near-constant", "Variable"),
                                      ({"color": "#005B85"}, {"color": "#BF541E", "linestyle": "--"})):
            offsets, costs, minimum = curves[arm][mode]
            axis.plot(offsets * 1000, (costs - minimum) / count, label=label, **style)
        axis.set_title("Shared translation and yaw" if mode == "geometry" else "With direction-dependent center")
        axis.set_xlabel("Offset under the replay clock convention (ms)")
        axis.grid(alpha=.24)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Excess Huber loss per radar frame")
    fig.suptitle(f"AIRY–ARS408 · R01 matched support · {count} frames per arm")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(tracks, out):
    config = json.loads((CORE / "inputs/filter_config.json").read_text(encoding="utf-8-sig"))
    loaded = {label: load_arm(tracks, label, config) for label in ARMS}
    selections, strata = matched_support(loaded)
    curves = {}
    result = {
        "protocol": {
            "matching": "Equal counts by whole-search direction and nominal vehicle longitudinal position bin, spaced along longitudinal rank within each bin. No ranking by residual or radar position.",
            "position_bin_edges_m": POSITION_EDGES.tolist(),
            "offset_search_ms": [1000 * x for x in profile.CONFIG["offset_search_s"]],
            "coarse_grid_step_ms": 1000 * profile.CONFIG["coarse_grid_step_s"],
            "fixed_direction_rule": "Existing per-arm reference axis, stable direction on the full grid.",
            "reference_gap_s": config["reference_gap_s"],
            "reference_smooth_half_window_s": config["reference_smooth_half_window_s"],
            "radar_receive_decimation_s": config["sensors"]["radar"]["sample_period_s"],
            "nominal_radar_noise_m": config["sensors"]["radar"]["sigma"],
            "local_information": "Fixed offset b=0, zero nuisance linearization, declared XY sigma whitening, nuisance-subspace projection. Scale 1000/sqrt(I) reflects local sensitivity and assumes independent residual coordinates.",
            "interpretation": "Two recorded arms on one vehicle, with position and direction balanced but clustered in traversals. The offset convention includes unknown fixed timestamp delays. Candidate values and local information do not measure physical clock error.",
        },
        "source_sha256": {f"{arm}_{sensor}.csv": sha256((tracks / f"{arm}_{sensor}.csv").read_bytes()).hexdigest()
                          for arm in ARMS for sensor in ("lidar", "radar")},
        "strata": strata,
        "arms": {},
    }
    for arm, (problem, _, velocity, radar_rows) in loaded.items():
        indices = selections[arm]
        matched = subset(problem, GEOMETRY, indices)
        info, fits, curves[arm] = evaluate(matched)
        result["arms"][arm] = {
            "radar_input_rows": radar_rows, "decimated_rows": len(problem.events),
            "whole_search_support_rows": problem.support["retained_events"],
            "matched_rows": len(indices), "frozen_axis_xy": problem.axis[:2].tolist(),
            "retained_event_ids": matched.ids[matched.mask].tolist(),
            "speed_stats": speed_stats(problem, matched, velocity),
            "local_diagnostics_b0": info, "profile_fits": fits,
        }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "profile_curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "model", "offset_ms", "huber_cost", "mean_excess_cost"))
        for arm, modes in curves.items():
            for mode, (grid, costs, minimum) in modes.items():
                writer.writerows((arm, mode, 1000 * b, cost, (cost - minimum) / len(selections[arm]))
                                 for b, cost in zip(grid, costs))
    profile.save(out / "profile_compare.json", result)
    render(curves, len(selections[ARMS[0]]), out / "profile_compare.png")
    for arm, values in result["arms"].items():
        print(arm, values["matched_rows"], values["local_diagnostics_b0"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, default=HERE / "tracks")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    main(args.tracks, args.out)
