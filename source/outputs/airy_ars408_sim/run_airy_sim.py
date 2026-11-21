"""Compare matched EKFs with known offsets on synthetic AIRY–ARS408 observations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from airy_streams import simulate

HERE = Path(__file__).resolve().parent
for folder in ("association_control_4_10m", "direction_center_ekf_4_10m",
               "direction_center_bag_08_09"):
    sys.path.insert(0, str(HERE.parent / folder))

from paired_check import (FILTER, GEOMETRY, VehicleObservationModel, make_reference,
                          match_pairs, select_observations, simulate_clock)
from run import DirectionCenterPairedEKF
from center_filter import reference_axis
from observability import paired_information
from sim_validate import filter_output


SEEDS = range(90, 120)
MOTIONS = ("near_constant", "variable")
TRUTHS = (70, 160)
METHODS = ("raw_paired", "axis_paired", "robust_axis_paired", "robust_axis_framewise")


def condition_rows(seed, motion, arm, step, truth, lidar, radar, reference, axis, model):
    samples = simulate_clock(radar[:, :3], truth)
    events, positions, slopes = select_observations(samples, "radar", 0, FILTER, reference, model)
    pairs = match_pairs(events, positions, slopes, "radar", FILTER)
    members = sorted(index for pair in pairs for index in pair)
    assert len(set(members)) == 2 * len(pairs)
    starts_ends = [(events[i].receive_s, events[j].receive_s) for i, j in pairs]
    prior = events[members[0]].receive_s if members else None
    terminal = events[members[-1]].receive_s if members else None
    assert not pairs or starts_ends[-1][1] == terminal
    schedule = dict(prior_receive_s=prior, terminal_receive_s=terminal)
    digest = hashlib.sha256(np.asarray(starts_ends, dtype="<f8").tobytes()).hexdigest()
    pairs_info = paired_information(events, pairs, slopes, FILTER)
    correct_by_receive = {row["bag_record_ns"]: bool(ok) for row, ok in zip(samples, radar[:, 3])}
    wrong_members = sum(not correct_by_receive[round(events[i].receive_s * 1e9)] for i in members)
    speed_sums = [float(abs(slopes[i, 0]) + abs(slopes[j, 0])) for i, j in pairs]
    common = dict(seed=seed, motion=motion, observation=arm, radar_step_m=step, truth_ms=truth,
                  lidar_frames=len(lidar), radar_supported=len(radar),
                  wrong_supported=int((radar[:, 3] == 0).sum()), wrong_members=wrong_members,
                  n_selected=len(events), n_pairs=len(pairs), n_members=len(members),
                  pair_speed_sum_std_mps=float(np.std(speed_sums)) if speed_sums else None,
                  pair_schedule_sha256=digest, **schedule, **pairs_info)
    records = []
    for name in METHODS:
        value = (filter_output(name, events, pairs, slopes, schedule, reference, model, axis)
                 if pairs else dict(estimate_ms=None, center_m=None, failed_updates=0,
                                    n_downweighted=0, n_weights=0, min_weight=None))
        estimate = value["estimate_ms"]
        error = None if estimate is None else estimate - truth
        records.append(dict(**common, method=name, **value, signed_error_ms=error,
                            abs_error_ms=None if error is None else abs(error)))
    return records


def run(task):
    seed, motion = task
    lidar, nominal = simulate(seed, motion)
    quantized_lidar, quantized = simulate(seed, motion, radar_position_step_m=0.20)
    np.testing.assert_array_equal(lidar, quantized_lidar)
    assert nominal["tracked"].shape[1] == quantized["tracked"].shape[1] == 4
    reference = make_reference(*lidar.T)
    axis, _ = reference_axis(reference)
    model = VehicleObservationModel(GEOMETRY)
    rows = []
    for step, arms in ((0.0, nominal), (0.20, {"tracked": quantized["tracked"]})):
        for arm, radar in arms.items():
            for truth in TRUTHS:
                rows.extend(condition_rows(seed, motion, arm, step, truth, lidar, radar,
                                           reference, axis, model))
    assert len(rows) == 32
    return rows


def write_trials(tasks, workers):
    temporary = HERE / "trials.partial.csv"
    with ProcessPoolExecutor(max_workers=workers) as pool, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = None
        for completed, rows in enumerate(pool.map(run, tasks), 1):
            if writer is None:
                writer = csv.DictWriter(destination, fieldnames=rows[0])
                writer.writeheader()
            writer.writerows(rows)
            destination.flush()
            if completed % 10 == 0 or completed == len(tasks):
                print(f"{completed}/{len(tasks)} seed-motion streams evaluated", flush=True)
    temporary.replace(HERE / "trials.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        for motion in MOTIONS:
            rows = run((SEEDS.start, motion))
            observed = [r for r in rows if r["observation"] == "tracked" and r["radar_step_m"] == 0]
            print(motion, "tracked", [(r["truth_ms"], r["method"], r["n_pairs"], r["abs_error_ms"])
                                      for r in observed])
        return
    write_trials([(seed, motion) for motion in MOTIONS for seed in SEEDS], args.workers)


if __name__ == "__main__":
    main()
