"""Rebuild the 4–10 m detection-level experiment with common random numbers."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "paper_innovation1_20261114/revision1"))
import extract_radar_motion_seeded as tracker

MOTIONS = ("near_constant", "variable")
ARMS = ("complete", "oracle_missing", "tracked")
OFFSETS_MS = (70, 160)
SEEDS = range(30)
GATES = tracker.old.ExtractionGates()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def motion_trace(times, motion):
    """Integrate a nonnegative speed on each 8 s leg, normalized to exactly 6 m."""
    knots = np.linspace(0, 8, 8001)
    ramp = np.minimum(np.minimum(knots, 8-knots) / 0.40, 1.0)
    envelope = (1 - np.cos(np.pi*ramp)) / 2
    position, velocity = np.empty(len(times)), np.empty(len(times))
    for leg in (0, 1):
        speed = envelope.copy()
        if motion == "variable":
            speed *= 1 + 0.70*np.sin(2*np.pi*knots/4 + 0.8*leg)
        distance = np.r_[0, np.cumsum((speed[1:]+speed[:-1])*0.0005)]
        scale = 6/distance[-1]
        use = (times < 8) if leg == 0 else (times >= 8)
        local = times[use] - 8*leg
        sign = 1 if leg == 0 else -1
        position[use] = (4 if leg == 0 else 10) + sign*scale*np.interp(local, knots, distance)
        velocity[use] = sign*scale*np.interp(local, knots, speed)
    return position, velocity


def generate_trial(seed, motion):
    rng = np.random.default_rng(20261119 + seed)
    lidar_t = np.linspace(0, 16, 161)
    lidar_t[1:-1] += rng.uniform(-0.003, 0.003, len(lidar_t)-2)
    radar_t = np.arange(rng.uniform(0.01, 0.06), 15.99, 1/13.6)
    radar_t += rng.uniform(-0.003, 0.003, len(radar_t))
    lidar_x = motion_trace(lidar_t, motion)[0] + rng.normal(0, 0.025, len(lidar_t))
    lidar_y = rng.normal(0, 0.025, len(lidar_t))
    x, speed = motion_trace(radar_t, motion)
    n = len(x)
    position = np.column_stack((x + np.where(radar_t < 8, 0.08, -0.08)
                                + rng.normal(0, 0.07, n), rng.normal(0, 0.07, n)))
    velocity = np.column_stack((speed + rng.normal(0, 0.08, n), rng.normal(0, 0.04, n)))
    missing = rng.random(n) < 0.12
    competitor = np.zeros(n, dtype=bool)
    delta_x = np.zeros(n)
    for leg in (0, 1):
        start = 8*leg + rng.uniform(0.6, 6.8)
        missing |= (radar_t >= start) & (radar_t < start + rng.uniform(0.35, 0.8))
        # One intermittent secondary return per leg, independent of the dropout draw.
        ghost_start = 8*leg + rng.uniform(0.3, 5.4)
        active = (radar_t >= ghost_start) & (radar_t < ghost_start + rng.uniform(1.0, 2.2))
        competitor |= active
        delta_x[active] = rng.choice([-1, 1])*rng.uniform(0.25, 0.65)
    # Labels are held separately for scoring. The tracker sees only detector ID, XY and velocity.
    ids = rng.permutation(256)
    id_changes = np.cumsum(rng.random(n) < 0.018)
    target_ids = ids[id_changes % 128]
    buff = {arm: [] for arm in ARMS}
    state = None
    decision_counts = Counter()
    for i, t in enumerate(radar_t):
        target = (position[i], velocity[i], SimpleNamespace(id=int(target_ids[i])))
        candidates = [] if missing[i] else [target]
        if competitor[i]:
            candidates.append((np.array([x[i]+delta_x[i], 0.24])+rng.normal(0, 0.06, 2),
                               np.array([speed[i], 0.0])+rng.normal(0, 0.08, 2),
                               SimpleNamespace(id=int(ids[180 + int(t >= 8)]))))
        for _ in range(rng.poisson(0.35)):
            candidates.append((rng.uniform([3.5,-1.5],[10.5,1.5]),
                               rng.normal(0, [0.32,0.12]), SimpleNamespace(id=int(rng.choice(ids[200:])))))
        rng.shuffle(candidates)
        pick, decision, prior, _ = tracker.association(candidates, float(t), state, GATES)
        decision_counts[decision] += 1
        state = prior
        if pick is not None:
            pos, vel, obj = pick
            state = dict(t=t, p=pos.copy(), v=vel.copy(), id=obj.id)
        choices = (target, None if missing[i] else target, pick)
        for arm, selection in zip(ARMS, choices):
            if selection is not None:
                p = selection[0]
                buff[arm].append(dict(t=float(t), header_ns=int(round(t*1e9)),
                                      x=float(p[0]), y=float(p[1]),
                                      correct=(selection is target), quality={}))
    suffix = f"{motion}_{seed}"
    arrays = {f"lidar_t_{suffix}":lidar_t, f"lidar_x_{suffix}":lidar_x,
              f"lidar_y_{suffix}":lidar_y}
    diagnostics = []
    for arm, rows in buff.items():
        supported = tracker.old._confirm(rows, GATES, Counter())
        array = np.array([[r["t"], r["x"], r["y"]] for r in supported]).reshape(-1,3)
        arrays[f"radar_{suffix}_{arm}"] = array
        diagnostics.append(dict(seed=seed, motion=motion, observation=arm, radar_frames=n,
            n_true_detected=int((~missing).sum()), n_chosen=len(rows), n_supported=len(supported),
            n_wrong_supported=sum(not r["correct"] for r in supported)))
    assert np.array_equal(arrays[f"radar_{suffix}_complete"][:,0], radar_t[1:-1])
    return arrays, diagnostics


def generate():
    streams, diagnostics = {}, []
    for seed in SEEDS:
        for motion in MOTIONS:
            arrays, counts = generate_trial(seed, motion)
            streams.update(arrays)
            diagnostics.extend(counts)
    np.savez_compressed(HERE/"streams.npz", **streams)
    write_csv(HERE/"association_counts.csv", diagnostics)


def estimate():
    from fit import fit
    results = []
    with np.load(HERE/"streams.npz") as streams:
        for seed in SEEDS:
            for motion in MOTIONS:
                tag = f"{motion}_{seed}"
                lt, lx = streams[f"lidar_t_{tag}"], streams[f"lidar_x_{tag}"]
                for arm in ARMS:
                    t, x, _ = streams[f"radar_{tag}_{arm}"].T
                    for truth in OFFSETS_MS:
                        result = fit(lt, lx, t + truth/1000, x, t + 0.20)
                        value = result["estimate_ms"]
                        error = None if value is None else float(value) - truth
                        results.append(dict(seed=seed, motion=motion, observation=arm, truth_ms=truth,
                            **result, error_ms=error, abs_error_ms=abs(error) if error is not None else None))
    write_csv(HERE/"trials.csv", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "estimate", "all"], default="all", nargs="?")
    args = parser.parse_args()
    if args.stage in ("all","generate"):
        generate()
    if args.stage in ("all","estimate"):
        estimate()
