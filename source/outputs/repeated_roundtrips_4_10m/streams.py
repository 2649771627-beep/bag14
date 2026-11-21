"""Continuous repeated 4–10 m round trips for the parked-vehicle timing experiment.

simulate returns eight segments of raw, causally associated detections. get_prefix
combines a requested number of segments, then applies the archived centered
temporal-support check on only the available history. All times are physical
seconds. LiDAR columns are t,x,y. Radar columns are t,x,y,is_correct, with the
last field reserved for offline scoring. It is never passed to the tracker.
"""
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

SINGLE_CYCLE = Path(__file__).resolve().parents[1] / "association_control_4_10m"
sys.path.insert(0, str(SINGLE_CYCLE))
from run_simulation import ARMS, GATES, MOTIONS, motion_trace, tracker


def simulate(seed, motion):
    if motion not in MOTIONS:
        raise ValueError(f"Unknown motion {motion}")
    segments = []
    state = None
    next_nominal = None
    ids = None
    id_index = 0
    for cycle in range(8):
        start = 16.0 * cycle
        rng_seed = 20261119 + seed if cycle == 0 else [20261119 + seed, cycle]
        rng = np.random.default_rng(rng_seed)
        # Cycle 1 exactly reproduces the earlier runs, including their two end knots.
        lidar_t = np.linspace(0, 16, 161)
        lidar_t[1:-1] += rng.uniform(-0.003, 0.003, len(lidar_t)-2)
        if cycle > 0:
            lidar_t = lidar_t[1:]
        if next_nominal is None:
            first = rng.uniform(0.01, 0.06)
            end = 15.99
        else:
            # Continue the radar clock. Keep a seam-adjacent timestamp in its own
            # physical cycle after bounded clock jitter (at most 13 ms adjustment).
            first = max(next_nominal - start, 0.003)
            end = 16.0 - 0.003
        nominal = np.arange(first, end, 1/13.6)
        next_nominal = start + nominal[-1] + 1/13.6
        radar_t = nominal + rng.uniform(-0.003, 0.003, len(nominal))
        lidar_x = motion_trace(lidar_t, motion)[0] + rng.normal(0, 0.025, len(lidar_t))
        lidar_y = rng.normal(0, 0.025, len(lidar_t))
        x, speed = motion_trace(radar_t, motion)
        n = len(x)
        position = np.column_stack((
            x + np.where(radar_t < 8, 0.08, -0.08) + rng.normal(0, 0.07, n),
            rng.normal(0, 0.07, n)))
        velocity = np.column_stack((speed + rng.normal(0, 0.08, n), rng.normal(0, 0.04, n)))
        missing = rng.random(n) < 0.12
        competitor = np.zeros(n, dtype=bool)
        delta_x = np.zeros(n)
        for leg in (0, 1):
            onset = 8*leg + rng.uniform(0.6, 6.8)
            missing |= (radar_t >= onset) & (radar_t < onset + rng.uniform(0.35, 0.8))
            ghost_start = 8*leg + rng.uniform(0.3, 5.4)
            active = (radar_t >= ghost_start) & (radar_t < ghost_start + rng.uniform(1.0, 2.2))
            competitor |= active
            delta_x[active] = rng.choice([-1, 1])*rng.uniform(0.25, 0.65)
        if ids is None:
            ids = rng.permutation(256)
        id_changes = id_index + np.cumsum(rng.random(n) < 0.018)
        target_ids = ids[id_changes % 128]
        id_index = int(id_changes[-1])
        rows = {arm: [] for arm in ARMS}
        for i, local_t in enumerate(radar_t):
            stamp = float(start + local_t)
            target = (position[i], velocity[i], SimpleNamespace(id=int(target_ids[i])))
            candidates = [] if missing[i] else [target]
            if competitor[i]:
                secondary_id = int(ids[180 + 2*cycle + int(local_t >= 8)])
                candidates.append((np.array([x[i]+delta_x[i], 0.24])+rng.normal(0, 0.06, 2),
                                   np.array([speed[i], 0.0])+rng.normal(0, 0.08, 2),
                                   SimpleNamespace(id=secondary_id)))
            for _ in range(rng.poisson(0.35)):
                candidates.append((rng.uniform([3.5,-1.5],[10.5,1.5]),
                                   rng.normal(0, [0.32,0.12]), SimpleNamespace(id=int(rng.choice(ids[200:])))))
            rng.shuffle(candidates)
            pick, _, state, _ = tracker.association(candidates, stamp, state, GATES)
            if pick is not None:
                pos, vel, obj = pick
                state = dict(t=stamp, p=pos.copy(), v=vel.copy(), id=obj.id)
            choices = (target, None if missing[i] else target, pick)
            for arm, observation in zip(ARMS, choices):
                if observation is not None:
                    p = observation[0]
                    rows[arm].append((stamp, p[0], p[1], observation is target))
        lidar = np.column_stack((lidar_t+start, lidar_x, lidar_y))
        segments.append((lidar, {a:np.asarray(r, dtype=float).reshape(-1,4) for a,r in rows.items()}))
    return segments


def get_prefix(segments, count):
    if not 1 <= count <= len(segments):
        raise ValueError("count must select a nonempty prefix of the recording")
    selected = segments[:count]
    lidar = np.concatenate([points for points, _ in selected])
    arms = {}
    for name in ARMS:
        scans = np.concatenate([records[name] for _, records in selected])
        rows = [dict(t=t, header_ns=int(round(t*1e9)), x=x, y=y, correct=ok, quality={})
                for t, x, y, ok in scans]
        confirmed = tracker.old._confirm(rows, GATES, Counter())
        arms[name] = np.asarray([(r["t"], r["x"], r["y"], r["correct"]) for r in confirmed],
                                dtype=float).reshape(-1, 4)
    return lidar, arms
