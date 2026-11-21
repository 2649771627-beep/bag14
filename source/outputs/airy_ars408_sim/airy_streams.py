"""Generate synthetic AIRY–ARS408 target-position streams for 3–6 m round trips.

Only the nominal sensor rates are based on the four in-house 3/6 m holds.
Positions, jitter, missed detections, and clutter are synthetic. Radar column
four is a scoring label and is unavailable to the tracker or clock estimator.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


REPEATED_TRIALS = Path(__file__).resolve().parents[1] / "repeated_roundtrips_4_10m"
sys.path.insert(0, str(REPEATED_TRIALS))
from streams import ARMS, GATES, MOTIONS, get_prefix, motion_trace, tracker
sys.path.remove(str(REPEATED_TRIALS))


LIDAR_HZ = 10.0
RADAR_HZ = 13.65
ROUNDTRIPS = 8
LEG_SECONDS = 6.0
RANGE_MIN_M = 3.0
RANGE_MAX_M = 6.0
LIDAR_SIGMA_M = 0.025
RADAR_SIGMA_M = 0.07
DIRECTION_BIAS_M = 0.08
JITTER_S = 0.003
INDEPENDENT_MISS = 0.12
BURST_SECONDS = (0.35, 0.80)
COMPETITOR_SECONDS = (1.0, 2.2)
CLUTTER_COUNT_MEAN = 0.35


def airy_motion(times, motion):
    """Scale the original trajectory to travel 3 m per 6 s leg."""
    time_scale = 8.0 / LEG_SECONDS
    range_scale = (RANGE_MAX_M - RANGE_MIN_M) / 6.0
    position, velocity = motion_trace(times * time_scale, motion)
    return RANGE_MIN_M + range_scale * (position - 4.0), range_scale * time_scale * velocity


def simulate(seed, motion, *, direction_bias_m=DIRECTION_BIAS_M, radar_position_step_m=0.0):
    """Return LiDAR [t,x,y] and three radar arms [t,x,y,is_correct].

    The optional radar step rounds true detections' longitudinal coordinates
    before association. It leaves lateral coordinates, velocities, and clutter
    continuous, and draws no additional random numbers.
    """
    if motion not in MOTIONS:
        raise ValueError(f"Unknown motion {motion!r}")
    if not np.isfinite(radar_position_step_m) or radar_position_step_m < 0:
        raise ValueError("Radar position step must be finite and nonnegative")
    if not np.isfinite(direction_bias_m):
        raise ValueError("Direction bias must be finite")

    cycle_seconds = 2.0 * LEG_SECONDS
    segments = []
    prior_track = None
    next_nominal_time = None
    id_index = 0
    for cycle in range(ROUNDTRIPS):
        start = cycle_seconds * cycle
        rng = np.random.default_rng([20261119, 408, seed, cycle])

        lidar_t = np.linspace(0.0, cycle_seconds, round(cycle_seconds * LIDAR_HZ) + 1)
        lidar_t[1:-1] += rng.uniform(-JITTER_S, JITTER_S, len(lidar_t) - 2)
        if cycle:
            lidar_t = lidar_t[1:]
        lidar_target, _ = airy_motion(lidar_t, motion)
        lidar_xy = np.column_stack((lidar_target, np.zeros_like(lidar_target)))
        lidar_xy += rng.normal(0.0, LIDAR_SIGMA_M, lidar_xy.shape)

        first_time = (rng.uniform(0.01, 0.06) if next_nominal_time is None
                      else max(next_nominal_time - start, JITTER_S))
        nominal = np.arange(first_time, cycle_seconds - JITTER_S, 1.0 / RADAR_HZ)
        next_nominal_time = start + nominal[-1] + 1.0 / RADAR_HZ
        radar_t = nominal + rng.uniform(-JITTER_S, JITTER_S, len(nominal))

        x, speed = airy_motion(radar_t, motion)
        direction = np.where(radar_t < LEG_SECONDS, 1.0, -1.0)
        positions = np.column_stack((x + direction_bias_m * direction, np.zeros_like(x)))
        positions += rng.normal(0.0, RADAR_SIGMA_M, positions.shape)
        if radar_position_step_m:
            positions[:, 0] = np.rint(positions[:, 0] / radar_position_step_m) * radar_position_step_m
        velocities = np.column_stack((speed, np.zeros_like(speed)))
        velocities += rng.normal(0.0, (0.08, 0.04), velocities.shape)

        missing = rng.random(len(radar_t)) < INDEPENDENT_MISS
        competitor = np.zeros(len(radar_t), dtype=bool)
        competitor_offset = np.zeros(len(radar_t))
        for leg in (0, 1):
            leg_start = LEG_SECONDS * leg
            burst_start = leg_start + rng.uniform(0.6, LEG_SECONDS - 1.2)
            burst_end = burst_start + rng.uniform(*BURST_SECONDS)
            missing |= (radar_t >= burst_start) & (radar_t < burst_end)
            ghost_start = leg_start + rng.uniform(0.3, LEG_SECONDS - 2.6)
            ghost_end = ghost_start + rng.uniform(*COMPETITOR_SECONDS)
            active = (radar_t >= ghost_start) & (radar_t < ghost_end)
            competitor |= active
            competitor_offset[active] = rng.choice((-1, 1)) * rng.uniform(0.25, 0.65)

        if cycle == 0:
            ids = rng.permutation(256)
        id_changes = id_index + np.cumsum(rng.random(len(radar_t)) < 0.018)
        target_ids = ids[id_changes % 128]
        id_index = int(id_changes[-1])
        rows = {arm: [] for arm in ARMS}
        for index, local_time in enumerate(radar_t):
            stamp = float(start + local_time)
            target = (positions[index], velocities[index], SimpleNamespace(id=int(target_ids[index])))
            candidates = [] if missing[index] else [target]
            if competitor[index]:
                leg = int(local_time >= LEG_SECONDS)
                ghost_id = int(ids[180 + 2 * cycle + leg])
                ghost_position = np.array([x[index] + competitor_offset[index], 0.24])
                ghost_velocity = np.array([speed[index], 0.0])
                candidates.append((ghost_position + rng.normal(0.0, 0.06, 2),
                                   ghost_velocity + rng.normal(0.0, 0.08, 2),
                                   SimpleNamespace(id=ghost_id)))
            for _ in range(rng.poisson(CLUTTER_COUNT_MEAN)):
                candidates.append((rng.uniform([RANGE_MIN_M - 0.5, -1.5],
                                               [RANGE_MAX_M + 0.5, 1.5]),
                                   rng.normal(0.0, (0.32, 0.12)),
                                   SimpleNamespace(id=int(rng.choice(ids[200:])))))
            rng.shuffle(candidates)
            pick, _, prior_track, _ = tracker.association(candidates, stamp, prior_track, GATES)
            if pick is not None:
                pos, vel, obj = pick
                prior_track = dict(t=stamp, p=pos.copy(), v=vel.copy(), id=obj.id)
            observations = (target, None if missing[index] else target, pick)
            for arm, observation in zip(ARMS, observations):
                if observation is not None:
                    pos = observation[0]
                    rows[arm].append((stamp, pos[0], pos[1], observation is target))

        lidar = np.column_stack((lidar_t + start, lidar_xy))
        segments.append((lidar, {arm: np.asarray(values, dtype=float).reshape(-1, 4)
                                 for arm, values in rows.items()}))
    return get_prefix(segments, ROUNDTRIPS)
