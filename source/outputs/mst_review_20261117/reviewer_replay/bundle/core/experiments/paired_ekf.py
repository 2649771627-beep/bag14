"""Evaluate opposite-direction EKF contrasts on frozen target observations from a ROS bag."""

import numpy as np

from src.temporal_model.model import Observation
from src.temporal_model.vehicle import UnsupportedReference


def select_observations(rows, sensor, epoch_ns, config, reference, model):
    spec = config['sensors'][sensor]
    events, positions, slopes = [], [], []
    last_time = -np.inf
    for row in sorted(rows, key=lambda r: r['bag_record_ns']):
        if 'min_detection_score' in spec and row['quality']['score'] < spec['min_detection_score']:
            continue
        t = float(row['t'])
        receive = (row['bag_record_ns'] - epoch_ns) * 1e-9
        if receive - last_time < spec['sample_period_s']:
            continue
        value = [row['x'], row['y']] if sensor == 'radar' else [row['u']]
        if not np.all(np.isfinite([t, receive, *value])):
            continue
        event = Observation(len(events), sensor, t, receive, np.array(value))
        try:
            position, _ = reference(t)
            margin = config['candidate_time_margin_s']
            reference(t - margin)
            reference(t + margin)
            _, slope = model(event, 0.0, spec, reference)
        except UnsupportedReference:
            continue
        events.append(event)
        positions.append(position[:2])
        slopes.append(slope)
        last_time = receive
    return events, np.array(positions), np.array(slopes)


def match_pairs(events, positions, slopes, sensor, config):
    if not events:
        return []
    times = np.array([event.receive_s for event in events])
    speed = np.linalg.norm(slopes, axis=1)
    minimum = config['sensors'][sensor]['min_slope']
    used = np.zeros(len(events), dtype=bool)
    pairs = []
    for later in range(1, len(events)):
        if speed[later] < minimum:
            continue
        gap = times[later] - times[:later]
        eligible = np.flatnonzero((~used[:later]) & (gap >= config['pair_min_gap_s'])
                                 & (gap <= config['pair_max_gap_s']) & (speed[:later] >= minimum))
        if not len(eligible):
            continue
        distances = np.linalg.norm(positions[eligible] - positions[later], axis=1)
        cosine = (slopes[eligible] @ slopes[later]) / (speed[eligible] * speed[later])
        ratios = np.minimum(speed[eligible], speed[later]) / np.maximum(speed[eligible], speed[later])
        keep = ((distances <= config['pair_max_distance_m']) & (cosine <= config['pair_max_cosine'])
                & (ratios >= config['pair_min_speed_ratio']))
        if np.any(keep):
            choices = eligible[keep]
            score = distances[keep] + 0.005 * gap[choices]
            earlier = int(choices[np.argmin(score)])
            pairs.append((earlier, later))
            used[[earlier, later]] = True
    return pairs


