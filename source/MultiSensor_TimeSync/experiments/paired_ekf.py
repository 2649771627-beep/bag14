"""Evaluate opposite-direction EKF contrasts on frozen target observations from a ROS bag."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from src.ekf_estimator.filter import TemporalEKF
from src.ekf_estimator.paired import PairedTemporalEKF
from src.temporal_model.model import Observation, SENSORS
from src.temporal_model.vehicle import TargetReference, UnsupportedReference, VehicleObservationModel, load_geometry


METHODS = ('ekf_all', 'ekf_same_observations', 'paired_ekf')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


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


def run_filters(events, pairs, sensor, config, reference, model):
    specs = config['sensors']
    baseline = TemporalEKF(specs, config['filter'], reference, model)
    matched = TemporalEKF(specs, config['filter'], reference, model)
    paired = PairedTemporalEKF(specs, config['filter'], reference, model)
    members = {index for pair in pairs for index in pair}
    endpoints = {later: earlier for earlier, later in pairs}
    estimates = {method: {} for method in METHODS}
    for event in events:
        index = event.id
        try:
            all_result = baseline.step(event)
        except UnsupportedReference:
            all_result = None
        same_result = None
        if index in members:
            try:
                same_result = matched.step(event)
            except UnsupportedReference:
                pass
        if index in endpoints:
            try:
                result = paired.step_pair(events[endpoints[index]], event, specs[sensor]['geometry_sigma'])
            except UnsupportedReference:
                continue
            if all_result is not None and same_result is not None:
                for method, output in zip(METHODS, (all_result, same_result, result)):
                    estimates[method][index] = output['offset_s']
    return estimates


def rmse_ms(values):
    return float(1000 * np.sqrt(np.mean(np.square(values))))


def paired_metric(runs, function):
    ids = sorted(set.intersection(*(set(run[method]) for run in runs for method in METHODS)))
    metrics = {method: None for method in METHODS}
    if ids:
        for method in METHODS:
            metrics[method] = rmse_ms([function(index, *[run[method][index] for run in runs]) for index in ids])
    return {'common_endpoints': len(ids), 'rmse_ms': metrics}


def shift_headers(events, shift):
    return [replace(event, reported_s=event.reported_s + shift) for event in events]


def evaluate_real(events, pairs, sensor, config, reference, models):
    # Keep receive time, measured value, association and pair indices fixed in every rerun.
    delta = config['injection_ms'] * 1e-3
    replays = {sign: shift_headers(events, sign * delta) for sign in (-1, 1)}
    nominal = run_filters(events, pairs, sensor, config, reference, models['nominal'])
    results = {}
    for name, model in models.items():
        observed = nominal if name == 'nominal' else run_filters(events, pairs, sensor, config, reference, model)
        results[name] = {
            'geometry_sensitivity': paired_metric([observed, nominal], lambda i, a, b: a - b),
            'incremental_time_recovery': {
                str(sign * config['injection_ms']): paired_metric(
                    [run_filters(replay, pairs, sensor, config, reference, model), observed],
                    lambda i, a, b: a - b - sign * delta,
                ) for sign, replay in replays.items()
            },
        }
    return results


def evaluate_reference_replay(events, pairs, sensor, config, reference, models):
    # Only the C16 trajectory and acquisition schedule come from the bag in this experiment.
    cfg = config['reference_replay']
    start = events[0].receive_s
    truth = np.array([(cfg['offset_ms'] + cfg['drift_ms_per_s'] * (event.receive_s - start)) / 1000 for event in events])
    clean = [models['nominal'](event, 0.0, config['sensors'][sensor], reference)[0] for event in events]
    results = {name: [] for name in models}
    sigma = config['sensors'][sensor]['sigma']
    for seed in cfg['seeds']:
        rng = np.random.default_rng(seed)
        replay = [replace(event, reported_s=event.reported_s + truth[event.id], value=value + rng.normal(0.0, sigma))
                  for event, value in zip(events, clean)]
        for name, model in models.items():
            run = run_filters(replay, pairs, sensor, config, reference, model)
            results[name].append(paired_metric([run], lambda index, value: value - truth[index]))
    return results


def summary(lidar_radar, camera, calibration, config_path):
    tracks = read_json(lidar_radar)
    camera_data = read_json(camera) if camera else None
    if camera_data and int(camera_data['epoch_ns']) != int(tracks['epoch_ns']):
        raise ValueError('Camera and LiDAR/radar tracks use different bag epochs')
    config = read_json(config_path)
    geometry = load_geometry(calibration)
    reference = TargetReference(tracks['lidar'], config['reference_gap_s'], config['reference_smooth_half_window_s'])
    models = {'nominal': VehicleObservationModel(geometry)}
    for p in config['perturbations']:
        models[p['name']] = VehicleObservationModel(geometry, p['lateral_m'], p['yaw_deg'])
    result = {
        'config_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'tracks': str(lidar_radar), 'camera_tracks': str(camera) if camera else None,
        'epoch_ns': int(tracks['epoch_ns']), 'config': config, 'sensors': {},
    }
    for sensor in SENSORS:
        rows = tracks['radar'] if sensor == 'radar' else (camera_data.get('camera', []) if camera_data else [])
        events, positions, slopes = select_observations(rows, sensor, int(tracks['epoch_ns']), config, reference, models['nominal'])
        pairs = match_pairs(events, positions, slopes, sensor, config)
        info = {'extracted_rows': len(rows), 'available_events': len(events), 'pairs': len(pairs)}
        result['sensors'][sensor] = info
        if not pairs:
            continue
        info['pair_details'] = [
            {'earlier_reported_s': events[a].reported_s, 'later_reported_s': events[b].reported_s,
             'later_receive_s': events[b].receive_s,
             'gap_s': events[b].receive_s - events[a].receive_s,
             'distance_m': float(np.linalg.norm(positions[b] - positions[a]))}
            for a, b in pairs
        ]
        info['recorded_measurements'] = evaluate_real(events, pairs, sensor, config, reference, models)
        info['lidar_trajectory_simulation'] = evaluate_reference_replay(events, pairs, sensor, config, reference, models)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracks', type=Path, required=True)
    parser.add_argument('--camera', type=Path)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('config/paired_ekf.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summary(args.tracks, args.camera, args.calibration, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({sensor: {k: info[k] for k in ('extracted_rows', 'available_events', 'pairs')}
                      for sensor, info in result['sensors'].items()}, indent=2))


if __name__ == '__main__':
    main()
