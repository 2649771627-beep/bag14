"""Frozen-pair replay using the recorded radar features after extraction."""

from dataclasses import replace

import numpy as np

import profile_estimator as profile
from src.ekf_estimator.filter import TemporalEKF
from src.ekf_estimator.paired import PairedTemporalEKF
from src.temporal_model.vehicle import UnsupportedReference, VehicleObservationModel


SHIFTS = {'nominal': (0., False), 'common_plus': (.08, False), 'common_minus': (-.08, False),
          'direction_plus': (.08, True), 'direction_minus': (-.08, True)}
INJECTIONS_MS = (0, -30, 30)
METHODS = ('raw_stream', 'difference', 'raw_completed_pairs')
SENSOR = 'radar'


def freeze_events(events, pairs, ref, selected_rows, identity):
    axis = np.asarray(identity['axis'])
    nominal_times = np.array([e.reported_s for e in events])
    _, velocities, support = profile.reference_many(ref, nominal_times)
    projected_velocity = velocities @ axis
    directions = np.sign(projected_velocity).astype(int)
    assert support.all() and directions.tolist() == identity['direction']
    assert all(directions[a] * directions[b] == -1 for a, b in pairs)
    members = sorted(i for pair in pairs for i in pair)
    assert len(set(members)) == 2 * len(pairs)
    endpoints = [late for _, late in pairs]
    assert sorted(endpoints) == endpoints
    speed_sums = np.array([abs(projected_velocity[a]) + abs(projected_velocity[b]) for a, b in pairs])
    frozen = {
        'axis_lidar_xy': axis[:2].tolist(),
        'prior_receive_s': events[members[0]].receive_s,
        'median_pair_axis_speed_sum_mps': float(np.median(speed_sums)),
        'median_directional_equivalent_offset_ms': float(np.median(160. / speed_sums)),
        'endpoint_receive_range_s': [events[endpoints[0]].receive_s, events[endpoints[-1]].receive_s],
        'pairs': [{'early_id': a, 'late_id': b} for a, b in pairs],
        'members': [{'event_id': i, 'row_id': selected_rows[i], 'reported_s': events[i].reported_s,
                     'receive_s': events[i].receive_s, 'nominal_lidar_xy_m': events[i].value.tolist(),
                     'direction': int(directions[i]), 'axis_velocity_mps': float(projected_velocity[i])}
                    for i in members],
        'stream_extra_early_frames_by_endpoint': [sum(i <= late for i in members) - 2 * n
                                                  for n, late in enumerate(endpoints, 1)],
    }
    return frozen, {i: events[i] for i in members}


def perturb(events, pairs, frozen, amplitude, directional, injection_ms):
    axis = np.asarray(frozen['axis_lidar_xy'])
    directions = {row['event_id']: row['direction'] for row in frozen['members']}
    modified = {i: replace(event, reported_s=event.reported_s + injection_ms / 1000,
                           value=event.value + amplitude * (directions[i] if directional else 1) * axis)
                for i, event in events.items()}
    for early, late in pairs:
        added_contrast = ((modified[late].value - modified[early].value)
                          - (events[late].value - events[early].value))
        expected = amplitude * (directions[late] - directions[early]) * axis if directional else np.zeros(2)
        np.testing.assert_allclose(added_contrast, expected, rtol=0., atol=1e-14)
    return modified


def replay(events, pairs, cfg, reference, model, prior_epoch):
    def start(filter_class):
        estimator = filter_class(cfg['sensors'], cfg['filter'], reference, model)
        estimator.predict_to(prior_epoch)
        return estimator

    def step(estimator, event, early=None):
        try:
            result = (estimator.step(event) if early is None else
                      estimator.step_pair(early, event, cfg['sensors'][SENSOR]['geometry_sigma']))
            return {**result, 'reference_failure': None}
        except UnsupportedReference as exc:
            return {**estimator.estimate(SENSOR, estimator.x[0]), 'reference_failure': str(exc)}

    endpoints = {late: early for early, late in pairs}
    raw_filter = start(TemporalEKF)
    raw = {i: step(raw_filter, e) for i, e in events.items()}
    difference_filter = start(PairedTemporalEKF)
    difference = {late: step(difference_filter, events[late], events[early]) for early, late in pairs}
    completed_raw = {}
    completed_members = []
    for early, late in pairs:
        completed_members.extend((early, late))
        estimator = start(TemporalEKF)
        prefix_results = [step(estimator, events[i]) for i in sorted(completed_members)]
        completed_raw[late] = {**prefix_results[-1], 'prefix_reference_failures': sum(
            row['reference_failure'] is not None for row in prefix_results)}
    return {'outputs': {'raw_stream': {i: raw[i] for i in endpoints}, 'difference': difference,
                        'raw_completed_pairs': completed_raw},
            'raw_member_updates': raw,
            'reference_failures': {'raw_stream': [i for i, row in raw.items() if row['reference_failure']],
                                   'difference': [i for i, row in difference.items() if row['reference_failure']],
                                   'raw_completed_pairs': [i for i, row in completed_raw.items()
                                                           if row['prefix_reference_failures']]}}


def stats(values, endpoint_ids):
    values = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(values)) and len(values) == len(endpoint_ids) > 0
    return {'n': len(values), 'signed_mean_ms': float(values.mean()),
            'rmse_ms': float(np.sqrt(np.mean(values ** 2))), 'mean_abs_ms': float(np.abs(values).mean()),
            'max_abs_ms': float(np.max(np.abs(values))), 'final_ms': float(values[-1]),
            'final_endpoint_id': int(endpoint_ids[-1])}


def endpoint_delta(current, baseline, method, ids, subtract_ms=0.):
    return [1000 * (current[method][i]['offset_s'] - baseline[method][i]['offset_s']) - subtract_ms for i in ids]


def evaluate(events, pairs, ref, selected_rows, cfg, geometry, identity):
    frozen, events = freeze_events(events, pairs, ref, selected_rows, identity)
    count = len(pairs)
    model = VehicleObservationModel(geometry)
    runs = {case: {ms: replay(perturb(events, pairs, frozen, amp, direction, ms), pairs, cfg, ref,
                            model, frozen['prior_receive_s']) for ms in INJECTIONS_MS}
            for case, (amp, direction) in SHIFTS.items()}
    baseline = runs['nominal'][0]['outputs']
    endpoints = [late for _, late in pairs]
    # Keep every attempted result in the report. The fixed-denominator analysis is valid only if all updates succeed.
    failures = {case: {ms: result['reference_failures'] for ms, result in replays.items()}
                for case, replays in runs.items()}
    members = list(events)
    np.testing.assert_allclose([identity['raw_nominal_offsets_s'][str(i)] for i in members],
                               [runs['nominal'][0]['raw_member_updates'][i]['offset_s'] for i in members],
                               rtol=0., atol=1e-8)
    for replays in runs.values():
        for replay_result in replays.values():
            output = replay_result['outputs']
            assert output['raw_stream'][endpoints[-1]]['offset_s'] == output['raw_completed_pairs'][endpoints[-1]]['offset_s']
            assert all(list(output[method]) == endpoints for method in METHODS)
    for common in ('common_plus', 'common_minus'):
        for ms in INJECTIONS_MS:
            np.testing.assert_allclose(endpoint_delta(runs[common][ms]['outputs'], runs['nominal'][ms]['outputs'],
                                                       'difference', endpoints), 0., rtol=0., atol=1e-9)
    summary = {}
    for case, replays in runs.items():
        base = replays[0]['outputs']
        summary[case] = {}
        for method in METHODS:
            changes = endpoint_delta(base, baseline, method, endpoints)
            recovery = {ms: endpoint_delta(replays[ms]['outputs'], base, method, endpoints, ms)
                        for ms in INJECTIONS_MS if ms}
            pooled_errors = [v for errors in recovery.values() for v in errors]
            summary[case][method] = {'displacement_change': stats(changes, endpoints),
                                     'recovery_error': {ms: stats(errors, endpoints) for ms, errors in recovery.items()},
                                     'recovery_pooled_rmse_ms': float(np.sqrt(np.mean(np.square(pooled_errors))))}
    pooled_shift = {kind: {method: float(np.sqrt(np.mean([
        summary[f'{kind}_{sign}'][method]['displacement_change']['rmse_ms'] ** 2 for sign in ('plus', 'minus')
    ]))) for method in METHODS} for kind in ('common', 'direction')}
    success = not any(ids for case in failures.values() for ms in case.values() for ids in ms.values())
    return {'frozen': frozen, 'fixed_endpoint_count': count, 'all_updates_supported': success,
            'reference_failures': failures, 'metrics': summary if success else None,
            'pooled_displacement_change_rmse_ms': pooled_shift if success else None, 'replays': runs}


