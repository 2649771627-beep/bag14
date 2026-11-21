"""Re-run the core experiments from the compact reviewer evidence."""

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

import independent_theory_validation as scalar
import paired_displacement
import profile_estimator as profile
import speed_groups
from experiments.paired_ekf import match_pairs, select_observations
from src.temporal_model.vehicle import TargetReference, VehicleObservationModel


HERE = Path(__file__).resolve().parent
DATA = HERE / 'inputs'
RUNS = ('08', '09')
MODES = ('geometry', 'direction')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_integrity():
    manifest = load(DATA / 'manifest.json')
    for name, expected in manifest['package_sha256'].items():
        actual = sha256((HERE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch in {name}')


def load_recording(name, cfg, geometry, expected):
    tracks = load(DATA / f'recording_{name}.json')
    reference = TargetReference(tracks['lidar'], cfg['reference_gap_s'], cfg['reference_smooth_half_window_s'])
    model = VehicleObservationModel(geometry)
    events, positions, slopes = select_observations(tracks['radar'], 'radar', 0, cfg, reference, model)
    pairs = match_pairs(events, positions, slopes, 'radar', cfg)
    assert pairs == [tuple(p) for p in expected['pairs']]
    row_ids = {(row['t'], row['bag_record_ns'] * 1e-9): row['row_id'] for row in tracks['radar']}
    selected_rows = [row_ids[(e.reported_s, e.receive_s)] for e in events]
    assert selected_rows == expected['selected_radar_row_ids']
    return events, pairs, reference, selected_rows


def profile_checks(recordings, cfg, geometry, expected, output):
    full = [profile.ProfileProblem(events, reference, geometry, 'radar', cfg['sensors']['radar']['sigma'])
            for events, _, reference, _ in recordings]
    indices, matching = speed_groups.choose_groups(full[0])
    problems = {'08_near_constant': speed_groups.subset(full[0], geometry, indices['near_constant']),
                '08_variable': speed_groups.subset(full[0], geometry, indices['variable']),
                '09_full': full[1]}
    groups = {}
    rows = []
    for group, problem in problems.items():
        saved = expected[group]
        details = speed_groups.sample_details(problem)
        assert details['ids'] == saved['ids']
        nominal = speed_groups.fixed_information(problem)
        fits = {mode: problem.fit(mode) for mode in MODES}
        for mode, fit in fits.items():
            previous = saved[mode]
            assert fit['accepted'] == previous['accepted'] and fit['reasons'] == previous['reasons']
            info = nominal[mode]['projected_information_s_minus2']
            np.testing.assert_allclose(info, previous['fixed_information_s_minus2'], rtol=1e-5, atol=1e-8)
            # Brent-free grid refinement and least-squares operations can differ slightly across BLAS builds.
            np.testing.assert_allclose(fit['profile_cost'], previous['profile_cost'], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(fit['profile_grid_s'], previous['profile_grid_s'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(fit['offset_s'], previous['offset_s'], rtol=0, atol=1e-5)
            for metric in ('projected_information', 'block_inflated_offset_std_s', 'split_offsets_s'):
                np.testing.assert_allclose(fit[metric], previous[metric], rtol=1e-4, atol=1e-5, err_msg=f'{group} {mode} {metric}')
            rows.append({'group': group, 'model': mode, 'n': len(details['ids']),
                         'nominal_projected_information_s_minus2': info,
                         'fitted_projected_information_s_minus2': fit['projected_information'],
                         'block_diagnostic_ms': 1000 * fit['block_inflated_offset_std_s'],
                         'accepted': fit['accepted'], 'withhold_reasons': '; '.join(fit['reasons'])})
        groups[group] = {'group': details, 'fixed_jacobian': nominal, 'fits': fits}
    prior = expected['09_archived_revised_profiles']
    for mode, fit in groups['09_full']['fits'].items():
        previous = prior[mode]
        assert fit['support']['mask'] == previous['support_mask']
        assert fit['accepted'] == previous['accepted'] and fit['reasons'] == previous['reasons']
        np.testing.assert_allclose(fit['profile_cost'], previous['profile_cost'], rtol=1e-6, atol=1e-7)
    profile.save(output / 'comparison.json', {'groups': groups, 'matching': matching,
                                             'archive_check': '09 profiles match the archived common support, costs and verdicts'})
    write_csv(output / 'table_profile.csv', rows)


def pair_checks(recordings, cfg, geometry, identity, expected, output):
    runs = {}
    summary_rows = []
    endpoint_rows = []
    for name, loaded in zip(RUNS, recordings):
        events, pairs, reference, rows = loaded
        run = paired_displacement.evaluate(events, pairs, reference, rows, cfg, geometry, identity[name])
        runs[name] = run
        assert run['all_updates_supported']
        assert run['fixed_endpoint_count'] == expected[name]['fixed_endpoint_count']
        for method in paired_displacement.METHODS:
            for key in ('common', 'direction'):
                np.testing.assert_allclose(run['pooled_displacement_change_rmse_ms'][key][method],
                    expected[name]['pooled_displacement_change_rmse_ms'][key][method], rtol=0, atol=1e-5)
            for case, metrics in run['metrics'].items():
                np.testing.assert_allclose(metrics[method]['recovery_pooled_rmse_ms'],
                    expected[name]['recovery_pooled_rmse_ms'][case][method], rtol=0, atol=1e-5)
            summary_rows.append({'recording': name, 'pairs': len(pairs), 'estimator': method,
                'common_shift_change_rms_ms': run['pooled_displacement_change_rmse_ms']['common'][method],
                'direction_shift_change_rms_ms': run['pooled_displacement_change_rmse_ms']['direction'][method],
                **{f'{case}_injection_recovery_rmse_ms': m[method]['recovery_pooled_rmse_ms']
                   for case, m in run['metrics'].items()}})
        baseline = run['replays']['nominal'][0]['outputs']
        for case, conditions in run['replays'].items():
            for injection_ms, result in conditions.items():
                for method, estimates in result['outputs'].items():
                    for i, state in estimates.items():
                        original = (baseline if injection_ms == 0 else conditions[0]['outputs'])[method][i]['offset_s']
                        offset = state['offset_s']
                        compare = expected[name]['offsets_s'][case][str(injection_ms)][method][str(i)]
                        np.testing.assert_allclose(offset, compare, rtol=0, atol=1e-8)
                        endpoint_rows.append({'recording': name, 'scenario': case, 'header_shift_ms': injection_ms,
                           'estimator': method, 'late_event_id': i, 'late_receive_s': events[i].receive_s,
                           'offset_ms': 1000 * offset, 'offset_change_ms': 1000 * (offset - original),
                           'increment_recovery_error_ms': 1000 * (offset - original) - injection_ms if injection_ms else ''})
    profile.save(output / 'experiment.json', {'runs': runs})
    write_csv(output / 'table_pair.csv', summary_rows)
    write_csv(output / 'pair_endpoints.csv', endpoint_rows)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scalar_checks(output, expected):
    # The original scalar script writes using HERE. Redirect its outputs without changing its model.
    scalar.HERE = output
    result = scalar.run()
    for name, group in result['cases'].items():
        saved = expected[name]
        assert group['truth_model_temporally_identifiable'] == saved['truth_model_temporally_identifiable']
        for method, values in group['methods'].items():
            archived = saved['methods'][method]
            assert values['accepted_trials'] == archived['accepted_trials']
            error = values['absolute_rmse_s_on_accepted']
            if error is None:
                assert archived['absolute_rmse_s_on_accepted'] is None
            else:
                np.testing.assert_allclose(error, archived['absolute_rmse_s_on_accepted'], rtol=1e-8, atol=1e-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('reproduced_results'),
                        help='new directory for numerical outputs and optional figures')
    parser.add_argument('--figures', action='store_true', help='regenerate manuscript figures using Matplotlib')
    args = parser.parse_args()
    check_integrity()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = load(DATA / 'filter_config.json')
    radar_from_lidar = np.array(load(DATA / 'radar_from_lidar.json'))
    # VehicleObservationModel uses only this rigid transform in the supplied radar experiments.
    geometry = np.eye(4), radar_from_lidar, np.eye(3), np.zeros(4)
    identity = load(DATA / 'event_identity.json')
    expected = load(DATA / 'expected.json')
    recordings = [load_recording(name, cfg, geometry, identity[name]) for name in RUNS]
    profile_checks(recordings, cfg, geometry, expected['profiles'], output)
    print('Verified both matched 08 speed groups and the 09 whole-support profile against the archive.', flush=True)
    pair_checks(recordings, cfg, geometry, identity, expected['pairs'], output)
    print('Verified 36 / 13 pair identities and all 2,205 endpoint offsets against the archive.', flush=True)
    scalar_checks(output, expected['scalar'])
    if args.figures:
        from plot_profiles import plot as plot_profiles
        from plot_pair_response import plot as plot_pairs
        plot_profiles(output)
        plot_pairs(output)
    print(f'All comparisons passed. Results in {output}')


if __name__ == '__main__':
    main()
