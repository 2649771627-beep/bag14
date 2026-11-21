"""Recompute same-observation displacement responses on frozen own08/09 pairs."""
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle'
BASELINE = 'raw_completed_pairs'
PAIRED = 'difference'
METHODS = (BASELINE, PAIRED)


def load_verified_bundle():
    spec = importlib.util.spec_from_file_location('verified_own_replay', BUNDLE / 'run.py')
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    return package, package.integrity()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def run_matrix(pd, members, pairs, frozen, config, reference, model):
    ids = [late for _, late in pairs]
    conditions = {}
    for case, (amplitude, directional) in pd.SHIFTS.items():
        conditions[case] = {}
        for injection in pd.INJECTIONS_MS:
            events = pd.perturb(members, pairs, frozen, amplitude, directional, injection)
            for i, event in events.items():
                assert event.receive_s == members[i].receive_s
                np.testing.assert_allclose(event.reported_s - members[i].reported_s,
                                           injection / 1000, rtol=0, atol=5e-14)
            result = pd.replay(events, pairs, config, reference, model, frozen['prior_receive_s'])
            assert all(list(result['outputs'][method]) == ids for method in pd.METHODS)
            final = result['outputs']
            assert final[BASELINE][ids[-1]]['offset_s'] == final['raw_stream'][ids[-1]]['offset_s']
            conditions[case][injection] = result
    return conditions


def summarize(pd, conditions, pairs, members):
    endpoints = [late for _, late in pairs]
    baseline = conditions['nominal'][0]['outputs']
    failures = {case: {injection: result['reference_failures'] for injection, result in branch.items()}
                for case, branch in conditions.items()}
    rows = []
    for case, branch in conditions.items():
        for injection, run in branch.items():
            for method in METHODS:
                output = run['outputs']
                spatial = pd.endpoint_delta(output, conditions['nominal'][injection]['outputs'], method, endpoints)
                increment_error = pd.endpoint_delta(output, branch[0]['outputs'], method, endpoints, injection)
                for endpoint, change, error in zip(endpoints, spatial, increment_error, strict=True):
                    rows.append(dict(case=case, injection_ms=injection, method=method, late_id=endpoint,
                                     receive_s=members[endpoint].receive_s, offset_ms=1000 * output[method][endpoint]['offset_s'],
                                     spatial_response_ms=change, injection_recovery_error_ms=error if injection else None,
                                     output_reference_failures=failures[case][injection][method]))
    spatial_values = {kind: {method: [r['spatial_response_ms'] for r in rows
                       if r['case'].startswith(kind + '_') and not r['injection_ms'] and r['method'] == method]
                       for method in METHODS} for kind in ('common', 'direction')}
    common = {method: rms(values) for method, values in spatial_values['common'].items()}
    directional = {method: rms(values) for method, values in spatial_values['direction'].items()}
    nominal = {method: rms([r['injection_recovery_error_ms'] for r in rows
                 if r['case'] == 'nominal' and r['injection_ms'] and r['method'] == method]) for method in METHODS}
    for case in ('common_plus', 'common_minus'):
        for injection in pd.INJECTIONS_MS:
            changes = pd.endpoint_delta(conditions[case][injection]['outputs'],
                                        conditions['nominal'][injection]['outputs'], PAIRED, endpoints)
            np.testing.assert_allclose(changes, 0, rtol=0, atol=1e-9)
    assert all(len(values) == 2 * len(pairs) for kind in spatial_values.values() for values in kind.values())
    supported = not any(ids for branch in failures.values() for injection in branch.values() for ids in injection.values())
    spatial_supported = not any(ids for branch in failures.values() for ids in branch[0].values())
    nominal_supported = not any(ids for branch in failures['nominal'].values() for ids in branch.values())
    return {
        'all_updates_supported': supported,
        'spatial_updates_supported': spatial_supported,
        'nominal_injection_updates_supported': nominal_supported,
        'common_rms_ms': common if spatial_supported else dict.fromkeys(METHODS),
        'direction_rms_ms': directional if spatial_supported else dict.fromkeys(METHODS),
        'nominal_injection_rmse_ms': nominal if nominal_supported else dict.fromkeys(METHODS),
        'common_reduction_ms': common[BASELINE] - common[PAIRED] if spatial_supported else None,
        'common_reduction_percent': 100 * (1 - common[PAIRED] / common[BASELINE]) if spatial_supported else None,
        'nominal_offset_at_last_pair_ms': {m: 1000 * baseline[m][endpoints[-1]]['offset_s'] for m in METHODS},
        'reference_failures': failures,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'replayed_08_09')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle, module_paths = load_verified_bundle()
    base, pd, fair, adapter = bundle.base, bundle.base.pd, bundle.fair, bundle.adapter
    cfg = base.load(base.REPRO/'inputs/filter_config.json')
    geom = (np.eye(4), np.asarray(base.load(base.REPRO/'inputs/radar_from_lidar.json')), np.eye(3), np.zeros(4))
    identities = base.load(base.REPRO/'inputs/event_identity.json')
    gold = base.load(BUNDLE/'inputs/expected_own.json')
    old_header = base.load(base.REPRO/'inputs/expected.json')['pairs']
    manifest = base.load(BUNDLE/'manifest.json')
    rows, endpoints, checks, detail = [], [], {}, {}

    for name, expected_counts in {'08': (36, 29, 18), '09': (13, 12, 7)}.items():
        identity = identities[name]
        source = base.load(BUNDLE/'inputs'/f'epoch_source_{name}.json')
        converted, ledger = adapter.convert(source)
        compact = base.load(base.REPRO/'inputs'/f'recording_{name}.json')
        target, _ = base.corrected_rows(name, compact)
        assert [r['t'] for r in converted['lidar']] == [r['t'] for r in target]
        assert [[r[k] for k in ('x','y','z')] for r in converted['lidar']] == [[r[k] for k in ('x','y','z')] for r in target]
        assert source['radar'] == converted['radar'] == compact['radar']
        events, pairs, oldref, selected = base.rep.load_recording(name, cfg, geom, identity)
        frozen, members = pd.freeze_events(events, pairs, oldref, selected, identity)
        newref = base.TargetReference(target, cfg['reference_gap_s'], cfg['reference_smooth_half_window_s'])
        model = base.VehicleObservationModel(geom)
        selected_counts = {epoch: base.pair_audit(events, pairs, ref, model, cfg, np.asarray(identity['axis']))['eligible_pairs']
                           for epoch, ref in [('header_epoch', oldref), ('target_point_epoch', newref)]}
        assert (len(pairs), selected_counts['target_point_epoch']) == expected_counts[:2]
        prepared = fair.prepare(name, cfg, geom, identities)
        fz = prepared['frozen']
        for key in ('pairsets','axis','direction','prior_receive_s','shared_terminal_receive_s'):
            assert fz[key] == gold['fair'][name]['frozen'][key], (name, key)
        intersection = prepared['pairsets']['identical_pair_intersection']
        assert len(intersection) == expected_counts[2]
        keys = sorted(i for pair in intersection for i in pair)
        strict_members = {i: prepared['events'][i] for i in keys}
        strict_frozen = dict(axis_lidar_xy=fz['axis'][:2], prior_receive_s=fz['prior_receive_s'],
                             members=[dict(event_id=i, direction=fz['direction'][i]) for i in keys])
        branches = [
            ('old_frozen','header_epoch',members,pairs,frozen,oldref,selected_counts['header_epoch']),
            ('old_frozen','target_point_epoch',members,pairs,frozen,newref,selected_counts['target_point_epoch']),
            ('identical_pair_intersection','header_epoch',strict_members,intersection,strict_frozen,prepared['oldref'],len(intersection)),
            ('identical_pair_intersection','target_point_epoch',strict_members,intersection,strict_frozen,prepared['newref'],len(intersection)),
        ]
        checks[name] = {'exact_epoch_conversion_rows': len(ledger), 'source_full_track_sha256': None,
            'source_compact_sha256': base.sha(base.REPRO/'inputs'/f'recording_{name}.json'), 'package_sha256': {
            key:value for key,value in manifest['package_sha256'].items() if key.startswith('inputs/') and name in key},
            'archive_header_max_endpoint_delta_ms': None, 'fixed_nominal_max_state_delta_s': {},
            'strict_matrix_max_shared_terminal_delta_ms': {}}
        detail[name] = {}
        for label, epoch, cohort, pairlist, definition, ref, eligible in branches:
            conditions = run_matrix(pd, cohort, pairlist, definition, cfg, ref, model)
            summary, endpoint_rows = summarize(pd, conditions, pairlist, cohort)
            if label == 'old_frozen':
                delta = bundle.offsets_equal(conditions['nominal'][0]['outputs'], gold['fixed'][name][epoch]['outputs'])
                checks[name]['fixed_nominal_max_state_delta_s'][epoch] = delta
                if epoch == 'header_epoch':
                    known = old_header[name]['offsets_s']
                    maximum = max(abs(state['offset_s'] - known[case][str(injection)][method][str(i)])
                                  for case, branch in conditions.items() for injection, result in branch.items()
                                  for method, values in result['outputs'].items() for i, state in values.items())
                    assert maximum < 1e-8
                    checks[name]['archive_header_max_endpoint_delta_ms'] = 1000 * maximum
            else:
                maximum = 0
                for case, branch in conditions.items():
                    for injection, res in branch.items():
                        terminal = fair.finish_state(res, pairlist, prepared['events'], fz['shared_terminal_receive_s'])
                        expected = gold['matrix_terminal'][name][epoch][case][str(injection)]
                        maximum = max(maximum, *(abs(terminal[m]['shared_terminal_offset_s'] - expected[m]['shared_terminal_offset_s']) for m in METHODS))
                assert maximum < 1e-8
                checks[name]['strict_matrix_max_shared_terminal_delta_ms'][epoch] = 1000 * maximum
            prefix = dict(recording=name, epoch=epoch, pair_set=label)
            rows.append(dict(**prefix, pairs=len(pairlist), eligible_pair_count=eligible,
                             prior_receive_s=definition['prior_receive_s'], last_receive_s=cohort[pairlist[-1][1]].receive_s, **summary))
            endpoints.extend(dict(**prefix, **row) for row in endpoint_rows)
            detail[name].setdefault(label,{})[epoch] = dict(frozen=definition,pairs=pairlist,replays=conditions)

    scope = ('RMS of numerically imposed displacement responses on the recordings. '
             'Each condition uses the same observed frames, prior epoch and completed-pair epochs. '
             'Unknown absolute delay; original 50 ms diagnosis and innovation 2 unchanged.')
    base.save(out/'summary.json', dict(rows=rows, scope=scope, checks=checks, local_module_paths=module_paths,
             original_profile_config=base.pe.CONFIG, innovation2_changed=False))
    base.save(out/'replays.json', detail)
    with (out/'endpoints.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(endpoints[0]))
        writer.writeheader()
        writer.writerows(endpoints)
    for r in rows:
        print(r['recording'], r['pair_set'], r['epoch'], r['pairs'], r['eligible_pair_count'],
              r['common_rms_ms'], r['direction_rms_ms'], r['nominal_injection_rmse_ms'], r['all_updates_supported'])


if __name__ == '__main__':
    main()
