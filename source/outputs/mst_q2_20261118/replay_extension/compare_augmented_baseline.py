"""Compare the production EKFs with a framewise EKF that estimates shared XY bias."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STD_M = (0.15, 0.30, 0.60)
MAIN_STD_M = 0.30
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261118
LOWER_RETURN_S = 7.45
UPPER_OUTBOUND_S = 5.55


def shared_bias_filter(events, scenario_values, cfg, model, reference, sign, prior_std):
    """Vectorize the exact straight-leg EKF over trials with shared covariance."""
    from src.temporal_model.model import initial_covariance, transition

    n = len(scenario_values)
    x = np.zeros((n, 4))
    P = np.zeros((4, 4))
    P[:2, :2] = initial_covariance(cfg['filter'])
    P[2:, 2:] = np.eye(2) * prior_std**2
    sensor = cfg['sensors']['radar']
    R = np.diag(np.square(sensor['sigma']))
    last_receive = events[0].receive_s
    outbound_query_max = float('-inf')
    return_query_min = float('inf')

    for k, event in enumerate(events):
        dt = event.receive_s - last_receive
        clock_F, clock_Q = transition(dt, cfg['filter']['ekf_drift_density'],
                                     cfg['filter']['ekf_offset_density'])
        x[:, :2] = x[:, :2] @ clock_F.T
        F = np.eye(4)
        F[:2, :2] = clock_F
        P = F @ P @ F.T
        P[:2, :2] += clock_Q
        last_receive = event.receive_s

        query = event.reported_s - x[:, 0]
        if sign[k] > 0:
            outbound_query_max = max(outbound_query_max, float(query.max()))
            assert query.max() < UPPER_OUTBOUND_S
        else:
            return_query_min = min(return_query_min, float(query.min()))
            assert query.min() > LOWER_RETURN_S
        expected, offset_slope = model(event, 0., sensor, reference)
        for b in (float(x[:, 0].min()), float(x[:, 0].max())):
            actual_expected, actual_slope = model(event, b, sensor, reference)
            np.testing.assert_allclose(actual_expected, expected + b * offset_slope,
                                       atol=2e-11, rtol=0)
            np.testing.assert_allclose(actual_slope, offset_slope, atol=2e-11, rtol=0)
        H = np.column_stack((offset_slope, np.zeros(2), np.eye(2)))
        cross = P @ H.T
        K = np.linalg.solve(H @ cross + R, cross.T).T
        z_pred = expected + x @ H.T
        x += (scenario_values[:, k, :] - z_pred) @ K.T
        left = np.eye(4) - K @ H
        P = left @ P @ left.T + K @ R @ K.T
        P = (P + P.T) * 0.5

    assert last_receive == events[-1].receive_s
    return x[:, 0], {
        'last_outbound_query_s': outbound_query_max,
        'earliest_return_query_s': return_query_min,
    }


def rmse(values):
    return float(np.sqrt(np.mean(values**2)))


def comparison(errors, bootstrap):
    values = {method: rmse(err) for method, err in errors.items()}
    paired = errors['paired_ekf']
    raw = errors['framewise_ekf']
    augmented = errors['framewise_shared_bias_std_0.30m']
    # The first axis is the shared noise draw, including when polarities are pooled.
    def boot_rmse(e):
        selected = e[bootstrap]
        return np.sqrt(np.mean(selected**2, axis=tuple(range(1, selected.ndim))))

    return {
        'rmse_ms': values,
        'paired_minus_augmented_rmse_ms': rmse(paired) - rmse(augmented),
        'paired_minus_augmented_95pct_mc_ci_ms': np.quantile(
            boot_rmse(paired) - boot_rmse(augmented), [.025, .975]).tolist(),
        'paired_minus_raw_95pct_mc_ci_ms': np.quantile(
            boot_rmse(paired) - boot_rmse(raw), [.025, .975]).tolist(),
    }


def simulate(replay_root):
    sys.path.insert(0, str(replay_root / 'simulation'))
    from run_known_truth import (MOTIONS, N_PAIRS, REPETITIONS, REPRO, SEED, SHIFTS,
                                 TRUTH_S, Observation, build_trial, endpoint_estimators,
                                 metrics)

    cfg_path = REPRO / 'inputs/filter_config.json'
    archive_path = replay_root / 'expected/simulation/signed_endpoint_errors_ms.npz'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    sigma = np.asarray(cfg['sensors']['radar']['sigma'])
    noise = np.random.default_rng(SEED).normal(size=(REPETITIONS, 2*N_PAIRS, 2)) * sigma
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).integers(
        REPETITIONS, size=(BOOTSTRAP_REPLICATES, REPETITIONS))
    designs = {}
    errors_archive = {}
    with np.load(archive_path, allow_pickle=False) as saved:
        for motion, (u, w) in MOTIONS.items():
            events, pairs, model, ref, true_x, _, sign = build_trial(u, w, cfg)
            conditions = {}
            all_errors = {}
            archived_match = 0.
            raw_match = 0.
            for condition, (a, c) in SHIFTS.items():
                observations = noise.copy()
                observations[:, :, 0] += true_x + a + c * sign
                errors = {m: saved[f'{motion}__{condition}__{m}']
                          for m in ('framewise_ekf', 'paired_ekf')}
                # Re-run the production executables on the first and last shared draws.
                for trial in (0, REPETITIONS-1):
                    trial_obs = [Observation(e.id, e.sensor, e.reported_s, e.receive_s, value)
                                 for e, value in zip(events, observations[trial], strict=True)]
                    raw, paired, _, _ = endpoint_estimators(trial_obs, pairs, cfg, model, ref)
                    for key, value in (('framewise_ekf', raw), ('paired_ekf', paired)):
                        terminal_error = 1000 * (value - TRUTH_S)
                        archived_match = max(archived_match, abs(errors[key][trial] - terminal_error))
                # A zero intercept covariance must reproduce the original framewise EKF.
                raw_offset, _ = shared_bias_filter(events, observations, cfg, model, ref, sign, 0.)
                raw_match = max(raw_match, float(np.max(np.abs(
                    1000 * (raw_offset - TRUTH_S) - errors['framewise_ekf']))))
                support = {}
                for prior in STD_M:
                    name = f'framewise_shared_bias_std_{prior:.2f}m'
                    offset, query_support = shared_bias_filter(events, observations, cfg, model, ref, sign, prior)
                    errors[name] = 1000 * (offset - TRUTH_S)
                    support[name] = query_support
                all_errors[condition] = errors
                for method, err in errors.items():
                    errors_archive[f'{motion}__{condition}__{method}'] = err
                conditions[condition] = {
                    'common_shift_m': a,
                    'direction_shift_m': c,
                    'statistics': {method: metrics(err) for method, err in errors.items()},
                    'query_support': support,
                    **comparison(errors, bootstrap),
                }

            groups = {
                'common_polarities': ('common_plus', 'common_minus'),
                'direction_polarities': ('direction_plus', 'direction_minus'),
            }
            pooled = {}
            for group, (plus, minus) in groups.items():
                combined = {name: np.column_stack((all_errors[plus][name], all_errors[minus][name]))
                            for name in all_errors[plus]}
                pooled[group] = comparison(combined, bootstrap)
            assert archived_match < 1e-8, archived_match
            assert raw_match < 1e-8, raw_match
            designs[motion] = {
                'outbound_speed_mps': u,
                'return_speed_mps': w,
                'pair_count': len(pairs),
                'pairs': pairs,
                'prior_receive_s': events[0].receive_s,
                'terminal_receive_s': events[-1].receive_s,
                'max_replayed_archived_production_error_difference_ms': archived_match,
                'max_framewise_zero_prior_equivalence_error_ms': raw_match,
                'conditions': conditions,
                'pooled': pooled,
            }
    source_files = [cfg_path, archive_path, replay_root / 'simulation/run_known_truth.py',
                    REPRO / 'src/temporal_model/model.py', REPRO / 'src/temporal_model/vehicle.py',
                    REPRO / 'src/ekf_estimator/filter.py', REPRO / 'src/ekf_estimator/paired.py']
    report = {
        'scope': 'Synthetic known-clock terminal comparison under the specified motion, prior and noise.',
        'trial_count': REPETITIONS,
        'seed': SEED,
        'offset_truth_ms': 1000 * TRUTH_S,
        'drift_truth_sps': 0.,
        'shared_xy_prior_std_m': STD_M,
        'primary_shared_xy_prior_std_m': MAIN_STD_M,
        'shared_xy_process_noise': 0.,
        'bootstrap_replicates': BOOTSTRAP_REPLICATES,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': {p.relative_to(replay_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
        'designs': designs,
    }
    return report, errors_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewer-replay', '--replay-dir', required=True, type=Path)
    args = parser.parse_args()
    result, errors = simulate(args.reviewer_replay.resolve())
    (HERE / 'augmented_baseline_results.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    np.savez_compressed(HERE / 'augmented_baseline_signed_errors_ms.npz', **errors)
    for name, design in result['designs'].items():
        print(name)
        for use, metrics in {**design['conditions'], **design['pooled']}.items():
            print(use, {method: round(value, 4) for method, value in metrics['rmse_ms'].items()},
                  'paired - augmented RMSE CI (ms)', metrics['paired_minus_augmented_95pct_mc_ci_ms'])


if __name__ == '__main__':
    main()
