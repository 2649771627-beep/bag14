"""Recompute the pooled simulation gain using paired bootstrap resampling."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('signed_errors', type=Path)
    args = parser.parse_args()
    motion = 'unequal_1p2_1p8'
    methods = ('framewise_ekf', 'paired_ekf')
    polarities = ('common_plus', 'common_minus')
    with np.load(args.signed_errors, allow_pickle=False) as signed:
        per_trial_mse = np.stack([
            np.mean([signed[f'{motion}__{polarity}__{method}'] ** 2
                     for polarity in polarities], axis=0)
            for method in methods
        ], axis=1)
    assert per_trial_mse.shape == (500, 2)
    iterations, seed = 2000, 20261118
    draw = np.random.default_rng(seed).integers(0, len(per_trial_mse),
                                               size=(iterations, len(per_trial_mse)))
    rmse = np.sqrt(per_trial_mse.mean(axis=0))
    boot = np.sqrt(per_trial_mse[draw].mean(axis=1))
    intervals = np.quantile(100 * (1 - boot[:, 1] / boot[:, 0]), [0.025, 0.975])
    print(json.dumps({'simulation_trials': len(per_trial_mse), 'bootstrap_trials': iterations,
                      'bootstrap_seed': seed, 'pooled_rmse_ms': rmse.tolist(),
                      'relative_reduction_percent': float(100 * (1 - rmse[1] / rmse[0])),
                      'percentile_interval_95_percent': intervals.tolist()}, indent=2))


if __name__ == '__main__':
    main()
