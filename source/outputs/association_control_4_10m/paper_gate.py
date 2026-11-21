"""Apply the unmodified reviewer-replay 2-D 'direction' profile to synthetic streams.

All XY observations already share the simulation frame. The simulation assumes
coincident nominal origins and aligned axes, hence identity radar extrinsics.
LiDAR XY is observed with noise, z is fixed at zero. TargetReference uses the
paper's 0.35 s gap and 0.45 s local quadratic half-window. select_observations
uses the paper's 0.13 s recording-time spacing and support check. Each radar
stamp equals acquisition time + injected offset; receive time equals acquisition
time + 0.20 s, a synthetic constant that preserves inter-arrival times.
The full profile uses the paper's radar sigmas (0.15, 0.10 m), ±600 ms search,
20 ms boundary margin, and unchanged 1 s block / 50 ms validity diagnostic.
These verdicts evaluate the 2-D replay estimator separately from fit.py.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / 'mst_review_20261117/reviewer_replay/bundle/core'
sys.path.insert(0, str(CORE))
from experiments.paired_ekf import select_observations
from profile_estimator import CONFIG, ProfileProblem
from src.temporal_model.vehicle import TargetReference, VehicleObservationModel

FILTER = json.loads((CORE / 'inputs/filter_config.json').read_text(encoding='utf-8'))
GEOMETRY = (np.eye(4), np.eye(4), np.eye(3), np.zeros(4))
OBSERVATIONS = ('complete', 'oracle_missing', 'tracked')


def make_reference(t, x, y):
    rows = [dict(t=stamp, x=px, y=py, z=0.0) for stamp, px, py in zip(t, x, y)]
    return TargetReference(rows, FILTER['reference_gap_s'], FILTER['reference_smooth_half_window_s'])


def evaluate(reference, radar, offset_ms):
    rows = [dict(t=t + offset_ms / 1000.0, x=x, y=y,
                 bag_record_ns=round((t + 0.20) * 1e9)) for t, x, y in radar]
    events, _, _ = select_observations(rows, 'radar', 0, FILTER, reference,
                                       VehicleObservationModel(GEOMETRY))
    problem = ProfileProblem(events, reference, GEOMETRY, 'radar', FILTER['sensors']['radar']['sigma'])
    fit = problem.fit('direction', split_check=True)
    time_scale = fit.get('block_inflated_offset_std_s')
    offset = fit.get('offset_s')
    reasons = fit['reasons']
    return dict(offset_ms=offset_ms, estimate_ms=None if offset is None else 1000 * offset,
                scale_ms=None if time_scale is None else 1000 * time_scale,
                accepted=fit['accepted'], reasons='|'.join(reasons),
                gate50_pass=(None if offset is None else time_scale is not None
                             and time_scale <= CONFIG['max_block_inflated_offset_std_s']),
                boundary_margin_flag='search_boundary' in reasons,
                n_stream=len(radar), n_selected=len(events),
                n_fit=fit['support']['retained_events'],
                n_positive=fit['support']['positive_events'], n_negative=fit['support']['negative_events'])


def run_trial(task):
    motion, seed, streams_path = task
    with np.load(streams_path, allow_pickle=False) as data:
        reference = make_reference(*(data[f'lidar_{axis}_{motion}_{seed}'] for axis in ('t', 'x', 'y')))
        rows = []
        for observation in OBSERVATIONS:
            radar = data[f'radar_{motion}_{seed}_{observation}']
            for offset_ms in (70, 160):
                rows.append(dict(motion=motion, seed=seed, observation=observation,
                                 **evaluate(reference, radar, offset_ms)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--streams', type=Path, default=HERE / 'streams.npz')
    parser.add_argument('--output', type=Path, default=HERE / 'paper_gate.csv')
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    tasks = [(motion, seed, args.streams) for motion in ('near_constant', 'variable') for seed in range(30)]
    temporary = args.output.with_suffix('.csv.partial')
    count = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, temporary.open('w', newline='', encoding='utf-8') as out:
        for batch in pool.map(run_trial, tasks):
            if count == 0:
                writer = csv.DictWriter(out, fieldnames=list(batch[0]))
                writer.writeheader()
            writer.writerows(batch)
            out.flush()
            count += len(batch)
            if count % 60 == 0:
                print(f'{count}/360 profiles', flush=True)
    if count != 360:
        raise ValueError(f'Expected 360 profiles, found {count}.')
    temporary.replace(args.output)


if __name__ == '__main__':
    main()
