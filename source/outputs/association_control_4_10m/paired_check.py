"""Run the paper's same-observation paired and framewise EKFs at their last common endpoint.

Reference, zero prior, radar covariance and geometry are those of paper_gate.py.
The estimator only sees the radar's reported/receive times and noisy observations.
At each trial/arm/offset, the original matcher selects non-reused pair members.
Both filters start at the earliest member's receive epoch and end at the last
pair endpoint. Endpoints and membership can differ between arms. These two
filters have no direction-specific center state, and no bounded profile search.
"""
import csv
from pathlib import Path

from paper_gate import FILTER, GEOMETRY, OBSERVATIONS, make_reference
import numpy as np
from experiments.paired_ekf import match_pairs, select_observations
from src.ekf_estimator.filter import TemporalEKF
from src.ekf_estimator.paired import PairedTemporalEKF
from src.temporal_model.vehicle import UnsupportedReference, VehicleObservationModel


HERE = Path(__file__).resolve().parent


def simulate_clock(stream, offset_ms):
    return [dict(t=t + offset_ms/1000, x=x, y=y, bag_record_ns=round((t+0.20)*1e9))
            for t, x, y in stream]


def estimate(rows, reference, validate=False):
    model = VehicleObservationModel(GEOMETRY)
    events, positions, slopes = select_observations(rows, "radar", 0, FILTER, reference, model)
    pairs = match_pairs(events, positions, slopes, "radar", FILTER)
    members = sorted(index for pair in pairs for index in pair)
    record = dict(n_selected=len(events), n_pairs=len(pairs), n_members=len(members),
                  prior_receive_s=None, terminal_receive_s=None, framewise_ms=None,
                  paired_ms=None, framewise_failures=0, paired_failures=0)
    if not pairs:
        return record
    assert len(set(members)) == 2 * len(pairs)
    assert members[-1] == pairs[-1][1]
    prior = events[members[0]].receive_s
    terminal = events[members[-1]].receive_s
    record.update(prior_receive_s=prior, terminal_receive_s=terminal)
    filters = (("framewise", TemporalEKF), ("paired", PairedTemporalEKF))
    for name, cls in filters:
        estimator = cls(FILTER["sensors"], FILTER["filter"], reference, model)
        estimator.predict_to(prior)
        failures = 0
        output = None
        updates = [(None, i) for i in members] if name == "framewise" else pairs
        for early, late in updates:
            try:
                if early is None:
                    output = estimator.step(events[late])
                else:
                    output = estimator.step_pair(events[early], events[late],
                                                 FILTER["sensors"]["radar"]["geometry_sigma"])
            except UnsupportedReference:
                failures += 1
        assert abs(estimator.last_receive_s - terminal) < 1e-9
        record[name + "_failures"] = failures
        if failures == 0:
            record[name + "_ms"] = 1000 * output["offset_s"]
    if validate and not (record["framewise_failures"] or record["paired_failures"]):
        # Full published prefix replay must agree at the last completed pair.
        from paired_displacement import replay
        original = replay({i: events[i] for i in members}, pairs, FILTER, reference, model, prior)
        late = pairs[-1][1]
        for name, key in (("framewise", "raw_completed_pairs"), ("paired", "difference")):
            np.testing.assert_allclose(record[name + "_ms"],
                                       1000 * original["outputs"][key][late]["offset_s"],
                                       atol=1e-10, rtol=0)
    return record


def main():
    results = []
    with np.load(HERE / "streams.npz", allow_pickle=False) as archive:
        for motion in ("near_constant", "variable"):
            for seed in range(30):
                tag = f"{motion}_{seed}"
                reference = make_reference(*(archive[f"lidar_{axis}_{tag}"] for axis in ("t","x","y")))
                for observation in OBSERVATIONS:
                    stream = archive[f"radar_{tag}_{observation}"]
                    for truth in (70,160):
                        rows = simulate_clock(stream, truth)
                        verify = seed == 0 and observation == "complete" and truth == 70
                        result = estimate(rows, reference, verify)
                        for name in ("framewise", "paired"):
                            value = result[name + "_ms"]
                            result[name + "_error_ms"] = None if value is None else value - truth
                        results.append(dict(seed=seed, motion=motion, observation=observation,
                                            truth_ms=truth, **result))
    assert len(results) == 360
    with (HERE / "paired_check.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"{len(results)} same-endpoint comparisons written.")


if __name__ == "__main__":
    main()
