"""Recover signed fitted-axis velocities from the archived AIRY fixed schedules."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE.parents[1]


def build(archive):
    field = archive / "outputs" / "airy_ars408_vehicle_r01"
    sys.path[:0] = [str(field), str(archive / "outputs" / "mst_airy_submission" / "factorial")]
    import replay
    import run_factorial

    config = json.loads(replay.CONFIG_PATH.read_text(encoding="utf-8"))
    groups, trials = run_factorial.archive_tables(field)
    result = []
    for arm in replay.ARMS:
        _, _, axis, _, events, _, slopes, _ = replay.load_arm(field / "tracks", arm, config)
        signed_velocity = -(slopes @ axis)
        raw = groups[arm, "primary"]
        pairs, _ = run_factorial.archive_group(
            arm, "primary", raw, events, np.sign(signed_velocity), replay, trials
        )
        velocities = signed_velocity[np.asarray(pairs)]
        np.testing.assert_allclose(
            np.abs(velocities).sum(axis=1),
            [float(row["speed_sum_mps"]) for row in raw],
            atol=1e-9, rtol=0,
        )
        for row, (early, late) in zip(raw, velocities, strict=True):
            result.append((arm, row["pair_number"], early, late, int(early * late < 0)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean_run", type=Path, nargs="?", default=DEFAULT_RUN)
    args = parser.parse_args()
    rows = build(args.clean_run.resolve())
    assert len(rows) == 197 and sum(row[-1] for row in rows) == 194
    with (HERE / "field_pair_speeds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("arm", "pair_number", "early_axis_mps", "late_axis_mps", "strict_opposite"))
        writer.writerows(rows)
