"""Reproduce the strict-direction replay and the simulation scale check."""

import argparse
import csv
from dataclasses import replace
from hashlib import sha256
from itertools import groupby
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent.parent
DEFAULT_FIELD = OUTPUTS / "airy_ars408_vehicle_r01"
DEFAULT_SIMULATION = OUTPUTS / "airy_ars408_sim/trials.csv"
DEFAULT_FACTORIAL = HERE.parent / "factorial/factorial_trials.csv"
ARM = "variable"
RULE = "primary"
SIMULATION_METHOD = "robust_axis_paired"


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replay_inputs(field, tracks, pair_csv, replay_trials):
    sys.path[:0] = [str(field), str(HERE.parent / "factorial")]
    import replay
    from run_factorial import SPACE_M, TIMES_MS, archive_group, read_csv

    cfg = json.loads(replay.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    reference, model, axis, _, events, _, slopes, _ = replay.load_arm(tracks, ARM, cfg)
    directions = -np.sign(slopes @ axis)
    pair_rows = [row for row in read_csv(pair_csv)
                 if row["arm"] == ARM and row["pair_rule"] == RULE]
    originals = {(r["arm"], r["pair_rule"], r["method"], r["case"]): r
                 for r in read_csv(replay_trials)}
    pairs, schedule = archive_group(ARM, RULE, pair_rows, events, directions, replay, originals)
    strict = [(i, j) for i, j in pairs if directions[i] * directions[j] == -1]
    rejected = [(i, j) for i, j in pairs if directions[i] * directions[j] != -1]
    assert len(pairs) == 110 and len(strict) == 108
    assert rejected == [(148, 197), (1281, 1376)]
    assert directions[148] == directions[197] == directions[1281] == directions[1376] == -1
    assert len({i for pair in strict for i in pair}) == 216
    prior = schedule["prior_receive_s"]
    terminal = schedule["terminal_receive_s"]
    assert events[min(i for i, _ in strict)].receive_s == prior
    assert events[strict[-1][1]].receive_s == terminal
    return (replay, read_csv, cfg, reference, model, axis, directions, events, SPACE_M, TIMES_MS,
            strict, rejected, schedule)


def strict_opposite_check(field, tracks, pair_csv, replay_trials, factorial):
    (replay, read_csv, cfg, reference, model, axis, directions, events, spatial_settings,
     timestamp_settings, strict, rejected, schedule) = replay_inputs(field, tracks, pair_csv, replay_trials)
    selected = [r for r in read_csv(factorial)
                if r["arm"] == ARM and r["pair_rule"] == RULE and r["space_condition"] != "common"]
    prior = schedule["prior_receive_s"]
    terminal = schedule["terminal_receive_s"]
    parent_digest = schedule["pair_schedule_sha256"]
    assert len(selected) == 36 and {r["pair_schedule_sha256"] for r in selected} == {parent_digest}
    for row in selected:
        assert int(row["n_pairs"]) == 110 and int(row["n_members"]) == 220
        assert row["output_status"] == "computed" and int(row["failed_updates"]) == 0
        assert float(row["prior_receive_s"]) == prior and float(row["terminal_receive_s"]) == terminal
    archived = {(r["method"], float(r["injected_space_m"]), float(r["injected_time_ms"])): r
                for r in selected}
    assert len(archived) == len(selected)

    strict_times = np.asarray([(events[i].receive_s, events[j].receive_s) for i, j in strict], dtype="<f8")
    strict_digest = sha256(strict_times.tobytes()).hexdigest()
    assert strict_digest != parent_digest
    replayed = {}
    for spatial in spatial_settings:
        for increment in timestamp_settings:
            disturbed = [replace(event, reported_s=event.reported_s + increment / 1000,
                                 value=event.value + spatial * directions[event.id] * axis)
                         for event in events]
            for method in replay.METHODS:
                replayed[method, spatial, increment] = replay.comparison(
                    method, disturbed, strict, directions, axis, reference, model, cfg, prior, terminal,
                )

    trials = []
    summary = []
    for support, records, n_pair, digest in (
        ("original_110", archived, 110, parent_digest),
        ("strict_108", replayed, 108, strict_digest),
    ):
        for method in replay.METHODS:
            nominal = float(records[method, 0, 0]["estimate_ms"])
            cross_errors = []
            shifts = {}
            for spatial in spatial_settings:
                shifted = float(records[method, spatial, 0]["estimate_ms"])
                spatial_change = shifted - nominal
                shifts[spatial] = spatial_change
                for increment in timestamp_settings:
                    result = records[method, spatial, increment]
                    assert result["output_status"] == "computed" and int(result["failed_updates"]) == 0
                    estimate = float(result["estimate_ms"])
                    recovered = estimate - shifted
                    increment_error = recovered - increment
                    if spatial and increment:
                        cross_errors.append((spatial, increment, increment_error))
                    trials.append(dict(
                        support=support, method=method, n_pairs=n_pair, n_members=2 * n_pair,
                        pair_schedule_sha256=digest, prior_receive_s=prior, terminal_receive_s=terminal,
                        spatial_shift_m=spatial, timestamp_increment_ms=increment, estimate_ms=estimate,
                        zero_shift_baseline_ms=nominal, same_displacement_zero_increment_ms=shifted,
                        fitted_baseline_change_ms=spatial_change, recovered_increment_ms=recovered,
                        increment_error_ms=increment_error, output_status=result["output_status"],
                        failed_updates=int(result["failed_updates"]), valid_updates=int(result["valid_updates"]),
                        downweighted_updates=int(result["downweighted_updates"]),
                    ))
            assert len(cross_errors) == 4
            spatial, increment, worst = max(cross_errors, key=lambda item: abs(item[2]))
            summary.append(dict(
                support=support, method=method, n_pairs=n_pair, n_members=2 * n_pair,
                zero_shift_baseline_ms=nominal, baseline_change_minus_8cm_ms=shifts[-0.08],
                baseline_change_plus_8cm_ms=shifts[0.08], n_nonzero_crossed_settings=len(cross_errors),
                mean_abs_increment_error_ms=math.fsum(abs(row[2]) for row in cross_errors) / 4,
                max_abs_increment_error_ms=abs(worst), worst_spatial_shift_m=spatial,
                worst_timestamp_increment_ms=increment, worst_signed_increment_error_ms=worst,
            ))
    assert len(trials) == 72 and len(summary) == 8
    return trials, summary, dict(
        arm=ARM, original_rule=RULE, original_pairs=110, original_members=220,
        strict_pairs=108, strict_members=216, removed_event_id_pairs=rejected,
        prior_receive_s=prior, terminal_receive_s=terminal,
        original_pair_schedule_sha256=parent_digest, strict_pair_schedule_sha256=strict_digest,
        strict_configurations=36, original_configurations_read=36,
        archived_filter_config_sha256=sha256(replay.CONFIG_PATH.read_bytes()).hexdigest(),
        summaries=summary,
    )


def mean_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    for _, indices in groupby(order, key=values.__getitem__):
        tied = list(indices)
        end = start + len(tied)
        rank = (start + 1 + end) / 2
        for i in tied:
            ranks[i] = rank
        start = end
    return ranks


def spearman(pairs):
    x, y = map(mean_ranks, zip(*pairs))
    centre = (len(pairs) + 1) / 2
    x = [rank - centre for rank in x]
    y = [rank - centre for rank in y]
    covariance = math.fsum(i * j for i, j in zip(x, y))
    return covariance / math.sqrt(math.fsum(i * i for i in x) * math.fsum(j * j for j in y))


def finite_or_none(value):
    if not value:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def simulation_scale_check(path):
    source = [row for row in csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines())
              if row["observation"] == "tracked" and float(row["radar_step_m"]) == 0.0
              and row["method"] == SIMULATION_METHOD]
    source.sort(key=lambda row: (row["motion"], int(row["truth_ms"]), int(row["seed"])))
    assert len(source) == 120
    seen = set()
    records = []
    for row in source:
        key = (row["motion"], int(row["truth_ms"]), int(row["seed"]))
        assert key not in seen
        seen.add(key)
        scale = finite_or_none(row["local_x_data_scale_ms"])
        error = finite_or_none(row["abs_error_ms"])
        failures = int(row["failed_updates"])
        valid = scale is not None and error is not None and failures == 0
        status = "valid"
        if failures:
            status = "failed_reference_update"
        elif not valid:
            status = "unavailable_scale_or_error"
        records.append(dict(
            motion=key[0], truth_ms=key[1], seed=key[2], method=row["method"],
            observation=row["observation"], radar_step_m=float(row["radar_step_m"]),
            pair_schedule_sha256=row["pair_schedule_sha256"], n_pairs=int(row["n_pairs"]),
            local_x_data_scale_ms=scale, estimated_offset_ms=finite_or_none(row["estimate_ms"]),
            abs_error_ms=error, failed_updates=failures, valid_for_correlation=int(valid),
            trial_status=status,
        ))

    def summarise(motion, truth, cohort):
        valid = [(r["local_x_data_scale_ms"], r["abs_error_ms"])
                 for r in cohort if r["valid_for_correlation"]]
        return dict(
            motion=motion, truth_ms=truth, scheduled=len(cohort), valid=len(valid),
            excluded_from_correlation=len(cohort) - len(valid), spearman_rho=spearman(valid),
        )

    groups = [summarise(*key, list(group))
              for key, group in groupby(records, key=lambda r: (r["motion"], r["truth_ms"]))]
    assert [(r["motion"], r["truth_ms"], r["scheduled"], r["valid"]) for r in groups] == [
        ("near_constant", 70, 30, 30), ("near_constant", 160, 30, 30),
        ("variable", 70, 30, 29), ("variable", 160, 30, 30),
    ]
    pooled = summarise("pooled", None, records)
    excluded = [r for r in records if not r["valid_for_correlation"]]
    assert len(excluded) == 1 and excluded[0]["failed_updates"] == 1
    assert excluded[0]["abs_error_ms"] is None
    return records, groups + [pooled], dict(
        scale="local_x_data_scale_ms", terminal_error="abs_error_ms", estimator=SIMULATION_METHOD,
        observation="tracked", radar_step_m=0.0, scheduled=120, valid=119,
        groups=groups, pooled=pooled, excluded_trial=excluded[0],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--replay-trials", type=Path)
    parser.add_argument("--factorial", type=Path, default=DEFAULT_FACTORIAL)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--output", "--out", type=Path, default=HERE)
    args = parser.parse_args()
    field = args.field.resolve()
    strict_trials, strict_summary, strict_result = strict_opposite_check(
        field, args.tracks or field / "tracks", args.pairs or field / "pairs.csv",
        args.replay_trials or field / "replay_trials.csv", args.factorial,
    )
    scale_trials, scale_summary, scale_result = simulation_scale_check(args.simulation)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "strict_opposite_trials.csv", strict_trials)
    write_rows(args.output / "strict_opposite_summary.csv", strict_summary)
    write_rows(args.output / "simulation_scale_trials.csv", scale_trials)
    write_rows(args.output / "simulation_scale_summary.csv", scale_summary)
    result = {
        "strict_opposite_replay": strict_result,
        "simulation_scale_check": scale_result,
    }
    (args.output / "diagnostic_supplement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    strict_original = next(row for row in strict_summary
                           if row["support"] == "strict_108" and row["method"] == "original_paired")
    print(f"Strict replay with 108 pairs. Maximum absolute increment error "
          f"{strict_original['max_abs_increment_error_ms']:.6f} ms.")
    print(f"Simulation diagnostic with {scale_result['valid']}/{scale_result['scheduled']} "
          f"valid runs. Pooled Spearman rho {scale_result['pooled']['spearman_rho']:.6f}.")


if __name__ == "__main__":
    main()
