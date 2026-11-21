"""Cross spatial and timestamp perturbations on the archived R01 pair schedules."""

import argparse
from collections import defaultdict
import csv
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_FIELD = HERE.parent.parent / "airy_ars408_vehicle_r01"
TIMES_MS = (-30.0, 0.0, 30.0)
SPACE_M = (-0.08, 0.0, 0.08)
RULES = {"primary": 0.45, "min_speed_0p25_sensitivity": 0.25}
METHOD_LABELS = {
    "original_paired": "Original paired EKF",
    "motion_axis_paired": "Motion-axis paired EKF",
    "robust_motion_axis_paired": "Robust motion-axis paired EKF",
    "robust_motion_axis_framewise": "Same-model robust framewise EKF",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def archive_tables(field):
    pair_rows = defaultdict(list)
    for row in read_csv(field / "pairs.csv"):
        pair_rows[row["arm"], row["pair_rule"]].append(row)
    original_trials = {}
    for row in read_csv(field / "replay_trials.csv"):
        key = (row["arm"], row["pair_rule"], row["method"], row["case"])
        assert key not in original_trials
        original_trials[key] = row
    return pair_rows, original_trials


def archive_group(arm, rule, rows, events, directions, replay, originals):
    rows = sorted(rows, key=lambda row: int(row["pair_number"]))
    assert [int(row["pair_number"]) for row in rows] == list(range(1, len(rows) + 1))
    pairs = []
    for row in rows:
        early, late = int(row["early_id"]), int(row["late_id"])
        for event_id, position in ((early, "early"), (late, "late")):
            event = events[event_id]
            assert event.id == event_id
            assert np.isclose(event.receive_s, float(row[f"{position}_receive_s"]), rtol=0, atol=1e-9)
            assert np.isclose(event.reported_s, float(row[f"{position}_reported_s"]), rtol=0, atol=1e-9)
            assert directions[event_id] == int(row[f"{position}_direction"])
        pairs.append((early, late))
    members = sorted(i for pair in pairs for i in pair)
    assert len(set(members)) == len(members)
    assert members[-1] == pairs[-1][1]
    assert all(i < j for i, j in pairs)
    assert all(previous[1] < current[1] for previous, current in zip(pairs, pairs[1:]))
    prior, terminal = events[members[0]].receive_s, events[members[-1]].receive_s
    times = np.asarray([(events[i].receive_s, events[j].receive_s) for i, j in pairs], dtype="<f8")
    digest = sha256(times.tobytes()).hexdigest()
    for method in replay.METHODS:
        archived = originals[arm, rule, method, "nominal"]
        assert int(archived["n_pairs"]) == len(pairs)
        assert archived["pair_schedule_sha256"] == digest
        assert np.isclose(prior, float(archived["prior_receive_s"]), rtol=0, atol=1e-9)
        assert np.isclose(terminal, float(archived["terminal_receive_s"]), rtol=0, atol=1e-9)
    return pairs, {
        "arm": arm, "pair_rule": rule, "n_pairs": len(pairs), "n_members": len(members),
        "same_axis_direction_pairs": sum(bool(directions[i] == directions[j]) for i, j in pairs),
        "prior_receive_s": prior, "terminal_receive_s": terminal, "pair_schedule_sha256": digest,
    }


def archived_case(spatial, injected_ms, common):
    if not spatial:
        if not injected_ms:
            return "nominal"
        return f"time_{'plus' if injected_ms > 0 else 'minus'}_30ms"
    if injected_ms == 0:
        sign = "plus" if spatial > 0 else "minus"
        return f"{'common' if common else 'direction'}_{sign}_8cm"
    return None


def validate_against_archive(trial, old):
    for key in ("output_status", "failed_updates", "valid_updates", "downweighted_updates"):
        assert str(trial[key]) == old[key], (key, trial, old)
    for key in ("estimate_ms", "center_m", "offset_center_correlation", "offset_posterior_std_ms"):
        present = trial[key] is not None
        assert present == bool(old[key]), (key, trial, old)
        if present:
            assert np.isclose(trial[key], float(old[key]), rtol=1e-11, atol=1e-8), (key, trial, old)


def difference(value, baseline):
    return value - baseline if value is not None and baseline is not None else None


def run_group(replay, reference, model, axis, events, directions, settings, pairs, group, originals):
    trials = []
    conditions = [(spatial, injected, False) for spatial in SPACE_M for injected in TIMES_MS]
    conditions += [(spatial, 0.0, True) for spatial in SPACE_M if spatial]
    for spatial, injected_ms, common in conditions:
        if not spatial and not injected_ms:
            inputs = events
        else:
            inputs = [
                replace(event, reported_s=event.reported_s + injected_ms / 1000.0,
                        value=event.value + spatial * (1.0 if common else directions[event.id]) * axis)
                for event in events
            ]
        stress = "common" if common else "direction" if spatial else "none"
        for method in replay.METHODS:
            result = replay.comparison(method, inputs, pairs, directions, axis, reference, model,
                                       settings, group["prior_receive_s"], group["terminal_receive_s"])
            old_case = archived_case(spatial, injected_ms, common)
            if old_case is not None:
                validate_against_archive(result, originals[group["arm"], group["pair_rule"], method, old_case])
            trials.append({
                **group, "method": method, "space_condition": stress,
                "injected_space_m": spatial, "injected_time_ms": injected_ms, **result,
            })
    by_condition = {
        (r["method"], r["space_condition"], r["injected_space_m"], r["injected_time_ms"]): r
        for r in trials
    }
    assert len(by_condition) == len(trials)
    for row in trials:
        method, condition = row["method"], row["space_condition"]
        spatial, injected_ms = row["injected_space_m"], row["injected_time_ms"]
        nominal = by_condition[method, "none", 0.0, 0.0]["estimate_ms"]
        spatial_baseline = by_condition[method, condition, spatial, 0.0]["estimate_ms"]
        spatial_change = difference(spatial_baseline, nominal)
        recovered = difference(row["estimate_ms"], spatial_baseline) if injected_ms else None
        row.update(nominal_baseline_ms=nominal, spatial_baseline_ms=spatial_baseline,
                   spatial_baseline_change_ms=spatial_change,
                   absolute_spatial_baseline_change_ms=abs(spatial_change) if spatial_change is not None else None,
                   recovered_increment_ms=recovered,
                   increment_error_ms=recovered - injected_ms if recovered is not None else None)
    return trials


def metrics(values):
    valid = [abs(value) for value in values if value is not None]
    return (len(valid), len(values), float(np.mean(valid)) if valid else None,
            max(valid) if valid else None)


def summarize(trials):
    grouped = defaultdict(list)
    for row in trials:
        grouped[row["arm"], row["pair_rule"], row["method"]].append(row)
    summary = []
    for (arm, rule, method), rows in grouped.items():
        nominal = next(row for row in rows if row["space_condition"] == "none" and not row["injected_time_ms"])
        out = {
            "arm": arm, "pair_rule": rule, "method": method, "n_pairs": nominal["n_pairs"],
            "nominal_offset_ms": nominal["nominal_baseline_ms"],
            "computed_runs": sum(row["output_status"] == "computed" for row in rows),
            "scheduled_runs": len(rows),
        }
        for name, field, condition, time_injected in (
            ("common", "spatial_baseline_change_ms", "common", False),
            ("direction", "spatial_baseline_change_ms", "direction", False),
            ("nominal_increment_error", "increment_error_ms", "none", True),
            ("crossed_increment_error", "increment_error_ms", "direction", True),
        ):
            values = [row[field] for row in rows if row["space_condition"] == condition
                      and bool(row["injected_time_ms"]) == time_injected]
            count, scheduled, average, worst = metrics(values)
            assert scheduled == (4 if name == "crossed_increment_error" else 2)
            out.update({
                f"{name}_computed": count, f"{name}_scheduled": scheduled,
                f"{name}_mean_abs_ms": average, f"{name}_max_abs_ms": worst,
            })
        summary.append(out)
    return summary


def render(path, trials):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    # Use the variable-speed primary schedule to show the same-observation counterexample.
    support = [row for row in trials if row["arm"] == "variable" and row["pair_rule"] == "primary"
               and row["method"] == "original_paired" and row["space_condition"] != "common"]
    assert len(support) == 9 and len({row["pair_schedule_sha256"] for row in support}) == 1
    colors = ("#176588", "#5B6570", "#AE4A24")
    markers = ("o", "s", "^")
    lines = ("--", ":", "-")
    labels = ("−8 cm", "No shift", "+8 cm")
    with matplotlib.rc_context({
        "font.family": "DejaVu Serif", "font.size": 9.0, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    }):
        fig, panels = plt.subplots(1, 2, figsize=(6.6, 3.12), sharex=True)
        for j, spatial in enumerate(SPACE_M):
            rows = sorted((row for row in support if row["injected_space_m"] == spatial),
                          key=lambda row: row["injected_time_ms"])
            y = [row["estimate_ms"] for row in rows]
            dy = [row["increment_error_ms"] for row in rows]
            style = dict(color=colors[j], marker=markers[j], linestyle=lines[j], lw=1.3, ms=4.4)
            panels[0].plot(TIMES_MS, y, **style)
            # The zero-increment sample defines each baseline so no error is plotted for it.
            panels[1].plot([TIMES_MS[0], TIMES_MS[2]], [dy[0], dy[2]],
                           **{**style, "linestyle": "none"})
        panels[1].axhline(0, color="#81868B", linewidth=0.65, zorder=0)
        panel_titles = ("(a) Terminal offsets", "(b) Increment errors")
        y_labels = ("Estimated offset (ms)", "Recovered minus imposed (ms)")
        for index, panel in enumerate(panels):
            panel.grid(axis="y", color="#D8E1E6", linewidth=0.6)
            panel.set_axisbelow(True)
            panel.set(xticks=TIMES_MS, xlim=(-35, 35), ylabel=y_labels[index])
            panel.xaxis.set_major_locator(MultipleLocator(30))
            panel.tick_params(length=3, width=0.7, pad=3)
            panel.text(0, 1.04, panel_titles[index], transform=panel.transAxes, va="bottom")
        handles = [Line2D([], [], color=c, marker=m, linestyle=ls, lw=1.3, ms=4.4, label=label)
                   for c, m, ls, label in zip(colors, markers, lines, labels)]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.988),
                   ncol=3, frameon=False, handlelength=1.8, columnspacing=1.9)
        fig.supxlabel("Imposed timestamp increment (ms)", x=0.55, y=0.062)
        fig.subplots_adjust(left=0.12, right=0.982, top=0.791, bottom=0.211, wspace=0.35)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "factorial_en.png", dpi=400)
        fig.savefig(path / "factorial_en.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", type=Path, default=DEFAULT_FIELD,
                        help="R01 folder with replay.py, archived pairs/trials and prepared tracks")
    parser.add_argument("--output", type=Path, default=HERE, help="Output directory")
    parser.add_argument("--skip-figure", action="store_true", help="Skip both figure exports")
    args = parser.parse_args()
    field = args.field.resolve()
    sys.path.insert(0, str(field))
    import replay

    cfg = json.loads(replay.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    pair_rows, originals = archive_tables(field)
    trials = []
    groups = []
    for arm in replay.ARMS:
        if (arm, "primary") not in pair_rows:
            continue
        reference, model, axis, _, events, _, slopes, _ = replay.load_arm(field / "tracks", arm, cfg)
        directions = -np.sign(slopes @ axis)
        for rule, speed in RULES.items():
            settings = deepcopy(cfg)
            settings["sensors"]["radar"]["min_slope"] = speed
            pairs, group = archive_group(arm, rule, pair_rows[arm, rule], events, directions,
                                         replay, originals)
            groups.append(group)
            trials.extend(run_group(replay, reference, model, axis, events, directions,
                                    settings, pairs, group, originals))
    assert len(trials) == len(groups) * len(replay.METHODS) * 11
    results = summarize(trials)
    args.output.mkdir(parents=True, exist_ok=True)
    replay.write_csv(args.output / "factorial_trials.csv", trials)
    replay.write_csv(args.output / "factorial_summary.csv", results)
    summary = {
        "recorded_offset_truth_available": False,
        "pair_source": "archived R01 IDs, timestamps, sign labels, priors and terminal epochs",
        "filter_config_sha256": sha256(replay.CONFIG_PATH.read_bytes()).hexdigest(),
        "axis_model_center_prior_std_m": 0.15,
        "crossed_direction_shift_m": list(SPACE_M),
        "crossed_timestamp_increments_ms": list(TIMES_MS),
        "common_displacement_time_injected": False,
        "groups": groups, "methods": METHOD_LABELS,
        "computed_runs": sum(row["output_status"] == "computed" for row in trials),
        "scheduled_runs": len(trials), "summary": results,
    }
    (args.output / "factorial_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.skip_figure:
        render(args.output / "figures", trials)
    print(f"{summary['computed_runs']}/{summary['scheduled_runs']} completed; {len(groups)} frozen schedules")


if __name__ == "__main__":
    main()
