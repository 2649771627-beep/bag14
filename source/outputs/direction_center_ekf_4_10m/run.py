"""Replay the direction-center paired EKF on archived simulated radar observations.

The estimator receives the reported clock and horizontal positions. The physical
clock/direction labels are used only to generate controlled measurements and
score outputs. Profile decisions are historical results of a separate method.
"""
import argparse
import csv
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "repeated_roundtrips_4_10m"
sys.path.insert(0, str(HERE.parent / "association_control_4_10m"))
from paired_check import (FILTER, GEOMETRY, OBSERVATIONS, UnsupportedReference,
                          VehicleObservationModel, make_reference, match_pairs,
                          select_observations, simulate_clock)
sys.path.insert(0, str(HERE.parents[1]/"MultiSensor_TimeSync"/"src"/"ekf_estimator"))
from center_filter import DirectionCenterPairedEKF
from observability import paired_information

MOTIONS = ("near_constant", "variable")
TRUTHS = (70, 160)
SCENARIOS = {"direction": 0.08, "zero": 0.0, "common": 0.0, "opposite": -0.08}


def load_csv(file):
    with file.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key]) if row.get(key) not in (None,"") else None


def prepare(stream, truth, scenario, reference, model):
    t, x, y, _ = stream.T
    if scenario != "direction":
        # All counterfactuals use correct, complete returns and the same noise.
        sign = np.where(np.mod(t, 16.) < 8., 1., -1.)
        replacement = 0.08 if scenario == "common" else SCENARIOS[scenario] * sign
        x = x - 0.08 * sign + replacement
    rows = simulate_clock(np.column_stack((t,x,y)), truth)
    events, pos, slopes = select_observations(rows, "radar", 0, FILTER, reference, model)
    pairs = match_pairs(events, pos, slopes, "radar", FILTER)
    members = sorted(i for pair in pairs for i in pair)
    assert pairs and len(set(members)) == 2*len(pairs) and pairs[-1][1] == members[-1]
    epochs = np.asarray([(events[i].receive_s, events[j].receive_s) for i,j in pairs], dtype="<f8")
    schedule = dict(n_selected=len(events), n_pairs=len(pairs), n_members=len(members),
                    prior_receive_s=events[members[0]].receive_s,
                    terminal_receive_s=events[members[-1]].receive_s,
                    pair_schedule_sha256=hashlib.sha256(epochs.tobytes()).hexdigest())
    return events, pairs, slopes, schedule


def augmented(events, pairs, direction, schedule, reference, model, prior_sd, *, axis=(1.,0.), robust=False):
    ekf = DirectionCenterPairedEKF(FILTER["sensors"], FILTER["filter"], reference, model,
                                   center_std_m=prior_sd,axis=axis,robust=robust)
    ekf.predict_to(schedule["prior_receive_s"])
    failed = 0
    last = None
    for i,j in pairs:
        try:
            last = ekf.step_pair(events[i], events[j], FILTER["sensors"]["radar"]["geometry_sigma"],
                                 direction[i], direction[j])
        except UnsupportedReference:
            failed += 1
    assert abs(ekf.last_receive_s-schedule["terminal_receive_s"]) < 1e-8
    if failed or last is None:
        return dict(augmented_ms=None, center_m=None, drift_sps=None, offset_std_ms=None,
                    center_std_m=None, offset_center_corr=None, offset_center_cov_s_m=None,
                    augmented_failures=failed)
    p = np.asarray(ekf.P)
    assert np.isfinite(p).all() and np.min(np.linalg.eigvalsh((p+p.T)/2)) >= -1e-12
    return dict(augmented_ms=1000*last["offset_s"], center_m=last["center_m"],
                drift_sps=last["drift_sps"],offset_std_ms=1000*last["offset_std_s"],
                center_std_m=last["center_std_m"],offset_center_corr=last["offset_center_corr"],
                offset_center_cov_s_m=last["offset_center_cov_s_m"],augmented_failures=0)


def initialize():
    global NOMINAL, CONTROLS
    NOMINAL = {(int(r["seed"]),r["motion"],int(r["laps"]),r["observation"],int(r["truth_ms"])):r
               for r in load_csv(PREVIOUS/"trials.csv")}
    CONTROLS = {(int(r["seed"]),r["motion"],int(r["truth_ms"]),r["arm"]):r
                for r in load_csv(PREVIOUS/"analysis"/"ekf_bias_ablation.csv")}


def run(task):
    seed, motion = task
    output = []
    model = VehicleObservationModel(GEOMETRY)
    with np.load(PREVIOUS/"observations"/f"{motion}_{seed:02}.npz",allow_pickle=False) as data:
        for laps in (1,8):
            reference = make_reference(*data[f"lidar_{laps}"].T)
            for arm in OBSERVATIONS:
                stream = data[f"{arm}_{laps}"]
                experiment = SCENARIOS if laps==8 and arm=="complete" else ("direction",)
                for truth in TRUTHS:
                    nominal = NOMINAL[seed,motion,laps,arm,truth]
                    baseline_schedule = None
                    for scenario in experiment:
                        events,pairs,slopes,schedule = prepare(stream,truth,scenario,reference,model)
                        if baseline_schedule is None:
                            baseline_schedule = schedule
                            info = paired_information(events, pairs, slopes, FILTER)
                        assert schedule == baseline_schedule
                        assert int(nominal["n_pairs"]) == len(pairs)
                        if laps==8 and arm=="complete":
                            archived = CONTROLS[seed,motion,truth,scenario]
                            assert archived["pair_schedule_sha256"] == schedule["pair_schedule_sha256"]
                            for name in ("prior_receive_s","terminal_receive_s"):
                                assert abs(float(archived[name])-schedule[name]) < 1e-9
                            original_paired,original_raw = number(archived,"paired_ms"),number(archived,"framewise_ms")
                            old_failed = int(archived["paired_failures"])
                        else:
                            original_paired = number(nominal,"ekf_paired_ms")
                            original_raw = number(nominal,"ekf_framewise_ms")
                            old_failed = int(nominal["paired_failures"])
                        q = -np.sign(slopes[:,0])
                        assert all(q[i] in (-1,1) and q[j] in (-1,1) for i,j in pairs)
                        same_leg_pairs = sum(q[i]==q[j] for i,j in pairs)
                        if seed==0:
                            zero = augmented(events,pairs,q,schedule,reference,model,0.)
                            assert zero["augmented_failures"]==old_failed
                            if original_paired is not None:
                                np.testing.assert_allclose(zero["augmented_ms"],original_paired,atol=1e-7,rtol=0)
                        priors = (0.15,0.075,0.30) if laps==8 and arm=="tracked" else (0.15,)
                        expected_c = SCENARIOS[scenario]
                        for prior in priors:
                            result = augmented(events,pairs,q,schedule,reference,model,prior)
                            value,c = result["augmented_ms"],result["center_m"]
                            profile = nominal if scenario=="direction" else {}
                            output.append(dict(seed=seed,motion=motion,laps=laps,observation=arm,truth_ms=truth,
                                scenario=scenario,center_prior_std_m=prior,truth_center_m=expected_c,
                                **schedule,same_leg_pairs=same_leg_pairs,**info,original_paired_ms=original_paired,
                                original_framewise_ms=original_raw,original_paired_failures=old_failed,
                                original_profile_ms=number(profile,"estimate_ms"),original_profile_scale_ms=number(profile,"scale_ms"),
                                original_profile_accepted=profile.get("accepted"),**result,
                                augmented_error_ms=None if value is None else value-truth,
                                center_error_m=None if c is None else c-expected_c))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    tasks=[(seed,motion) for seed in range(30) for motion in MOTIONS]
    temporary=HERE/"trials.partial.csv"
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool, temporary.open("w",encoding="utf-8",newline="") as file:
        count=0
        for batch in pool.map(run,tasks):
            if not count:
                writer=csv.DictWriter(file,fieldnames=batch[0])
                writer.writeheader()
            writer.writerows(batch)
            count+=len(batch)
            file.flush()
            if count % 220 == 0:
                print(f"{count}/1320",flush=True)
    assert count==1320
    temporary.replace(HERE/"trials.csv")


if __name__=="__main__":
    main()
