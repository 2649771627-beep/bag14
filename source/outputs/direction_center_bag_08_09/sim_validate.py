"""Fresh-seed simulation test for the direction-axis and robust EKF revisions.

With the frozen simulator and eight round trips, seeds 60--89 were designated
before fitting. The two time offsets share stream draws. Nominal angular axis,
adaptive axis only, robust adaptive axis and matched framewise control share
the same observed pairs, prior and comparison terminal. Physical time is used
only to generate sensor stamps and calculate truth errors. The formal profile
gate is recomputed independently on its original selection.
"""
from concurrent.futures import ProcessPoolExecutor
import csv
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "direction_center_ekf_4_10m"
GENERATOR = HERE.parent / "repeated_roundtrips_4_10m"
SEEDS = range(60, 90)
METHODS = ("raw_paired","center_x_paired","axis_paired","robust_axis_paired","robust_axis_framewise")


def generate(task):
    sys.path.insert(0,str(GENERATOR))
    from streams import get_prefix, simulate
    seed,motion = task
    lidar, arms = get_prefix(simulate(seed,motion),8)
    observations = {"tracked": arms["tracked"]}
    if motion == "variable":
        observations["complete"] = arms["complete"]
    return seed,motion,lidar,observations


def filter_output(name, events, pairs, slopes, schedule, reference, model, axis):
    from run import DirectionCenterPairedEKF, FILTER, UnsupportedReference
    from src.ekf_estimator.paired import PairedTemporalEKF
    args = FILTER["sensors"], FILTER["filter"], reference, model
    if name == "raw_paired":
        ekf = PairedTemporalEKF(*args)
    else:
        used_axis = np.asarray((1.,0.)) if name == "center_x_paired" else axis
        ekf = DirectionCenterPairedEKF(*args,center_std_m=.15,axis=used_axis,
                                       robust=name.startswith("robust"))
        sign = -np.sign(slopes @ used_axis)
    ekf.predict_to(schedule["prior_receive_s"])
    weights = []
    failures = 0
    last = None
    geom = FILTER["sensors"]["radar"]["geometry_sigma"]
    if name.endswith("framewise"):
        for i in sorted(i for pair in pairs for i in pair):
            try:
                last = ekf.step(events[i],sign[i])
                weights.append(last["innovation_weight"])
            except UnsupportedReference:
                failures += 1
    else:
        for i,j in pairs:
            try:
                if name == "raw_paired":
                    last = ekf.step_pair(events[i],events[j],geom)
                else:
                    last = ekf.step_pair(events[i],events[j],geom,sign[i],sign[j])
                    weights.append(last["innovation_weight"])
            except UnsupportedReference:
                failures += 1
    assert abs(ekf.last_receive_s-schedule["terminal_receive_s"]) < 1e-9
    available = last is not None and failures == 0
    return {"estimate_ms":1000*last["offset_s"] if available else None,
            "center_m":last.get("center_m") if available else None, "failed_updates":failures,
            "n_downweighted":sum(w < 1.-1e-12 for w in weights), "n_weights":len(weights),
            "min_weight":min(weights) if weights else None}


def evaluate(task):
    sys.path.insert(0,str(PREVIOUS))
    from run import GEOMETRY, VehicleObservationModel, make_reference, prepare
    from center_filter import reference_axis
    from holdout import fingerprint
    from paper_gate import evaluate as profile
    seed,motion,lidar,observations = task
    reference = make_reference(*lidar.T)
    axis, energy = reference_axis(reference)
    model = VehicleObservationModel(GEOMETRY)
    rows = []
    for arm,radar in observations.items():
        digest = fingerprint(lidar,radar)
        for truth in (70,160):
            events,pairs,slopes,schedule = prepare(radar,truth,"direction",reference,model)
            gate = profile(reference,radar[:,:3],truth)
            for name in METHODS:
                values = filter_output(name,events,pairs,slopes,schedule,reference,model,axis)
                value = values["estimate_ms"]
                c = values["center_m"]
                rows.append(dict(seed=seed,motion=motion,observation=arm,laps=8,truth_ms=truth,
                    method=name,observations_sha256=digest,**schedule,axis_x=axis[0],axis_y=axis[1],
                    axis_energy_fraction=energy,**values,abs_error_ms=abs(value-truth) if value is not None else None,
                    center_abs_error_cm=100*abs(c-.08) if c is not None else None,
                    profile_estimate_ms=gate["estimate_ms"],profile_scale_ms=gate["scale_ms"],
                    profile_gate50_pass=gate["gate50_pass"],profile_accepted=gate["accepted"],
                    profile_reasons=gate["reasons"]))
    return rows


def main():
    tasks = [(seed,motion) for seed in SEEDS for motion in ("near_constant","variable")]
    with ProcessPoolExecutor(max_workers=6) as pool:
        streams = list(pool.map(generate,tasks))
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = [r for batch in pool.map(evaluate,streams) for r in batch]
    assert len(results) == len(SEEDS)*3*2*len(METHODS)
    old = set()
    sys.path.insert(0,str(PREVIOUS))
    from holdout import fingerprint
    with (PREVIOUS/"holdout.csv").open(encoding="utf-8") as f:
        old.update(r["observations_sha256"] for r in csv.DictReader(f))
    for path in (GENERATOR/"observations").glob("*.npz"):
        with np.load(path,allow_pickle=False) as a:
            old.add(fingerprint(a["lidar_8"],a["tracked_8"]))
    hashes = {r["observations_sha256"] for r in results}
    assert len(old)==90 and len(hashes)==90 and old.isdisjoint(hashes)
    temp = HERE/"sim_trials.partial.csv"
    with temp.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=results[0])
        writer.writeheader()
        writer.writerows(results)
    temp.replace(HERE/"sim_trials.csv")
    for motion,arm in (("variable","tracked"),("near_constant","tracked"),("variable","complete")):
        for t in (70,160):
            for name in METHODS:
                vals = [r["abs_error_ms"] for r in results if r["motion"]==motion and r["observation"]==arm
                        and r["truth_ms"]==t and r["method"]==name and r["abs_error_ms"] is not None]
                print(motion,arm,t,name,len(vals),np.quantile(vals,[.5,.9]).round(2).tolist())


if __name__=="__main__":
    main()
