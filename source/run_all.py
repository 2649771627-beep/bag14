"""Regenerate and verify results without a project checkout or ROS."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS = ROOT / "generated"
RAW_FIELD = OUTPUTS / "airy_ars408_vehicle_r01"
SIMULATION = OUTPUTS / "airy_ars408_sim"
OLDER_REPLAY = OUTPUTS / "mst_review_20261117/reviewer_replay"
AUGMENTED = OUTPUTS / "mst_q2_20261118/replay_extension"
CONTROLS = OUTPUTS / "mst_airy_submission"


def verify_inputs():
    ledger = (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8")
    for line in ledger.splitlines():
        expected, relative = line.split("  ", 1)
        if sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")


def execute(script, *args):
    print(f"Running {script.name}", flush=True)
    subprocess.run([sys.executable, "-B", str(script), *(str(arg) for arg in args)],
                   cwd=ROOT, check=True)


def rerun_simulation(trials_path, workers, reuse):
    sys.path.insert(0, str(SIMULATION))
    import numpy as np
    from summarize import load_trials, table_rows, write_rows

    trials_path.parent.mkdir(parents=True, exist_ok=True)
    if reuse:
        shutil.copyfile(ROOT / "expected/airy_ars408_sim/trials.csv", trials_path)
    else:
        import run_airy_sim

        run_airy_sim.HERE = trials_path.parent
        tasks = [(seed, motion) for motion in run_airy_sim.MOTIONS for seed in run_airy_sim.SEEDS]
        run_airy_sim.write_trials(tasks, workers)
    summary, comparisons = table_rows(load_trials(trials_path), np)
    write_rows(trials_path.with_name("summary.csv"), summary)
    write_rows(trials_path.with_name("paired_comparisons.csv"), comparisons)


def rerun_legacy():
    legacy = RESULTS / "legacy"
    shutil.copytree(OLDER_REPLAY, legacy, dirs_exist_ok=True)
    execute(legacy / "run_replay.py")
    execute(legacy / "simulation/run_known_truth.py")
    execute(legacy / "simulation/verify_terminal_affine.py")
    execute(legacy / "verify.py")
    execute(legacy / "bundle/run.py", "--check-only")
    return legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--reuse-simulation", action="store_true")
    parser.add_argument("--skip-legacy", action="store_true")
    replay_mode = parser.add_mutually_exclusive_group()
    replay_mode.add_argument("--point-timing-only", action="store_true",
                             help="Regenerate Table 15, Figures 11–12, and Supplementary Analysis S3")
    replay_mode.add_argument("--table6-only", action="store_true",
                             help="Refit the eight Table 6 profiles and check the selected flags")
    parser.add_argument("--cjk-font", type=Path,
                        help="Also render Chinese charts with the supplied local font file")
    options = parser.parse_args()
    if not 1 <= options.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    verify_inputs()
    RESULTS.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS / ".mplconfig"))
    os.environ["PYTHONNOUSERSITE"] = "1"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"

    table6 = RESULTS / "table6.csv"
    if options.table6_only:
        execute(ROOT / "run_table6.py", "--output", table6)
        return

    field = RESULTS / "field"
    tracks = field / "tracks"
    trials = RESULTS / "simulation" / "trials.csv"
    execute(RAW_FIELD / "prepare.py", "--raw", RAW_FIELD / "raw", "--out", tracks)
    point = RESULTS / "point_timing"
    font_option = ["--cjk-font", options.cjk_font.resolve()] if options.cjk_font else []
    execute(ROOT / "run_point_timing.py", "--tracks", tracks, "--output", point, *font_option)
    if options.point_timing_only:
        return
    execute(ROOT / "run_table6.py", "--output", table6)
    public = RESULTS / "public_evidence"
    execute(ROOT / "run_public_evidence.py", "--output", public)
    execute(RAW_FIELD / "profile_compare.py", "--tracks", tracks, "--out", field)
    execute(RAW_FIELD / "replay.py", "--tracks", tracks, "--output", field)
    execute(CONTROLS / "field_stability/run_field_stability.py", "--recordings", RAW_FIELD,
            "--out", RESULTS / "field_stability")
    execute(CONTROLS / "factorial/run_factorial.py", "--field", RAW_FIELD,
            "--output", RESULTS / "factorial")
    rerun_simulation(trials, options.workers, options.reuse_simulation)

    execute(CONTROLS / "diagnostic_supplement/run_diagnostic_supplement.py",
            "--field", RAW_FIELD, "--tracks", tracks, "--pairs", field / "pairs.csv",
            "--replay-trials", field / "replay_trials.csv",
            "--factorial", RESULTS / "factorial/factorial_trials.csv",
            "--simulation", trials, "--output", RESULTS / "diagnostic_supplement")

    legacy = OLDER_REPLAY if options.skip_legacy else rerun_legacy()
    augmented = RESULTS / "augmented"
    shutil.copytree(AUGMENTED, augmented, dirs_exist_ok=True)
    execute(augmented / "compare_augmented_baseline.py", "--reviewer-replay", legacy)
    execute(ROOT / "render_figures.py", "--field", field, "--simulation", trials,
            "--augmented", augmented / "augmented_baseline_results.json",
            "--figures", RESULTS / "figures")
    execute(ROOT / "verify_results.py")
    verification = RESULTS / "verification.json"
    result = json.loads(verification.read_text(encoding="utf-8"))
    result["simulated_trials_regenerated"] = not options.reuse_simulation
    result["legacy_controls_regenerated"] = not options.skip_legacy
    point_report = json.loads((point / "point_timing_verification.json").read_text(encoding="utf-8"))
    result["point_timing"] = point_report
    public_report = json.loads((public / "verification.json").read_text(encoding="utf-8"))
    result["public_evidence"] = public_report
    result["table6"] = {"rows": 8, "verified_csv": "table6.csv"}
    result["figures_rendered"].extend("public_evidence/" + name for name in public_report["figures_rendered"])
    result["numeric_checks"].update({"point_timing/" + key: value
                                      for key, value in point_report["numeric_checks"].items()})
    result["figures_rendered"].extend("point_timing/figures/" + name for name in point_report["figures_rendered"])
    verification.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
