"""Independent scalar validation of direction-center temporal ambiguity.

This is a conditional linear Gaussian model experiment, not a bag replay,
production-estimator benchmark, novel theorem, or hardware timing validation.
The observation model is y_i = a + c*q_i - v_i*b + epsilon_i.
Directions and velocities are assumed known exactly. No filter or profile
implementation is imported. The constant-offset truth is known by construction.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG = {
    "seed": 2026111407,
    "trials_per_motion": 1000,
    "samples_per_motion": 120,
    "velocity_rms_mps": 0.8,
    "true_offset_s": 0.030,
    "true_common_center_m": 0.050,
    "true_direction_center_m": 0.080,
    "noise_std_m": 0.030,
    "rank_relative_tolerance": 1e-10,
    "maximum_model_conditional_offset_std_s": 0.020,
    "absolute_error_reporting_threshold_s": 0.020,
    "gate_note": "20 ms is a declared engineering resolution criterion, not a fitted threshold or empirical coverage claim.",
    "truth_model": "y=a+c*q-v*b+Gaussian_noise; a,c constant; b constant; q and v known",
    "interpretation": "Independent scalar identifiability and known-truth validation only. No recorded observations or innovation-2 uncertainty model.",
}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def motion_cases() -> dict[str, dict[str, np.ndarray | str]]:
    n = CONFIG["samples_per_motion"]
    q_reverse = np.r_[np.ones(n // 2), -np.ones(n // 2)]
    varying_group_speed = np.linspace(0.2, 1.2, n // 2)
    cases = {
        "equal_speed_reversal": {
            "q": q_reverse.copy(), "v": 0.8 * q_reverse,
            "description": "Both directions have equal and constant speed magnitude.",
        },
        "unequal_direction_speeds_each_constant": {
            "q": q_reverse.copy(), "v": np.where(q_reverse > 0, 0.5, -1.1),
            "description": "Forward and reverse speed magnitudes differ, but each direction remains constant.",
        },
        "within_direction_variable_speed_reversal": {
            "q": q_reverse.copy(),
            "v": np.r_[varying_group_speed, -varying_group_speed[::-1]],
            "description": "Speed magnitude varies within each direction, with no zero-speed samples.",
        },
        "single_direction_variable_speed": {
            "q": np.ones(n), "v": np.linspace(0.2, 1.2, n),
            "description": "One direction with varying speed; a and c are redundant, but b can remain identifiable.",
        },
    }
    for case in cases.values():
        v = np.asarray(case["v"])
        case["v"] = v * CONFIG["velocity_rms_mps"] / np.sqrt(np.mean(v * v))
    return cases


def projector(B: np.ndarray) -> tuple[np.ndarray, int]:
    U, singular, _ = np.linalg.svd(B, full_matrices=False)
    threshold = CONFIG["rank_relative_tolerance"] * singular[0] if singular.size else 0.0
    rank = int(np.sum(singular > threshold))
    return U[:, :rank], rank


def conditional_design(v: np.ndarray, q: np.ndarray, direction_aware: bool) -> dict:
    sigma = CONFIG["noise_std_m"]
    nuisance = np.column_stack([np.ones(len(v)), q]) if direction_aware else np.ones((len(v), 1))
    Q, nuisance_rank = projector(nuisance / sigma)
    h = -v / sigma
    hp = h - Q @ (Q.T @ h)
    residual_norm = float(np.linalg.norm(hp))
    identifiable = residual_norm > CONFIG["rank_relative_tolerance"] * max(1.0, float(np.linalg.norm(h)))
    information = float(hp @ hp) if identifiable else 0.0
    std = 1.0 / np.sqrt(information) if identifiable else None
    accepted = bool(identifiable and std <= CONFIG["maximum_model_conditional_offset_std_s"])
    return {
        "hp": hp,
        "nuisance_rank": nuisance_rank,
        "identifiable": bool(identifiable),
        "information_s_minus2": information,
        "conditional_noise_std_s": None if std is None else float(std),
        "accepted": accepted,
    }


def run() -> dict:
    # Settings are written before designs are evaluated or random noise is drawn.
    save_json(HERE / "independent_theory_config.json", CONFIG)
    cases = motion_cases()
    streams = np.random.SeedSequence(CONFIG["seed"]).spawn(len(cases))
    n_trials = CONFIG["trials_per_motion"]
    b_true = CONFIG["true_offset_s"]
    sigma = CONFIG["noise_std_m"]
    output = {
        "config": CONFIG,
        "metrics": {
            "false_acceptance": "Acceptance when b is unidentifiable under the full truth nuisance space span{1,q}; not an empirical real-data error probability.",
            "absolute_error": "b_estimate minus known b_true, evaluated only for accepted estimates; rejected rank-deficient problems have no point estimate.",
            "large_error_acceptance": "Accepted estimate with absolute error over the separately declared 20 ms reporting threshold; distinct from structural false acceptance.",
            "conditional_noise_std": "Exact noise-only standard deviation for this fixed-design linear Gaussian model; it excludes omitted-center bias.",
            "coverage": "Number of accepted trials divided by all trials; design-based acceptance is deterministic here.",
        },
        "cases": {},
    }
    rows = []
    for (name, case), stream in zip(cases.items(), streams):
        v, q = np.asarray(case["v"]), np.asarray(case["q"])
        full = conditional_design(v, q, True)
        analytic_information = float(sum(np.sum((v[q == g] - np.mean(v[q == g])) ** 2) for g in np.unique(q)) / sigma**2)
        np.testing.assert_allclose(full["information_s_minus2"], analytic_information, atol=1e-9, rtol=1e-12)
        np.testing.assert_allclose(np.mean(v * v), CONFIG["velocity_rms_mps"] ** 2, atol=1e-14)
        rng = np.random.default_rng(stream)
        mean_y = CONFIG["true_common_center_m"] + CONFIG["true_direction_center_m"] * q - v * b_true
        y = mean_y[None, :] + rng.normal(0.0, sigma, (n_trials, len(v)))
        result = {
            "description": case["description"],
            "direction_counts": {str(int(g)): int(np.sum(q == g)) for g in np.unique(q)},
            "velocity_rms_mps": float(np.sqrt(np.mean(v * v))),
            "direction_mean_velocity_mps": {str(int(g)): float(np.mean(v[q == g])) for g in np.unique(q)},
            "within_direction_velocity_energy": float(analytic_information * sigma**2),
            "truth_model_temporally_identifiable": full["identifiable"],
            "truth_nuisance_rank": full["nuisance_rank"],
            "analytic_information_s_minus2": analytic_information,
            "methods": {},
        }
        for method, aware in [("common_bias_only", False), ("common_plus_direction_bias", True)]:
            design = conditional_design(v, q, aware)
            accepted = design["accepted"]
            n_accepted = n_trials if accepted else 0
            info = design["information_s_minus2"]
            if accepted:
                estimates = (y / sigma) @ design["hp"] / info
                errors = estimates - b_true
                deterministic_bias = float((mean_y / sigma) @ design["hp"] / info - b_true)
                rmse = float(np.sqrt(np.mean(errors * errors)))
                bias = float(np.mean(errors))
                mae = float(np.mean(np.abs(errors)))
                sampling_sd = float(np.std(errors, ddof=1))
                theoretical_rmse = float(np.sqrt(deterministic_bias**2 + design["conditional_noise_std_s"]**2))
                large_error_count = int(np.sum(np.abs(errors) > CONFIG["absolute_error_reporting_threshold_s"]))
            else:
                estimates = np.full(n_trials, np.nan)
                rmse = bias = mae = sampling_sd = deterministic_bias = theoretical_rmse = None
                large_error_count = 0
            false_accepted = n_accepted if not full["identifiable"] else 0
            result["methods"][method] = {
                "assumed_nuisance_rank": design["nuisance_rank"],
                "assumed_model_identifiable": design["identifiable"],
                "conditional_noise_std_s": design["conditional_noise_std_s"],
                "accepted_trials": n_accepted,
                "rejected_trials": n_trials - n_accepted,
                "acceptance_fraction": n_accepted / n_trials,
                "false_acceptance_trials": false_accepted,
                "false_acceptance_fraction": false_accepted / n_trials,
                "accepted_absolute_error_over_20ms_trials": large_error_count,
                "accepted_absolute_error_over_20ms_fraction_of_all_trials": large_error_count / n_trials,
                "absolute_rmse_s_on_accepted": rmse,
                "absolute_bias_s_on_accepted": bias,
                "absolute_mae_s_on_accepted": mae,
                "empirical_noise_sd_s_on_accepted": sampling_sd,
                "deterministic_omitted_nuisance_bias_s": deterministic_bias,
                "theoretical_absolute_rmse_s": theoretical_rmse,
            }
            for trial, estimate in enumerate(estimates):
                rows.append({
                    "motion": name, "method": method, "trial": trial,
                    "truth_offset_s": b_true, "accepted": accepted,
                    "truth_model_identifiable": full["identifiable"],
                    "false_acceptance": bool(accepted and not full["identifiable"]),
                    "estimated_offset_s": float(estimate) if accepted else "",
                    "absolute_error_s": float(estimate - b_true) if accepted else "",
                })
        output["cases"][name] = result
    assert not output["cases"]["equal_speed_reversal"]["truth_model_temporally_identifiable"]
    assert not output["cases"]["unequal_direction_speeds_each_constant"]["truth_model_temporally_identifiable"]
    assert output["cases"]["within_direction_variable_speed_reversal"]["truth_model_temporally_identifiable"]
    assert output["cases"]["single_direction_variable_speed"]["truth_model_temporally_identifiable"]
    save_json(HERE / "independent_theory_results.json", output)
    with (HERE / "independent_theory_trials.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "Independent scalar direction-center validation; 1000 fixed-seed trials per motion.",
        "Known absolute truth b=30 ms; shared bias a=50 mm; direction bias c=80 mm; noise sigma=30 mm.",
        "All motion groups have velocity RMS 0.8 m/s. No bag measurements or hardware truth are used.",
        "Acceptance is a deterministic identifiability/resolution check under the assumed model, not an estimated real-world reliability probability.",
        "The single-direction group is intentionally allowed by the algebra; a separate production requirement for reversal is an application policy, not mathematical necessity.",
        "",
    ]
    for name, case in output["cases"].items():
        lines.append(name)
        lines.append(f"  full-model identifiable={case['truth_model_temporally_identifiable']}; nuisance rank={case['truth_nuisance_rank']}; I_b={case['analytic_information_s_minus2']:.6f} s^-2")
        for method, metric in case["methods"].items():
            rmse = metric["absolute_rmse_s_on_accepted"]
            text_rmse = "unavailable (rejected)" if rmse is None else f"{1000*rmse:.3f} ms"
            lines.append(f"  {method}: accepted={metric['accepted_trials']}/1000; structurally false accepted={metric['false_acceptance_trials']}/1000; accepted error>20ms={metric['accepted_absolute_error_over_20ms_trials']}/1000; absolute RMSE={text_rmse}")
    lines.extend(["", "No novelty claim is made for this linear projection result. The experiment validates only the stated finite-dimensional model."])
    (HERE / "independent_theory_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return output


if __name__ == "__main__":
    run()
