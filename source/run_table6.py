"""Refit the eight Table 6 profiles and export the manuscript's selected fit flags."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
CORE = ROOT / "outputs/mst_review_20261117/reviewer_replay/bundle/core"
MODES = ("geometry", "direction")
SUPPORTS = {
    "08_near_constant": "08 near-constant",
    "08_variable": "08 variable",
    "09_full": "09 full support",
    "ghost18_right": "G18-C / right",
}
SELECTED_FLAGS = {
    "nuisance_solver_not_converged": ("Solver nonconvergence", "求解器未收敛"),
    "profile_multimodal": ("Multimodality", "多峰"),
    "search_boundary": ("Search boundary", "靠近搜索边界"),
    "temporal_nuisance_aliasing": ("Low retained timing information fraction", "保留时偏信息比例偏低"),
    "model_residual_mismatch": ("Residual mismatch", "残差失配"),
    "outside_local_geometry_model": ("Geometry outside local model", "几何参数超域"),
    "outside_center_bias_model": ("Centre outside local model", "中心偏差超域"),
    "linearization_invalid": ("Invalid projection linearization", "投影线性化失效"),
    "no_dominant_motion_axis": ("No dominant motion axis", "缺少主导运动轴"),
    "insufficient_reversal_support": ("Insufficient reversal support", "反向观测支持不足"),
    "insufficient_split_support": ("Insufficient split support", "子集支持不足"),
    "split_instability": ("Split instability", "子集结果不稳定"),
}


def table_row(group, mode, n, offset_ms, scale_ms, reasons):
    """Project the archived fit reasons onto the checks listed in the manuscript."""
    unknown = set(reasons) - SELECTED_FLAGS.keys() - {"weak_temporal_information"}
    if unknown:
        raise ValueError(f"Unclassified fit reasons: {sorted(unknown)}")
    flags = [reason for reason in reasons if reason in SELECTED_FLAGS]
    if group == "ghost18_right" and mode == "direction":
        flag_en, flag_zh = ("See text", "见正文")
    else:
        flag_en = ", ".join(SELECTED_FLAGS[key][0] for key in flags) or "None"
        flag_zh = "、".join(SELECTED_FLAGS[key][1] for key in flags) or "无"
    return {
        "support": SUPPORTS[group],
        "model": mode.capitalize(),
        "n": n,
        "candidate_offset_ms": f"{offset_ms:.3f}",
        "diagnostic_scale_ms": f"{scale_ms:.3f}",
        "selected_fit_flags": ", ".join(flags) or "None",
        "table6_label_en": flag_en,
        "table6_label_zh": flag_zh,
        "scale_note": "Evaluated at an out-of-domain candidate (†)" if (
            group == "ghost18_right" and mode == "direction"
        ) else "",
    }


def refit():
    sys.path.insert(0, str(CORE))
    import reproduce as archive
    from src.temporal_model.model import Observation
    from src.temporal_model.vehicle import TargetReference

    archive.check_integrity()
    cfg = archive.load(archive.DATA / "filter_config.json")
    expected_groups = archive.load(archive.DATA / "expected.json")["profiles"]
    identity = archive.load(archive.DATA / "event_identity.json")
    matrix = np.array(archive.load(archive.DATA / "radar_from_lidar.json"))
    geometry = (np.eye(4), matrix, np.eye(3), np.zeros(4))
    sigma = cfg["sensors"]["radar"]["sigma"]
    full = {}
    for name in ("08", "09"):
        events, _, reference, _ = archive.load_recording(name, cfg, geometry, identity[name])
        full[name] = archive.profile.ProfileProblem(events, reference, geometry, "radar", sigma)
    masks, _ = archive.speed_groups.choose_groups(full["08"])
    problems = {
        "08_near_constant": archive.speed_groups.subset(full["08"], geometry, masks["near_constant"]),
        "08_variable": archive.speed_groups.subset(full["08"], geometry, masks["variable"]),
        "09_full": full["09"],
    }
    for group, problem in problems.items():
        if problem.ids[problem.mask].tolist() != expected_groups[group]["ids"]:
            raise ValueError(f"Changed observation support in {group}")

    fixture = archive.load(ROOT / "inputs/table6_ghost18.json")
    yaw = -np.pi / 6
    c, s = np.cos(yaw), np.sin(yaw)
    common_from_radar = np.eye(4)
    common_from_radar[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    common_from_radar[:3, 3] = [3.739, -.658, .0305]
    geometry = (np.eye(4), np.linalg.inv(common_from_radar), np.eye(3), np.zeros(4))
    reference = TargetReference(
        [dict(zip(("t", "x", "y", "z"), (time, *xyz)))
         for time, xyz in zip(fixture["lidar_t_s"], fixture["lidar_xyz_m"], strict=True)],
        cfg["reference_gap_s"], cfg["reference_smooth_half_window_s"],
    )
    events = [
        Observation(i, "radar", reported, received, np.asarray(xy))
        for i, reported, received, xy in zip(
            fixture["event_ids"], fixture["reported_s"], fixture["receive_s"],
            fixture["radar_common_xy_m"], strict=True,
        )
    ]
    problems["ghost18_right"] = archive.profile.ProfileProblem(
        events, reference, geometry, "radar", sigma, support=fixture["support"],
    )
    rows = []
    for group, problem in problems.items():
        for mode in MODES:
            fit = problem.fit(mode)
            scale = fit.get("block_inflated_offset_std_s")
            if fit.get("offset_s") is None or scale is None:
                raise ValueError(f"Missing finite profile result for {group} / {mode}")
            rows.append(table_row(group, mode, int(problem.mask.sum()), 1000 * fit["offset_s"],
                                  1000 * scale, fit["reasons"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = refit()
    with (ROOT / "expected/table6.csv").open(encoding="utf-8", newline="") as source:
        expected = list(csv.DictReader(source))
    if len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} profile rows, received {len(rows)}")
    for actual, archived in zip(rows, expected, strict=True):
        if {key: str(value) for key, value in actual.items()} != archived:
            raise ValueError(f"Table 6 row mismatch. Expected {archived!r}, received {actual!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified eight Table 6 rows at manuscript precision: {args.output}", flush=True)


if __name__ == "__main__":
    main()
