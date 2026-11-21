"""Build a common-epoch, planar-position replay from the two raw R01 exports."""

import argparse
import csv
from dataclasses import dataclass
from itertools import groupby
import json
import math
from pathlib import Path

import numpy as np


NS = 1_000_000_000
AIRY_ORIGIN_XY = np.array([-0.05, -0.46])
NEAR_M, FAR_M = 2.55, 5.50
REFERENCE_PAD_S = 1.25
CLIP_EDGE_PAD_S = 0.65
MAX_LATERAL_M = 0.48
MAX_LONGITUDINAL_MISMATCH_M = 0.35
MAX_REFERENCE_GAP_S = 0.45
ROOT = Path(__file__).resolve().parent


@dataclass
class LidarTrack:
    stamps: np.ndarray
    receipts: np.ndarray
    positions: np.ndarray
    point_dts: np.ndarray
    n_points: np.ndarray


@dataclass
class RadarFrame:
    stamp: int
    receipt: int
    candidates: list


def read_lidar(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda row: int(row["receive_ns"]))
    stamps = np.array([int(r["stamp_ns"]) for r in rows], dtype=np.int64)
    receipts = np.array([int(r["receive_ns"]) for r in rows], dtype=np.int64)
    positions = np.array([[float(r[key]) for key in ("x_m", "y_m", "z_m")] for r in rows])
    point_dts = np.array([float(r["point_dt_s"]) if r["point_dt_s"] else np.nan for r in rows])
    n_points = np.array([int(r["n_points"]) if r["n_points"] else 0 for r in rows])
    if len(stamps) < 5 or np.any(np.diff(receipts) <= 0) or not np.all(np.isfinite(positions)):
        raise ValueError(f"Insufficient or invalid LiDAR samples in {path}")
    return LidarTrack(stamps, receipts, positions, point_dts, n_points)


def read_radar(path):
    def identity(row):
        return int(row["receive_ns"]), int(row["stamp_ns"])

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=identity)
    frames = []
    for (receipt, stamp), rows_at_frame in groupby(rows, key=identity):
        candidates = []
        for row in rows_at_frame:
            if row["x_m"] and row["y_m"]:
                x, y = float(row["x_m"]), float(row["y_m"])
                rcs = float(row["rcs"]) if row["rcs"] else -math.inf
                if math.isfinite(x) and math.isfinite(y):
                    candidates.append((x, y, row["object_id"], rcs))
        frames.append(RadarFrame(stamp, receipt, candidates))
    if not frames or any(a.receipt >= b.receipt for a, b in zip(frames, frames[1:])):
        raise ValueError(f"Missing or out-of-order radar frames in {path}")
    return frames


def quantiles(values):
    return {f"p{q:02d}": float(np.percentile(values, q)) for q in (5, 50, 95)}


def clock_mapping(lidars, radars):
    """Use one unit-rate representation and describe its recorder-time residuals."""
    first = min(min(int(lidars[k].receipts[0]), radars[k][0].receipt) for k in lidars)
    origin = (first // NS) * NS
    x = np.concatenate([track.stamps for track in lidars.values()])
    d = np.concatenate([track.receipts - track.stamps for track in lidars.values()])
    ref = int(d[0])
    offset = ref + int(round(np.percentile(d - ref, 5)))
    median_offset = ref + int(round(np.percentile(d - ref, 50)))
    anchor_ms = (d - offset) / 1e6
    global_drift = float(np.polyfit((x - x[0]) / NS, anchor_ms, 1)[0])
    per_arm = {}
    for arm, lidar in lidars.items():
        radar = radars[arm]
        deltas = lidar.receipts - lidar.stamps - offset
        receipt_s = (lidar.receipts - lidar.receipts[0]) / NS
        periodic_ms = np.diff(lidar.stamps) / 1e6
        radar_deltas = np.array([r.receipt - r.stamp for r in radar])
        per_arm[arm] = {
            "lidar_raw_stamp_first_ns": int(lidar.stamps[0]),
            "lidar_raw_stamp_last_ns": int(lidar.stamps[-1]),
            "lidar_receipt_first_ns": int(lidar.receipts[0]),
            "lidar_receipt_last_ns": int(lidar.receipts[-1]),
            "lidar_mapped_receipt_minus_header_ms": quantiles(deltas / 1e6),
            "lidar_receive_minus_device_trend_ms_per_s": float(
                np.polyfit(receipt_s, deltas / 1e6, 1)[0]
            ),
            "lidar_nonincreasing_stamp_count": int(np.sum(periodic_ms <= 0)),
            "lidar_device_interframe_ms": quantiles(periodic_ms),
            "radar_receipt_minus_original_header_ms": quantiles(radar_deltas / 1e6),
        }
    first_arm, second_arm = sorted(lidars, key=lambda arm: lidars[arm].receipts[0])
    device_gap = (int(lidars[second_arm].stamps[0]) - int(lidars[first_arm].stamps[-1])) / NS
    receive_gap = (int(lidars[second_arm].receipts[0]) - int(lidars[first_arm].receipts[-1])) / NS
    continuous = bool(device_gap > 0 and all(
        not fields["lidar_nonincreasing_stamp_count"] for fields in per_arm.values()))
    if not continuous:
        raise ValueError("AIRY counter is nonmonotonic or resets between recordings")
    return origin, offset, {
        "rule": "pooled 5th percentile of receive minus device time, unit clock rate",
        "origin_host_ns": origin,
        "device_to_host_offset_ns": offset,
        "pooled_median_anchor_difference_ms": (median_offset - offset) / 1e6,
        "pooled_mapped_receipt_minus_header_ms": quantiles(anchor_ms),
        "pooled_receive_minus_device_trend_ms_per_s": global_drift,
        "device_counter_monotone_across_recordings": continuous,
        "header_gap_between_bags_s": device_gap,
        "recorder_gap_between_bags_s": receive_gap,
        "header_minus_recorder_gap_ms": (device_gap - receive_gap) * 1e3,
        "per_recording": per_arm,
    }


def smooth_range(positions):
    ranges = np.hypot(positions[:, 0], positions[:, 1])
    padded = np.pad(ranges, 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)


def detect_trips(times, ranges):
    """Find complete near–far–near movements, requiring persistent crossings."""
    endpoints = np.select([ranges <= NEAR_M, ranges >= FAR_M], [1, 2])
    stable = np.r_[False, False, (endpoints[2:] == endpoints[1:-1]) &
                                  (endpoints[2:] == endpoints[:-2]) &
                                  (times[2:] - times[:-2] <= 0.55)]
    edges = []
    last_near = None
    for i, endpoint in enumerate(endpoints):
        if not stable[i]:
            continue
        if endpoint == 1:
            last_near = i
        if endpoint and (not edges or edges[-1][0] != endpoint):
            edges.append((int(endpoint), i, last_near))
    trips = []
    for near, far, returned in zip(edges, edges[1:], edges[2:]):
        if (near[0], far[0], returned[0]) == (1, 2, 1):
            start = far[2]
            if start is None or start < near[1]:
                continue
            trips.append({
                "near_entry_s": float(times[near[1]]),
                "near_departure_s": float(times[start]),
                "far_entry_s": float(times[far[1]]),
                "near_return_s": float(times[returned[1]]),
            })
    return trips


def fit_rotation(lidars, ranges):
    """Use one horizontal trajectory direction for both bags and the known +X route."""
    middles = [
        track.positions[(ranges[arm] >= 2.6) & (ranges[arm] <= 5.45), :2]
        for arm, track in lidars.items()
    ]
    xy = np.concatenate(middles)
    if len(xy) < 20:
        raise ValueError("Insufficient traversals to estimate the AIRY motion axis")
    covariance = np.cov(xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    if axis[0] < 0:
        axis = -axis
    rotation = np.array([axis, [-axis[1], axis[0]]])
    return rotation, {
        "lidar_xy_axis": axis.tolist(),
        "theta_lidar_to_vehicle_deg": float(-np.degrees(np.arctan2(axis[1], axis[0]))),
        "axis_pca_fraction": float(eigenvalues[-1] / eigenvalues.sum()),
        "lidar_vehicle_translation_xy_m": AIRY_ORIGIN_XY.tolist(),
        "rule": "pooled AIRY mid-range trajectory PCA, positive local X oriented to vehicle +X",
        "fitted_points": len(xy),
    }


def match_radar(frames, lidar_receipt_s, lidar_xy, origin):
    rows = []
    track_t = (np.array([f.receipt for f in frames]) - origin) / NS
    upper = np.searchsorted(lidar_receipt_s, track_t, side="left")
    support_xy = np.column_stack([
        np.interp(track_t, lidar_receipt_s, lidar_xy[:, d]) for d in (0, 1)
    ])
    for frame, t, i, xy in zip(frames, track_t, upper, support_xy):
        chosen = None
        bracket = 0 < i < len(lidar_receipt_s)
        if bracket and lidar_receipt_s[i] - lidar_receipt_s[i - 1] <= MAX_REFERENCE_GAP_S:
            candidates = [
                c for c in frame.candidates
                if abs(c[1]) <= MAX_LATERAL_M and abs(c[0] - xy[0]) <= MAX_LONGITUDINAL_MISMATCH_M
            ]
            if candidates:
                chosen = max(candidates, key=lambda c: (c[3], -abs(c[0] - xy[0]), -abs(c[1])))
        rows.append((float(t), frame, chosen))
    return rows


def write_csv(path, header, rows):
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def arm_outputs(out, label, lidar, lidar_xy, radar, trips, origin, offset):
    begin = trips[0]["near_departure_s"] - CLIP_EDGE_PAD_S
    end = trips[-1]["near_return_s"] + CLIP_EDGE_PAD_S
    reported_t = (lidar.stamps + (offset - origin)) / NS
    receive_t = (lidar.receipts - origin) / NS
    reference = (receive_t >= begin - REFERENCE_PAD_S) & (receive_t <= end + REFERENCE_PAD_S)
    use = np.flatnonzero(reference)
    radar_in_clip = [(t, frame, c) for t, frame, c in radar if begin <= t <= end]
    valid = [(t, frame, c) for t, frame, c in radar_in_clip if c is not None]
    write_csv(out / f"{label}_lidar.csv",
              ["t_s", "x_m", "y_m", "raw_stamp_ns", "raw_receive_ns",
               "receive_s", "z_lidar_m", "point_dt_s", "n_points"],
              ((reported_t[i], *lidar_xy[i], lidar.stamps[i], lidar.receipts[i],
                receive_t[i], lidar.positions[i, 2],
                lidar.point_dts[i] if np.isfinite(lidar.point_dts[i]) else "", lidar.n_points[i])
               for i in use))
    write_csv(out / f"{label}_radar.csv",
              ["reported_s", "receive_s", "x_m", "y_m", "raw_stamp_ns", "raw_receive_ns",
               "object_id", "rcs", "n_candidates"],
              (((frame.stamp - origin) / NS, t, x, y, frame.stamp, frame.receipt,
                object_id, rcs if math.isfinite(rcs) else "", len(frame.candidates))
               for t, frame, (x, y, object_id, rcs) in valid))
    write_csv(out / f"{label}_radar_coverage.csv",
              ["receive_s", "raw_stamp_ns", "raw_receive_ns", "n_candidates", "matched"],
              ((t, frame.stamp, frame.receipt, len(frame.candidates), int(c is not None))
               for t, frame, c in radar_in_clip))
    retained = np.array([t for t, _, _ in valid])
    if len(use) == 0 or len(retained) == 0:
        raise ValueError(f"No overlapping LiDAR/radar observations in {label}")
    return {
        "radar_clip_host_s": [begin, end],
        "lidar_reference_extent_s": [float(reported_t[use[0]]), float(reported_t[use[-1]])],
        "lidar_reference_frames": len(use),
        "radar_native_frames_in_clip": len(radar_in_clip),
        "radar_associated_frames": len(valid),
        "radar_coverage": len(valid) / len(radar_in_clip),
        "radar_largest_inter_association_gap_s": float(
            np.max(np.diff(np.r_[begin, retained, end]))
        ),
        "trip_count": len(trips),
        "trips": trips,
    }


def prepare(raw, out):
    names = ("near_constant", "variable")
    lidars = {name: read_lidar(raw / f"{name}_lidar.csv") for name in names}
    radars = {name: read_radar(raw / f"{name}_radar.csv") for name in names}
    origin, offset, clock = clock_mapping(lidars, radars)
    ranges = {name: smooth_range(track.positions) for name, track in lidars.items()}
    rot, geometry = fit_rotation(lidars, ranges)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "clock_mapping": clock,
        "geometry": geometry,
        "trip_gate_m": {"near": NEAR_M, "far": FAR_M},
        "radar_association": {
            "type": "retrospective strongest-RCS target in geometry corridor via recorder times",
            "max_lateral_m": MAX_LATERAL_M,
            "max_longitudinal_mismatch_m": MAX_LONGITUDINAL_MISMATCH_M,
            "max_lidar_receipt_bracket_s": MAX_REFERENCE_GAP_S,
        },
        "clip_edge_pad_s": CLIP_EDGE_PAD_S,
        "lidar_reference_pad_s": REFERENCE_PAD_S,
        "clips": {},
        "all_detected_trips": {},
    }
    for name in names:
        lidar = lidars[name]
        lidar_xy = lidar.positions[:, :2] @ rot.T + AIRY_ORIGIN_XY
        lidar_receipt_s = (lidar.receipts - origin) / NS
        trips = detect_trips(lidar_receipt_s, ranges[name])
        expected_trips = 7 if name == "near_constant" else 10
        if len(trips) != expected_trips:
            raise ValueError(
                f"Expected {expected_trips} complete traversals from prior field QC in {name}, "
                f"found {len(trips)}. Candidate windows: {trips}"
            )
        summary["all_detected_trips"][name] = trips
        radar = match_radar(radars[name], lidar_receipt_s, lidar_xy, origin)
        groups = {name: trips} if name == "near_constant" else {name: trips[:8], "variable_fast": trips[8:]}
        for label, selection in groups.items():
            summary["clips"][label] = arm_outputs(
                out, label, lidar, lidar_xy, radar, selection, origin, offset
            )
    (out / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, entry in summary["clips"].items():
        print(f"{name}: {entry['trip_count']} trips, {entry['lidar_reference_frames']} LiDAR "
              f"reference frames, {entry['radar_associated_frames']}/{entry['radar_native_frames_in_clip']} radar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "tracks")
    args = parser.parse_args()
    prepare(args.raw, args.out)
