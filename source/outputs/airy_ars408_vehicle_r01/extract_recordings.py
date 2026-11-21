"""Extract high-rate target observations and preserve both clock stamps in three ROS 2 bags."""

import argparse
import csv
import json
from pathlib import Path
import sqlite3

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensing_msgs.msg import RadarFrame


LIDAR_TOPIC = "/sensor/lidar1/points"
RADAR_TOPIC = "/radar0/radar_frame"
LOW = np.array([1.45, -0.40, -1.05])
HIGH = np.array([6.60, 1.60, 0.85])
VOXEL_M = 0.05
GRID_SIZE = tuple(np.ceil((HIGH - LOW) / VOXEL_M).astype(int))
FOREGROUND_EDGES = np.arange(LOW[0], HIGH[0] + 0.04, 0.04)


def database(bag):
    databases = sorted(bag.glob("*.db3"))
    if len(databases) != 1:
        raise ValueError(f"Expected one SQLite DB3 under {bag}, found {len(databases)}.")
    return sqlite3.connect(databases[0].resolve().as_uri() + "?mode=ro", uri=True)


def messages(db, topic, until_ns=None):
    topic_id = db.execute("select id from topics where name=?", (topic,)).fetchone()[0]
    where = "topic_id=?"
    params = (topic_id,)
    if until_ns is not None:
        where += " and timestamp<?"
        params = (topic_id, until_ns)
    return db.execute(f"select timestamp,data from messages where {where} order by timestamp", params)


def stamp_ns(msg):
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def points_in_roi(msg):
    fields = {field.name: field.offset for field in msg.fields if field.datatype == 7}
    endian = ">f4" if msg.is_bigendian else "<f4"
    names = ["x", "y", "z"]
    if "time" in fields:
        names.append("time")
    points = np.column_stack([
        np.ndarray((msg.height, msg.width), dtype=endian, buffer=msg.data,
                   offset=fields[name], strides=(msg.row_step, msg.point_step)).ravel()
        for name in names
    ])
    return points[np.all((points[:, :3] >= LOW) & (points[:, :3] < HIGH), axis=1)]


def voxel_ids(xyz):
    coordinates = np.floor((xyz - LOW) / VOXEL_M).astype(int)
    return np.ravel_multi_index(coordinates.T, GRID_SIZE)


def empty_scene(precheck, seconds=4.0):
    seen = np.zeros(np.prod(GRID_SIZE), dtype=np.int32)
    count = 0
    with database(precheck) as db:
        start = db.execute("select min(timestamp) from messages").fetchone()[0]
        for _, raw in messages(db, LIDAR_TOPIC, start + round(seconds * 1e9)):
            xyz = points_in_roi(deserialize_message(raw, PointCloud2))[:, :3]
            if len(xyz):
                seen[np.unique(voxel_ids(xyz))] += 1
            count += 1
    if count < 20:
        raise ValueError("The precheck does not contain sufficient empty-scene point clouds.")
    return seen >= 4, count


def board_cluster(points, background):
    if len(points) < 35:
        return None
    foreground = points[~background[voxel_ids(points[:, :3])]]
    if len(foreground) < 35:
        return None
    histogram = np.histogram(foreground[:, 0], FOREGROUND_EDGES)[0]
    peak = int(np.argmax(np.convolve(histogram, [1, 2, 3, 2, 1], mode="same")))
    x_peak = (FOREGROUND_EDGES[peak] + FOREGROUND_EDGES[peak + 1]) / 2
    board = foreground[np.abs(foreground[:, 0] - x_peak) < 0.13]
    if len(board) < 35:
        return None
    edges = np.quantile(board[:, 1:3], [0.04, 0.96], axis=0)
    if np.min(edges[1] - edges[0]) < 0.18:
        return None
    y, z = np.mean(edges, axis=0)
    x = float(np.median(board[:, 0]))
    dt = None
    if board.shape[1] == 4:
        native = board[:, 3]
        valid = native[np.isfinite(native)]
        dt = float(np.median(valid)) if len(valid) else None
    return x, float(y), float(z), dt, len(board)


def extract(bag, name, out, background):
    info = {"bag": str(bag), "lidar_frames": 0, "board_rows": 0, "radar_frames": 0,
            "empty_corridors": 0, "radar_candidates": 0}
    with database(bag) as db, (out / f"{name}_lidar.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("stamp_ns", "receive_ns", "x_m", "y_m", "z_m", "point_dt_s", "n_points"))
        for receipt, raw in messages(db, LIDAR_TOPIC):
            msg = deserialize_message(raw, PointCloud2)
            info["lidar_frames"] += 1
            cluster = board_cluster(points_in_roi(msg), background)
            if cluster is not None:
                writer.writerow((stamp_ns(msg), receipt, *cluster))
                info["board_rows"] += 1
            if info["lidar_frames"] % 1000 == 0:
                print(name, "decoded lidar", info["lidar_frames"], flush=True)
    with database(bag) as db, (out / f"{name}_radar.csv").open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("stamp_ns", "receive_ns", "x_m", "y_m", "object_id", "rcs", "n_candidates"))
        for receipt, raw in messages(db, RADAR_TOPIC):
            msg = deserialize_message(raw, RadarFrame)
            info["radar_frames"] += 1
            targets = [
                candidate for candidate in msg.targets
                if 1.35 <= candidate.longitudinal_position <= 6.65
                and abs(candidate.lateral_position) <= 0.55
                and np.isfinite(candidate.rcs)
            ]
            if targets:
                writer.writerows(
                    (stamp_ns(msg), receipt, float(candidate.longitudinal_position),
                     float(candidate.lateral_position), int(candidate.id), float(candidate.rcs),
                     len(targets))
                    for candidate in targets
                )
            else:
                writer.writerow((stamp_ns(msg), receipt, None, None, None, None, 0))
                info["empty_corridors"] += 1
            info["radar_candidates"] += len(targets)
    print(name, info, flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--near", type=Path, required=True)
    parser.add_argument("--variable", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    background, count = empty_scene(args.precheck)
    report = {"background_bag": str(args.precheck), "background_header_frames": count,
              "background_seconds": 4.0, "background_voxel_size_m": VOXEL_M,
              "background_voxel_min_frames": 4, "lidar_roi_min_m": LOW.tolist(),
              "lidar_roi_max_m": HIGH.tolist(), "radar_candidate_longitudinal_m": [1.35, 6.65],
              "radar_max_abs_lateral_m": 0.55, "point_dt_column": "native median field time"}
    report["near_constant"] = extract(args.near, "near_constant", args.out, background)
    report["variable"] = extract(args.variable, "variable", args.out, background)
    (args.out / "extraction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("R01_RAW_EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
