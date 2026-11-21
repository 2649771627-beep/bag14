"""Extract target observations from recorded C16 and ARS408 messages."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


_LIDAR_TOPIC = "/sensor/lidar0/points"
_RADAR_TOPIC = "/sensor/radar0/data"
_TF_TOPIC = "/tf_static"
_FIELDS = {7: "f4", 8: "f8"}
_RADAR_TYPES = ("RadarHeader", "RadarTarget", "RadarFrame")


@dataclass(frozen=True)
class ExtractionGates:
    """Geometry and support rules chosen on run 08 and frozen for run 09."""

    lidar_x_m: tuple = (1.6, 4.4)
    lidar_y_m: tuple = (-4.2, 5.3)
    lidar_z_m: tuple = (-1.43, 0.55)
    radar_x_m: tuple = (1.6, 4.4)
    radar_y_m: tuple = (-3.85, 4.2)
    background_initial_s: float = 3.0
    background_fraction: float = 0.45
    lidar_grid_m: float = 0.12
    radar_grid_m: float = 0.2
    lidar_min_points: int = 60
    lidar_min_z_span_m: float = 0.35
    lidar_max_xy_span_m: float = 1.5
    lidar_ambiguity_ratio: float = 0.68
    radar_ambiguity_x_margin_m: float = 0.36
    radar_ambiguity_y_separation_m: float = 0.75
    continuity_window_s: float = 0.37
    continuity_slack_m: float = 0.28
    continuity_speed_limit_mps: float = 2.5


@contextmanager
def open_bag(path):
    """Read a finalized SQLite3 split with no writes to the recording."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        yield connection


def message_store(setup):
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

    store = get_typestore(Stores.ROS2_HUMBLE)
    definitions = {}
    for name in _RADAR_TYPES:
        interface = Path(setup, f"{name}.interface.txt").read_text(encoding="utf-8-sig")
        # ros2 interface show expands nested definitions with indentation.
        lines = [line for line in interface.splitlines() if line and not line[0].isspace()]
        definitions.update(get_types_from_msg("\n".join(lines), f"sensing_msgs/msg/{name}"))
    store.register(definitions)
    return store


def bag_epoch(connection):
    epoch = connection.execute("SELECT MIN(timestamp) FROM messages").fetchone()[0]
    if epoch is None:
        raise ValueError("No messages in the recording.")
    return int(epoch)


def records(connection, store, topic):
    row = connection.execute("SELECT id, type FROM topics WHERE name = ?", (topic,)).fetchone()
    if row is None:
        raise ValueError(f"Missing recorded topic {topic}.")
    topic_id, message_type = row
    for timestamp, blob in connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp, id",
        (topic_id,),
    ):
        yield int(timestamp), store.deserialize_cdr(blob, message_type)


def header_ns(msg):
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quaternion_rotation(quat):
    x, y, z, w = np.asarray(quat, dtype=float)
    scale = x*x + y*y + z*z + w*w
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("Invalid transform quaternion.")
    return np.array([
        [scale - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), scale - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), scale - 2*(x*x + y*y)],
    ]) / scale


def radar_to_lidar_transform(connection, store):
    """Use the two base_link mounts from the recorded /tf_static."""
    frames = {}
    for _, message in records(connection, store, _TF_TOPIC):
        for item in message.transforms:
            if item.header.frame_id == "base_link":
                t, q = item.transform.translation, item.transform.rotation
                frames[item.child_frame_id] = (
                    quaternion_rotation([q.x, q.y, q.z, q.w]),
                    np.array([t.x, t.y, t.z], dtype=float),
                )
    if not {"radar0_link", "lidar0_link"}.issubset(frames):
        raise ValueError("Recorded /tf_static lacks direct base_link mounts for radar0 and lidar0.")
    Rl, tl = frames["lidar0_link"]
    Rr, tr = frames["radar0_link"]
    return Rl.T @ Rr, Rl.T @ (tr - tl)


def point_cloud(message, include_time=False):
    """Read xyz from a PointCloud2 with named offsets, endian and row stride."""
    fields = {field.name: field for field in message.fields}
    names = ["x", "y", "z"]
    byte_order = ">" if message.is_bigendian else "<"
    if any(name not in fields or fields[name].datatype not in _FIELDS for name in names):
        raise ValueError("Expected float x/y/z fields in PointCloud2.")
    has_time = include_time and "time" in fields and fields["time"].datatype in _FIELDS
    if has_time:
        names.append("time")
    layout = np.dtype({
        "names": names,
        "formats": [byte_order + _FIELDS[fields[name].datatype] for name in names],
        "offsets": [fields[name].offset for name in names],
        "itemsize": message.point_step,
    })
    points = np.ndarray(
        (message.height, message.width), dtype=layout, buffer=message.data,
        strides=(message.row_step, message.point_step),
    )
    decoded = np.column_stack([points[name].ravel() for name in names])
    if include_time and not has_time:
        decoded = np.column_stack((decoded, np.full(len(decoded), np.nan)))
    return decoded[np.isfinite(decoded[:, :3]).all(axis=1)]


def radar_rows(message, rotation, translation):
    for target in message.targets:
        p = rotation @ np.array([target.longitudinal_position, target.lateral_position, 0.0]) + translation
        v = rotation @ np.array([target.velocity_x, target.velocity_y, 0.0])
        if np.isfinite(p).all() and np.isfinite(v).all():
            yield p, v, target


def _inside(value, interval):
    return (value >= interval[0]) & (value <= interval[1])


def _grid(points, resolution):
    return [tuple(row) for row in np.floor(points / resolution).astype(int)]


def _components(grid):
    remaining = set(grid)
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        cells, queue = [seed], [seed]
        while queue:
            x, y = queue.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell = (x + dx, y + dy)
                    if cell in remaining:
                        remaining.remove(cell)
                        cells.append(cell)
                        queue.append(cell)
        yield [index for cell in cells for index in grid[cell]]


def _extract_lidar(connection, store, epoch, gates):
    candidates, reasons = [], Counter()
    occupancies, background_frames = Counter(), 0
    for receive_ns, message in records(connection, store, _LIDAR_TOPIC):
        stamp = header_ns(message)
        if not stamp or message.header.frame_id != "lidar0_link":
            reasons["invalid_stamp_or_frame"] += 1
            continue
        xyz_time = point_cloud(message, include_time=True)
        points = xyz_time[
            _inside(xyz_time[:, 0], gates.lidar_x_m)
            & _inside(xyz_time[:, 1], gates.lidar_y_m)
            & _inside(xyz_time[:, 2], gates.lidar_z_m)
        ]
        cells = _grid(points[:, :2], gates.lidar_grid_m)
        if (receive_ns - epoch) * 1e-9 < gates.background_initial_s:
            occupancies.update(set(cells))
            background_frames += 1
            reasons["background_initialization"] += 1
            continue
        if not background_frames:
            raise ValueError("No LiDAR frames in the configured background prefix.")
        foreground = defaultdict(list)
        for i, cell in enumerate(cells):
            if occupancies[cell] / background_frames < gates.background_fraction:
                foreground[cell].append(i)
        clusters = []
        for component in _components(foreground):
            if len(component) < gates.lidar_min_points:
                continue
            cluster = points[component]
            span = np.ptp(cluster[:, :3], axis=0)
            if span[2] >= gates.lidar_min_z_span_m and max(span[:2]) <= gates.lidar_max_xy_span_m:
                clusters.append((cluster, span))
        if not clusters:
            reasons["no_body_cluster"] += 1
            continue
        clusters.sort(key=lambda item: len(item[0]), reverse=True)
        cluster, span = clusters[0]
        x, y, z = map(float, np.median(cluster[:, :3], axis=0))
        if len(clusters) > 1 and len(clusters[1][0]) >= gates.lidar_ambiguity_ratio * len(cluster):
            alternative_xy = np.median(clusters[1][0][:, :2], axis=0)
            if np.linalg.norm(alternative_xy - [x, y]) > 0.6:
                reasons["ambiguous_body_clusters"] += 1
                continue
        valid_point_times = cluster[np.isfinite(cluster[:, 3]), 3]
        candidates.append({
            "t": (stamp - epoch) * 1e-9, "header_ns": stamp,
            "bag_record_ns": receive_ns, "x": x, "y": y, "z": z,
            "quality": {
                "point_count": len(cluster),
                "body_span_xyz_m": span.tolist(),
                "body_candidates": len(clusters),
                "point_time_field_median_raw": float(np.median(valid_point_times)) if len(valid_point_times) else None,
            },
        })
    return _confirm(candidates, gates, reasons), reasons


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _extract_radar(connection, store, epoch, rotation, translation, gates):
    candidates, reasons = [], Counter()
    occupancies, background_frames = Counter(), 0
    for receive_ns, message in records(connection, store, _RADAR_TOPIC):
        stamp = header_ns(message)
        if not stamp or message.header.frame_id != "radar0_link":
            reasons["invalid_stamp_or_frame"] += 1
            continue
        if len(message.targets) != message.radar_header.target_count:
            reasons["target_count_mismatch"] += 1
            continue
        rows = [(p, v, obj) for p, v, obj in radar_rows(message, rotation, translation)
                if _inside(p[0], gates.radar_x_m) and _inside(p[1], gates.radar_y_m)]
        cells = _grid(np.array([row[0][:2] for row in rows]), gates.radar_grid_m)
        if (receive_ns - epoch) * 1e-9 < gates.background_initial_s:
            occupancies.update(set(cells))
            background_frames += 1
            reasons["background_initialization"] += 1
            continue
        if not background_frames:
            raise ValueError("No radar frames in the configured background prefix.")
        rows = [row for row, cell in zip(rows, cells)
                if occupancies[cell] / background_frames < gates.background_fraction]
        if not rows:
            reasons["no_nearby_candidate"] += 1
            continue
        # Frontmost echo in the known crossing region. Target IDs may be reused
        # or change among multiple reflections, so no ID is fixed across frames.
        rows.sort(key=lambda row: (float(row[0][0]), int(row[2].id)))
        p, v, obj = rows[0]
        if any(alt[0][0] - p[0] <= gates.radar_ambiguity_x_margin_m
               and abs(alt[0][1] - p[1]) > gates.radar_ambiguity_y_separation_m
               for alt in rows[1:]):
            reasons["ambiguous_crossing_returns"] += 1
            continue
        candidates.append({
            "t": (stamp - epoch) * 1e-9, "header_ns": stamp,
            "bag_record_ns": receive_ns, "x": float(p[0]), "y": float(p[1]),
            "radar_id": int(obj.id),
            "quality": {
                "association": "frontmost_same_sensor_roi_then_temporal_support",
                "device_stamp_us": int(message.radar_header.device_stamp),
                "candidate_count": len(rows),
                "raw_longitudinal_m": float(obj.longitudinal_position),
                "raw_lateral_m": float(obj.lateral_position),
                "velocity_xy_mps": list(map(float, v[:2])),
                "rcs_dbsm": _finite(obj.rcs),
                "move_status": int(obj.move_status),
                "maintain_status": int(obj.maintain_status),
                "track_age_frames": int(obj.track_age),
                "existence_probability": _finite(obj.existence_probability),
                "position_std_codes": [int(obj.longitudinal_std_code), int(obj.lateral_std_code)],
            },
        })
    return _confirm(candidates, gates, reasons), reasons


def _confirm(candidates, gates, reasons):
    """Reject isolated jumps using same-stream support; keep raw positions."""
    candidates.sort(key=lambda row: row["header_ns"])
    if not candidates:
        return []
    times = np.array([row["t"] for row in candidates])
    xy = np.array([[row["x"], row["y"]] for row in candidates])
    selected = []
    for i, row in enumerate(candidates):
        if i > 0 and candidates[i - 1]["header_ns"] == row["header_ns"]:
            reasons["duplicate_header"] += 1
            continue
        a = np.searchsorted(times, times[i] - gates.continuity_window_s)
        b = np.searchsorted(times, times[i] + gates.continuity_window_s, side="right")
        dt = times[a:b] - times[i]
        distance = np.linalg.norm(xy[a:b] - xy[i], axis=1)
        close = distance <= gates.continuity_slack_m + gates.continuity_speed_limit_mps * abs(dt)
        before = int(np.count_nonzero(close & (dt < 0)))
        after = int(np.count_nonzero(close & (dt > 0)))
        if not before or not after or before + after < 3:
            reasons["insufficient_temporal_support"] += 1
            continue
        row["quality"]["temporal_support_before_after"] = [before, after]
        selected.append(row)
    return selected


def _stream_summary(rows, rejects):
    summary = {
        "record_count": len(rows) + sum(rejects.values()),
        "selected_count": len(rows),
        "rejections": dict(rejects),
    }
    if rows:
        times = np.array([row["t"] for row in rows])
        xy = np.array([[row["x"], row["y"]] for row in rows])
        summary["t_range_s"] = times[[0, -1]].tolist()
        summary["xy_p05_p50_p95_m"] = np.percentile(xy, [5, 50, 95], axis=0).tolist()
        summary["xy_minmax_m"] = [xy.min(axis=0).tolist(), xy.max(axis=0).tolist()]
        summary["gap_over_0_5s_count"] = int(np.count_nonzero(np.diff(times) > 0.5))
    return summary


def extract_targets(path, setup):
    gates = ExtractionGates()
    store = message_store(setup)
    with open_bag(path) as connection:
        epoch = bag_epoch(connection)
        rotation, translation = radar_to_lidar_transform(connection, store)
        lidar, lidar_rejects = _extract_lidar(connection, store, epoch, gates)
        radar, radar_rejects = _extract_radar(connection, store, epoch, rotation, translation, gates)
    return {
        "source_kind": "real_ros2_bag",
        "bag_file": str(Path(path).resolve()),
        "epoch_ns": epoch,
        "coordinate_frame": "lidar0_link",
        "time_basis": "t = (raw ROS header_ns - first bag record timestamp) * 1e-9",
        "tf_source": "/tf_static in the same bag",
        "radar_to_lidar": {"rotation": rotation.tolist(), "translation_m": translation.tolist()},
        "gate_provenance": "selected on development run 08 and applied unchanged to run 09",
        "background_provenance": "each stream's first 3 seconds, before extraction; per-bag unlabeled occupancy",
        "association_scope": "independent within each sensor; no cross-sensor residual or aligned timestamp used",
        "limitations": [
            "One pedestrian-sized target in the fixed, stationary-ego crossing ROI.",
            "Centered same-stream temporal support makes this an offline extraction.",
            "Radar echo and LiDAR body median may refer to different parts of the person.",
            "ROS header timestamps are not independent device timing ground truth.",
        ],
        "gates": asdict(gates),
        "summary": {"lidar": _stream_summary(lidar, lidar_rejects), "radar": _stream_summary(radar, radar_rejects)},
        "lidar": lidar,
        "radar": radar,
    }


def audit_radar(connection, store, epoch, rotation, translation):
    """Inspect nearby moving IDs on the development recording before setting gates."""
    groups = defaultdict(list)
    frames = 0
    for received, msg in records(connection, store, _RADAR_TOPIC):
        frames += 1
        for p, v, obj in radar_rows(msg, rotation, translation):
            if 1 <= p[0] <= 18 and abs(p[1]) <= 8:
                groups[int(obj.id)].append([(received - epoch) * 1e-9, *p[:2], *v[:2]])
    summaries = []
    for oid, values in groups.items():
        rows = np.asarray(values)
        moving = np.linalg.norm(rows[:, 3:], axis=1) > 0.35
        if moving.sum() < 4:
            continue
        summaries.append({
            "id": oid, "count": len(rows), "moving": int(moving.sum()),
            "t_range": rows[[0, -1], 0].round(3).tolist(),
            "moving_t_range": rows[moving][[0, -1], 0].round(3).tolist(),
            "moving_xy_p05_p50_p95": np.percentile(rows[moving, 1:3], [5, 50, 95], axis=0).round(3).tolist(),
        })
    summaries.sort(key=lambda item: item["moving"], reverse=True)
    return {"radar_frames": frames, "nearby_moving_ids": summaries[:30]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="Path to finalized bag_0.db3")
    parser.add_argument("--setup", type=Path, help="Directory containing recorded Radar*.interface.txt")
    parser.add_argument("--audit-radar", action="store_true", help="Inspect development-set radar candidates")
    parser.add_argument("--out", type=Path, help="Write extracted target trajectories to a new JSON file")
    args = parser.parse_args()
    setup = args.setup or args.bag.parent.parent / "setup"
    if args.audit_radar:
        store = message_store(setup)
        with open_bag(args.bag) as connection:
            epoch = bag_epoch(connection)
            rotation, translation = radar_to_lidar_transform(connection, store)
            print(json.dumps(audit_radar(connection, store, epoch, rotation, translation), indent=2))
        return
    if not args.out:
        parser.error("Select --audit-radar or provide --out.")
    output = args.out.resolve()
    if output.is_relative_to(args.bag.resolve().parent.parent):
        parser.error("Write the result outside the raw run directory.")
    report = extract_targets(args.bag, setup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
