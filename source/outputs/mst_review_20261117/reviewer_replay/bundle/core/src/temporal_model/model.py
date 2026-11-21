"""Relative offset sign convention: reported time = acquisition time + offset."""
from dataclasses import dataclass

import numpy as np


SENSORS = ("radar", "camera")
OFFSET_INDEX = {"radar": 0, "camera": 2}


@dataclass(frozen=True)
class Observation:
    id: int
    sensor: str
    reported_s: float
    receive_s: float
    value: np.ndarray


def kinematics(t, stationary=False):
    """Analytic reference target in an ideal LiDAR-aligned ground plane, in meters."""
    if stationary:
        return np.array([17.0, 1.2]), np.zeros(2)
    position = np.array([
        17.0 + 1.4 * np.sin(0.51 * t) + 1.1 * np.cos(0.23 * t),
        1.9 * np.sin(0.76 * t) + 0.52 * np.sin(0.29 * t),
    ])
    velocity = np.array([
        1.4 * 0.51 * np.cos(0.51 * t) - 1.1 * 0.23 * np.sin(0.23 * t),
        1.9 * 0.76 * np.cos(0.76 * t) + 0.52 * 0.29 * np.cos(0.29 * t),
    ])
    return position, velocity


def project(position, velocity, spec):
    if spec["measurement"] == "position_xy":
        return position, velocity
    if spec["measurement"] == "pinhole_u":
        depth, left = position
        if depth <= 0.0:
            raise ValueError("Pinhole projection requires a positive depth")
        v_depth, v_left = velocity
        fx = spec["fx"]
        u = spec["cx"] - fx * left / depth
        du = -fx * (v_left * depth - left * v_depth) / depth**2
        return np.array([u]), np.array([du])
    raise ValueError("Unknown observation model: " + spec["measurement"])


def measurement(event, offset_s, spec, reference):
    """Return predicted observation and the derivative with respect to offset."""
    position, velocity = reference(event.reported_s - offset_s)
    expected, derivative_t = project(position, velocity, spec)
    return expected, -derivative_t


def transition(dt, drift_density, offset_density=0.0):
    """db = d ds + sqrt(q_b) dW_b, dd = sqrt(q_d) dW_d."""
    if dt < 0 or drift_density < 0 or offset_density < 0:
        raise ValueError("Time interval and process spectral densities must be non-negative")
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = drift_density * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    Q[0, 0] += offset_density * dt
    return F, Q


def initial_covariance(cfg):
    return np.diag([cfg["initial_offset_std_s"] ** 2, cfg["initial_drift_std_sps"] ** 2])


def observability(events, specs, reference, offset_at):
    """Whitened local sensitivity to initial offset/drift under the supplied reference."""
    results = {}
    for sensor in SENSORS:
        selected = [event for event in events if event.sensor == sensor]
        s0 = selected[0].receive_s if selected else 0.0
        rows = []
        for event in selected:
            _, dh_db = measurement(event, offset_at(event), specs[sensor], reference)
            scaled = dh_db / np.array(specs[sensor]["sigma"])
            rows.append(np.outer(scaled, [0.010, 0.001 * (event.receive_s - s0)]))
        J = np.vstack(rows) if rows else np.empty((0, 2))
        values = np.linalg.svd(J, compute_uv=False)
        results[sensor] = {
            "rank": int(np.linalg.matrix_rank(J)) if J.size else 0,
            "singular_values": values.tolist(),
            "events": len(selected),
            "units_scaled_to": {"offset_s": 0.01, "drift_s_per_s": 0.001},
        }
    results["joint_rank"] = sum(results[sensor]["rank"] for sensor in SENSORS)
    return results
