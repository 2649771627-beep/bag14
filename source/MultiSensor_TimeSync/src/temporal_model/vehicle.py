"""Offline target trajectory and the nominal vehicle observation geometry."""

import json
from pathlib import Path

import numpy as np


class UnsupportedReference(ValueError):
    pass


class TargetReference:
    def __init__(self, rows, max_gap, smooth_half_window):
        self.times = np.array([row['t'] for row in rows], dtype=float)
        positions = np.array([[row['x'], row['y'], row['z']] for row in rows], dtype=float)
        if len(self.times) < 3 or np.any(np.diff(self.times) <= 0):
            raise ValueError('LiDAR target timestamps must be strictly increasing')
        self.max_gap = max_gap
        self.positions = np.empty_like(positions)
        self.velocities = np.empty_like(positions)
        runs = np.split(np.arange(len(self.times)), np.flatnonzero(np.diff(self.times) > max_gap) + 1)
        for run in runs:
            for i in run:
                near = run[np.abs(self.times[run] - self.times[i]) <= smooth_half_window]
                dt = self.times[near] - self.times[i]
                degree = min(2, len(near) - 1)
                terms = np.vander(dt, degree + 1, increasing=True)
                weights = np.exp(-0.5 * np.square(dt / (smooth_half_window / 2)))
                fit, *_ = np.linalg.lstsq(terms * weights[:, None], positions[near] * weights[:, None], rcond=None)
                self.positions[i] = fit[0]
                self.velocities[i] = fit[1] if degree else 0.0

    def __call__(self, t):
        right = int(np.searchsorted(self.times, t, side='right'))
        if right == 0 or right == len(self.times):
            raise UnsupportedReference('Time is outside the reference trajectory')
        left = right - 1
        dt = self.times[right] - self.times[left]
        if dt > self.max_gap:
            raise UnsupportedReference('Time is inside a LiDAR trajectory gap')
        u = (t - self.times[left]) / dt
        p0, p1 = self.positions[[left, right]]
        v0, v1 = self.velocities[[left, right]]
        position = (2*u**3 - 3*u**2 + 1)*p0 + (u**3 - 2*u**2 + u)*dt*v0 + (-2*u**3 + 3*u**2)*p1 + (u**3 - u**2)*dt*v1
        velocity = (6*u**2 - 6*u)*(p0 - p1)/dt + (3*u**2 - 4*u + 1)*v0 + (3*u**2 - 2*u)*v1
        return position, velocity


def quaternion_matrix(rotation):
    x, y, z, w = (rotation[k] for k in 'xyzw')
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def load_geometry(calibration):
    calibration = Path(calibration)
    static = json.loads((calibration / 'tf_static_live.json').read_text(encoding='utf-8'))
    info = json.loads((calibration / 'camera_info_live.json').read_text(encoding='utf-8'))
    edges = {edge['child_frame_id']: edge for edge in static}

    def to_base(frame):
        matrix = np.eye(4)
        while frame != 'base_link':
            edge = edges[frame]
            transform = edge['transform']
            parent_from_child = np.eye(4)
            parent_from_child[:3, :3] = quaternion_matrix(transform['rotation'])
            parent_from_child[:3, 3] = [transform['translation'][k] for k in 'xyz']
            matrix = parent_from_child @ matrix
            frame = edge['header']['frame_id']
        return matrix

    if info['distortion_model'] != 'equidistant':
        raise ValueError('Expected the recorded equidistant camera model')
    base_from_lidar = to_base('lidar0_link')
    camera_from_lidar = np.linalg.solve(to_base('camera2_optical_frame'), base_from_lidar)
    radar_from_lidar = np.linalg.solve(to_base('radar0_link'), base_from_lidar)
    return camera_from_lidar, radar_from_lidar, np.array(info['k']).reshape(3, 3), np.array(info['d'])


class VehicleObservationModel:
    def __init__(self, geometry, lateral_m=0.0, yaw_deg=0.0):
        self.camera_from_lidar, radar_from_lidar, self.K, self.D = geometry
        self.centers = {'camera': np.linalg.inv(self.camera_from_lidar)[:3, 3],
                        'radar': np.linalg.inv(radar_from_lidar)[:3, 3]}
        yaw = np.deg2rad(yaw_deg)
        c, s = np.cos(yaw), np.sin(yaw)
        self.rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.translation = np.array([0.0, lateral_m, 0.0])

    def project(self, sensor, point):
        # Use a fixed yaw about the sensor mounting point and a lateral error in LiDAR axes.
        center = self.centers[sensor]
        point = center + self.rotation @ (point - center) + self.translation
        return self._project_transformed(sensor, point)

    def geometry_sensitivity(self, sensor, point, lateral_m, yaw_deg):
        """Observation changes along scaled lateral and yaw extrinsic directions."""
        center = self.centers[sensor]
        relative = self.rotation @ (point - center)
        transformed = center + relative + self.translation
        lateral = np.array([0.0, lateral_m, 0.0])
        yaw = np.deg2rad(yaw_deg) * np.array([-relative[1], relative[0], 0.0])
        epsilon = 1e-3
        return np.column_stack([
            (self._project_transformed(sensor, transformed + epsilon * direction)
             - self._project_transformed(sensor, transformed - epsilon * direction)) / (2 * epsilon)
            for direction in (lateral, yaw)
        ])

    def _project_transformed(self, sensor, point):
        if sensor == 'radar':
            # Radar target reports were transformed to nominal LiDAR XY by the extractor.
            return point[:2]
        x, y, z = self.camera_from_lidar[:3, :3] @ point + self.camera_from_lidar[:3, 3]
        if z <= 0.1:
            raise UnsupportedReference('Reference target is behind the usable camera plane')
        a, b = x / z, y / z
        radius = np.hypot(a, b)
        theta = np.arctan(radius)
        theta2 = theta * theta
        distorted = theta * (1 + self.D @ np.power(theta2, np.arange(1, 5)))
        normalized_x = a * distorted / radius if radius else 0.0
        return np.array([self.K[0, 0] * normalized_x + self.K[0, 2]])

    def __call__(self, event, offset, spec, reference):
        position, velocity = reference(event.reported_s - offset)
        value = self.project(event.sensor, position)
        epsilon = 1e-4
        slope = (self.project(event.sensor, position - epsilon*velocity)
                 - self.project(event.sensor, position + epsilon*velocity)) / (2 * epsilon)
        return value, slope
