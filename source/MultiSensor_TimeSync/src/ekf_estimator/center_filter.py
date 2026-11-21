"""Paired radar timing EKF with a constant direction-dependent center displacement.

In the common vehicle frame the observed target location has the additive
term q*c*a, where a is the unit motion axis and q is the sign of the
noisy LiDAR reference velocity along a. Fix q at nominal selection time. c is a signed
displacement in metres, with a zero-mean prior and zero process noise. The
original radar/camera timing state ordering is preserved, with c appended.
"""
import numpy as np

from src.ekf_estimator.filter import TemporalEKF, gaussian_update
from src.ekf_estimator.paired import pair_measurement
from src.temporal_model.model import OFFSET_INDEX, transition
from src.temporal_model.vehicle import UnsupportedReference


HUBER_DISTANCE_2D = np.sqrt(5.991)


def reference_axis(reference):
    """Return the principal moving XY velocity axis and its kinetic-energy fraction.

    Use the origin-constrained second moment so the inbound and outbound samples
    contribute to the same axis. Ignore nonfinite rows and speeds below 0.10 m/s.
    The largest-magnitude component of the eigenvector is positive.
    """
    velocities = np.asarray(reference.velocities, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] < 2:
        raise ValueError("LiDAR reference must include XY velocity knots")
    xy = velocities[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    moving = xy[np.linalg.norm(xy, axis=1) >= 0.10]
    if len(moving) < 2:
        raise UnsupportedReference("Insufficient moving LiDAR samples for the motion axis")
    energy, vectors = np.linalg.eigh(moving.T @ moving)
    if energy[-1] - energy[0] <= 1e-8 * energy[-1]:
        raise UnsupportedReference("LiDAR velocities do not determine a unique principal axis")
    axis = vectors[:, -1]
    axis = axis * np.sign(axis[np.argmax(np.abs(axis))])
    return axis, float(energy[-1]/energy.sum())


class DirectionCenterPairedEKF(TemporalEKF):
    """Use a persistent center coefficient for radar contrasts or individual updates.

    Replay the matched observations with either step or step_pair. Both updates
    consume a measurement, so applying both to the same data double-counts it.
    """

    def __init__(self, specs, config, reference, model, center_std_m=0.15,
                 axis=(1., 0.), robust=False):
        if not np.isfinite(center_std_m) or center_std_m < 0:
            raise ValueError("Center standard deviation must be finite and nonnegative")
        axis = np.asarray(axis, dtype=float)
        if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) == 0:
            raise ValueError("Center direction must be a finite, nonzero XY vector")
        super().__init__(specs, config, reference, model)
        self.axis = axis / np.linalg.norm(axis)
        self.robust = robust
        self.x = np.append(self.x, 0.)
        self.P = np.pad(self.P, ((0, 1), (0, 1)))
        self.P[4, 4] = center_std_m**2

    def predict_to(self, receive_s):
        dt = 0. if self.last_receive_s is None else receive_s - self.last_receive_s
        F, Q = transition(dt, self.cfg["ekf_drift_density"], self.cfg["ekf_offset_density"])
        joint_F = np.eye(5)
        joint_F[:4, :4] = np.kron(np.eye(2), F)
        self.x = joint_F @ self.x
        self.P = joint_F @ self.P @ joint_F.T
        self.P[:4, :4] += np.kron(np.eye(2), Q)
        self.last_receive_s = receive_s

    def _direction(self, event, fixed):
        if fixed is None:
            _, velocity = self.reference(event.reported_s)
            fixed = np.sign(velocity[:2] @ self.axis)
        if fixed not in (-1, 1):
            raise UnsupportedReference("Missing longitudinal direction for a radar observation")
        return fixed

    def step(self, event, direction=None):
        """Assimilate one radar point using its nominal LiDAR direction sign."""
        if event.sensor != "radar":
            raise ValueError("Direction-center measurements require radar XY in the common vehicle frame")
        self.predict_to(event.receive_s)
        direction = self._direction(event, direction)
        index = OFFSET_INDEX["radar"]
        offset = self.x[index]
        spec = self.specs["radar"]
        expected, slope = self.observation_model(event, offset, spec, self.reference)
        H = np.zeros((2, 5))
        H[:, index] = slope
        H[:, 4] = direction * self.axis
        innovation = event.value - (expected + H[:, 4] * self.x[4])
        R = np.diag(np.square(spec["sigma"]))
        return self._radar_update(innovation, H, R, offset)

    def step_pair(self, earlier, later, geometry_sigma, q_early=None, q_late=None):
        if earlier.sensor != "radar" or later.sensor != "radar" or earlier.receive_s >= later.receive_s:
            raise ValueError("Expected two time-ordered radar observations in the common vehicle frame")
        self.predict_to(later.receive_s)
        q_early = self._direction(earlier, q_early)
        q_late = self._direction(later, q_late)
        index = OFFSET_INDEX["radar"]
        offset, drift = self.x[index:index + 2]
        spec = self.specs["radar"]
        expected, timing_J, early_slope = pair_measurement(
            earlier, later, offset, drift, spec, self.reference, self.observation_model)
        H = np.zeros((2, 5))
        H[:, index:index + 2] = timing_J
        H[:, 4] = (q_late - q_early) * self.axis
        expected = expected + H[:, 4] * self.x[4]

        gap = later.receive_s - earlier.receive_s
        _, Q = transition(gap, self.cfg["ekf_drift_density"], self.cfg["ekf_offset_density"])
        R = np.diag(2 * np.square(spec["sigma"]) + np.square(geometry_sigma))
        R += np.outer(early_slope, early_slope) * Q[0, 0]
        innovation = later.value - earlier.value - expected
        return self._radar_update(innovation, H, R, offset)

    def _radar_update(self, innovation, H, R, offset):
        """Single-pass Huber weighting of the 2-D standardized prediction error.

        Compute D before reweighting, with S = H P H^T + R. Beyond the
        fixed sqrt(5.991) radius the effective measurement covariance is R/w,
        w = sqrt(5.991)/D. Use the Joseph update with the effective covariance.
        """
        predictive = H @ self.P @ H.T + R
        distance = float(np.linalg.norm(np.linalg.solve(np.linalg.cholesky(predictive), innovation)))
        weight = HUBER_DISTANCE_2D / distance if self.robust and distance > HUBER_DISTANCE_2D else 1.
        self.x, self.P, _ = gaussian_update(self.x, self.P, innovation, H, R/weight)
        return dict(**self.estimate("radar", offset), innovation_weight=weight,
                    innovation_mahalanobis=distance)

    def estimate(self, sensor, prior_offset):
        timing = super().estimate(sensor, prior_offset)
        center_std = float(np.sqrt(self.P[4, 4]))
        cross = float(self.P[OFFSET_INDEX[sensor], 4])
        denominator = timing["offset_std_s"] * center_std
        return dict(**timing, center_m=float(self.x[4]), center_std_m=center_std,
                    offset_center_cov_s_m=cross,
                    offset_center_corr=cross / denominator if denominator else None)
