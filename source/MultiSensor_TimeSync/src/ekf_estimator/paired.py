"""Use non-reused opposite-direction observation contrasts in the temporal EKF."""

import numpy as np

from src.ekf_estimator.filter import TemporalEKF, gaussian_update
from src.temporal_model.model import OFFSET_INDEX, transition


def pair_measurement(earlier, later, offset, drift, spec, reference, observation_model):
    """Predict the contrast and its Jacobian at the later receive epoch."""
    gap = later.receive_s - earlier.receive_s
    early_value, early_slope = observation_model(earlier, offset - gap * drift, spec, reference)
    late_value, late_slope = observation_model(later, offset, spec, reference)
    return late_value - early_value, np.column_stack((late_slope - early_slope, gap * early_slope)), early_slope


class PairedTemporalEKF(TemporalEKF):
    def step_pair(self, earlier, later, geometry_sigma):
        if earlier.sensor != later.sensor or earlier.receive_s >= later.receive_s:
            raise ValueError("Pair members must be time-ordered observations of the same sensor")
        self.predict_to(later.receive_s)
        index = OFFSET_INDEX[later.sensor]
        offset, drift = self.x[index:index + 2]
        spec = self.specs[later.sensor]
        expected, slope, early_slope = pair_measurement(
            earlier, later, offset, drift, spec, self.reference, self.observation_model,
        )
        H = np.zeros((len(expected), 4))
        H[:, index:index + 2] = slope
        # Contrast noise includes both observations and a local geometry mismatch floor.
        R = np.diag(2 * np.square(spec["sigma"]) + np.square(geometry_sigma))
        gap = later.receive_s - earlier.receive_s
        _, Q = transition(gap, self.cfg["ekf_drift_density"], self.cfg["ekf_offset_density"])
        R += np.outer(early_slope, early_slope) * Q[0, 0]
        innovation = (later.value - earlier.value) - expected
        self.x, self.P, _ = gaussian_update(self.x, self.P, innovation, H, R)
        return self.estimate(later.sensor, offset)
