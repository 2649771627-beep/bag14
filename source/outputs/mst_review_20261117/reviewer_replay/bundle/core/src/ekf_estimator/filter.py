"""Four-state EKF with independent sensor observations on a shared event clock."""
import numpy as np

from src.temporal_model.model import OFFSET_INDEX, initial_covariance, measurement, transition


def gaussian_update(x, P, innovation, H, R):
    """Joseph covariance update and predicted Gaussian log likelihood."""
    cross = P @ H.T
    S = H @ cross + R
    K = np.linalg.solve(S, cross.T).T
    left = np.eye(len(x)) - K @ H
    posterior = left @ P @ left.T + K @ R @ K.T
    _, log_det = np.linalg.slogdet(S)
    log_likelihood = -0.5 * (
        len(innovation) * np.log(2 * np.pi) + log_det + innovation @ np.linalg.solve(S, innovation)
    )
    return x + K @ innovation, (posterior + posterior.T) * 0.5, float(log_likelihood)


class TemporalEKF:
    def __init__(self, specs, cfg, reference, observation_model=measurement):
        self.specs, self.cfg, self.reference = specs, cfg, reference
        self.observation_model = observation_model
        self.x = np.zeros(4)
        self.P = np.kron(np.eye(2), initial_covariance(cfg))
        self.last_receive_s = None

    def predict_to(self, receive_s):
        dt = 0.0 if self.last_receive_s is None else receive_s - self.last_receive_s
        F, Q = transition(dt, self.cfg["ekf_drift_density"], self.cfg["ekf_offset_density"])
        F, Q = np.kron(np.eye(2), F), np.kron(np.eye(2), Q)
        self.last_receive_s = receive_s
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q

    def step(self, event):
        self.predict_to(event.receive_s)
        index = OFFSET_INDEX[event.sensor]
        spec = self.specs[event.sensor]
        prior_offset = self.x[index]
        expected, slope = self.observation_model(event, prior_offset, spec, self.reference)
        H = np.zeros((len(expected), 4))
        H[:, index] = slope
        R = np.diag(np.square(spec["sigma"]))
        self.x, self.P, _ = gaussian_update(self.x, self.P, event.value - expected, H, R)
        return self.estimate(event.sensor, prior_offset)

    def estimate(self, sensor, prior_offset):
        index = OFFSET_INDEX[sensor]
        return {
            "offset_s": float(self.x[index]), "drift_sps": float(self.x[index + 1]),
            "offset_std_s": float(np.sqrt(self.P[index, index])),
            "prior_offset_s": float(prior_offset),
        }
