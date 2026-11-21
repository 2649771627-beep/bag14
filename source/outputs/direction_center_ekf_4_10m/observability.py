"""Prior-free local sensitivity of the paired observations to terminal clock offset.

The clock is linearized at zero offset and drift on the measured LiDAR reference.
Its terminal offset and one affine drift are free throughout the record. Each pair
uses the archived radar contrast covariance at that linearization. The diagnostic
conditions on the LiDAR reference and gives different pairs independent errors.
It omits cross-pair reference/process correlations and the EKF priors. Its inverse
square-root information is a descriptive local scale. The manuscript's separate
nonlinear profile, block inflation and validity checks belong to the archived study.
The current AIRY analysis reports the local scale as a continuous diagnostic.
"""
import numpy as np


def _project(column, nuisance):
    """Remove the whitened nuisance span, without imposing nuisance priors."""
    if not len(column):
        return np.empty(0), 0
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > 0
    normalized = nuisance[:, active] / norms[active]
    if not normalized.size:
        return column.copy(), 0
    basis, singular, _ = np.linalg.svd(normalized, full_matrices=False)
    rank = int(np.sum(singular > 1e-10 * singular[0]))
    basis = basis[:, :rank]
    residual = column - basis @ (basis.T @ column)
    return residual, rank


def _profile(design):
    clock, drift, center = design.T
    clock_residual, _ = _project(clock, drift[:, None])
    final_residual, rank = _project(clock, np.column_stack((drift, center)))
    raw = float(clock @ clock)
    no_center = float(clock_residual @ clock_residual)
    with_center = float(final_residual @ final_residual)
    # Exact collinearity is singular despite finite-precision projection roundoff.
    if no_center <= 1e-22 * raw:
        no_center = 0.0
    if with_center <= 1e-22 * raw:
        with_center = 0.0
    return no_center, with_center, rank


def paired_information(events, pairs, slopes, config):
    """Use the original selection's XY clock Jacobians, in vehicle coordinates."""
    columns = np.zeros((0, 2, 3))
    x_design = np.empty((0, 3))
    if len(pairs):
        i, j = np.asarray(pairs, dtype=int).T
        # Nominal clock derivatives are negative projected target velocity.
        earlier, later = np.asarray(slopes)[i], np.asarray(slopes)[j]
        stamp = np.asarray([event.receive_s for event in events])
        gap = stamp[j] - stamp[i]
        terminal = stamp[j].max()
        clock = later - earlier
        drift = (stamp[j] - terminal)[:, None]*clock + gap[:, None]*earlier
        direction = np.sign(earlier[:, 0]) - np.sign(later[:, 0])
        center = np.column_stack((direction, np.zeros_like(direction)))
        design = np.stack((clock, drift, center), axis=2)

        spec, cfg = config['sensors']['radar'], config['filter']
        variance = 2*np.square(spec['sigma']) + np.square(spec['geometry_sigma'])
        diffusion = cfg['ekf_drift_density']*gap**3/3 + cfg['ekf_offset_density']*gap
        covariance = (np.diag(variance)[None, :, :]
                      + diffusion[:, None, None]*earlier[:, :, None]*earlier[:, None, :])
        columns = np.linalg.solve(np.linalg.cholesky(covariance), design)
        # A strictly longitudinal synthetic road has zero true lateral velocity.
        # Report sensitivity excluding measured lateral motion as a secondary check.
        x_design = design[:, 0, :] / np.sqrt(covariance[:, 0, 0])[:, None]

    prior, info, rank = _profile(columns.reshape(-1, 3))
    prior_x, info_x, _ = _profile(x_design)
    return dict(local_info_no_center_s_inv2=prior, local_info_with_center_s_inv2=info,
                local_retained_fraction=info/prior if prior else None,
                local_data_scale_ms=1000/np.sqrt(info) if info else None,
                local_drift_center_rank=rank,
                local_x_retained_fraction=info_x/prior_x if prior_x else None,
                local_x_data_scale_ms=1000/np.sqrt(info_x) if info_x else None)


if __name__ == '__main__':
    from types import SimpleNamespace
    events = [SimpleNamespace(receive_s=float(i)) for i in range(16)]
    pairs = [(i, i+1) for i in range(0, 16, 2)]
    speeds = np.tile([1.0, -1.0], 8)
    cfg = {'sensors': {'radar': {'sigma': [.15, .1], 'geometry_sigma': [.04, .04]}},
           'filter': {'ekf_drift_density': 2e-9, 'ekf_offset_density': 2e-7}}
    def check(v):
        return paired_information(events, pairs, np.column_stack((-v, np.zeros_like(v))), cfg)
    assert check(speeds)['local_data_scale_ms'] is None
    # Distinct absolute speeds of both legs still alias if every pair's sum is fixed.
    magnitudes = np.linspace(.5, 1.5, 8)
    speeds[::2], speeds[1::2] = magnitudes, -(2 - magnitudes)
    assert check(speeds)['local_data_scale_ms'] is None
    speeds[::4] *= 1.5
    result = check(speeds)
    assert result['local_data_scale_ms'] > 0 and 0 < result['local_retained_fraction'] < 1
    assert np.isclose(result['local_data_scale_ms'],result['local_x_data_scale_ms'])
    assert paired_information([], [], np.empty((0, 2)), cfg)['local_data_scale_ms'] is None
    print('Verified constant speed, varying legs with fixed speed sum, varied pair sums and empty inputs.')
