"""Matched speed groups on bag 08 and a whole-recording check on bag 09."""
import hashlib

import numpy as np

import profile_estimator as profile

MEDIAN_HALF_WIDTH_MPS = .08
POSITION_BIN_M = 1.
OFFSET_FOR_GROUPS_S = 0.
DIFFERENCE_STEP_S = .0001


def choose_groups(problem):
    """Match direction and position-box counts using the reference only."""
    available = np.flatnonzero(problem.mask)
    position, velocity, valid = profile.reference_many(problem.ref, problem.t[available])
    assert valid.all()
    axis_position = position @ problem.axis
    signed_velocity = velocity @ problem.axis
    sign = problem.direction[available]
    speed = np.abs(signed_velocity)
    position_bin = np.floor(axis_position / POSITION_BIN_M).astype(int)
    groups = {'near_constant': [], 'variable': []}
    matches = []
    quantiles = {}
    for direction in (1, -1):
        same_direction = sign == direction
        q1, median, q3 = np.quantile(speed[same_direction], [.25, .5, .75], method='linear')
        quantiles[str(direction)] = [float(q1), float(median), float(q3)]
        near = np.abs(speed - median) <= MEDIAN_HALF_WIDTH_MPS
        tails = ((speed <= q1) | (speed >= q3)) & ~near
        for k in np.unique(position_bin[same_direction]):
            in_bin = same_direction & (position_bin == k)
            a = available[in_bin & near]
            b = available[in_bin & tails]
            count = min(len(a), len(b))
            matches.append({'direction': direction, 'position_bin_start_m': int(k),
                            'near_candidates': len(a), 'variable_candidates': len(b),
                            'selected_in_each': count})
            for label, candidates in (('near_constant', a), ('variable', b)):
                keep = np.linspace(0, len(candidates) - 1, count).astype(int)
                groups[label].extend(candidates[keep].tolist())
    groups = {label: np.array(sorted(indices), dtype=int) for label, indices in groups.items()}
    assert not np.intersect1d(*groups.values()).size
    assert len(groups['near_constant']) == len(groups['variable']) > 0
    return groups, {'speed_quantiles_mps': quantiles, 'bins': matches}


def subset(problem, geometry, indices):
    mask = np.zeros(len(problem.events), dtype=bool)
    mask[indices] = True
    assert np.all(~mask | problem.mask)
    signs = problem.direction[mask]
    support = {**problem.support, 'mask': mask.tolist(), 'retained_events': len(indices),
               'positive_events': int(np.sum(signs > 0)), 'negative_events': int(np.sum(signs < 0))}
    return profile.ProfileProblem(problem.events, problem.ref, geometry, 'radar', problem.sigma,
                                  support=support)


def fixed_information(problem):
    t = problem.t[problem.mask]
    q = problem.direction[problem.mask]

    def design(offset):
        y, columns, valid = profile.prediction_design(problem.ref, problem.model, 'radar', t,
                                                       offset, problem.axis, q, 'direction')
        assert valid.all()
        return y, columns

    y_plus, _ = design(DIFFERENCE_STEP_S)
    y_minus, _ = design(-DIFFERENCE_STEP_S)
    _, columns = design(0.)
    scales = np.tile(problem.sigma, len(t))
    h = (y_plus - y_minus).ravel() / (2 * DIFFERENCE_STEP_S * scales)
    matrix = columns / scales[:, None]
    raw = float(h @ h)
    results = {}
    for mode, A in (('geometry', matrix[:, :3]), ('direction', matrix)):
        U, singular, _ = np.linalg.svd(A, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * profile.CONFIG['nuisance_rank_rtol']))
        basis = U[:, :rank]
        projected = h - basis @ (basis.T @ h)
        info = float(projected @ projected)
        results[mode] = {'projected_information_s_minus2': info, 'retained_information_fraction': info / raw,
                         'nuisance_rank': rank, 'unprojected_information_s_minus2': raw}
    assert results['direction']['projected_information_s_minus2'] <= results['geometry']['projected_information_s_minus2']
    return results


def sample_details(problem):
    mask = problem.mask
    t = problem.t[mask]
    p, v, valid = profile.reference_many(problem.ref, t)
    assert valid.all()
    position = p @ problem.axis
    speed = np.abs(v @ problem.axis)
    signs = problem.direction[mask]
    ids = problem.ids[mask].tolist()
    stats = {}
    for q in (1, -1):
        s = speed[signs == q]
        stats[str(q)] = {'n': len(s), 'speed_mean_mps': float(s.mean()),
                         'speed_std_mps': float(s.std()), 'speed_rms_mps': float(np.sqrt(np.mean(s*s)))}
    return {'ids': ids, 'id_sha256': hashlib.sha256(','.join(map(str, ids)).encode('ascii')).hexdigest(),
            'heading_speed': stats, 'header_span_s': [float(t.min()), float(t.max())],
            'receive_span_s': [float(problem.r[mask].min()), float(problem.r[mask].max())],
            'samples': [{'id': k, 'header_s': float(tt), 'receive_s': float(tr), 'direction': int(q),
                         'axis_position_m': float(x), 'axis_speed_mps': float(s)}
                        for k, tt, tr, q, x, s in zip(ids, t, problem.r[mask], signs, position, speed)]}


