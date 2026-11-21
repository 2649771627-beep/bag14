"""Use production filters on an analytic reference with a known radar offset."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPRO = ROOT / 'bundle/core'
sys.path.insert(0, str(REPRO))

from experiments.paired_ekf import select_observations, match_pairs
from paired_displacement import replay
from src.ekf_estimator.filter import TemporalEKF
from src.ekf_estimator.paired import PairedTemporalEKF
from src.temporal_model.model import Observation
from src.temporal_model.vehicle import TargetReference, VehicleObservationModel


N_PAIRS = 16
REPETITIONS = 500
TRUTH_S = .030
SEED = 20261117
BOOTSTRAP_SEED = 20261118
SHIFTS = {'zero': (0., 0.), 'common_plus': (.08, 0.), 'common_minus': (-.08, 0.),
          'direction_plus': (0., .08), 'direction_minus': (0., -.08)}
MOTIONS = {'unequal_1p2_1p8': (1.2, 1.8), 'equal_1p5_1p5': (1.5, 1.5)}


def build_trial(u, w, cfg):
    pos = 2. + .27*np.arange(N_PAIRS)
    up_t = 1. + (pos-2.) / u
    t0, t1 = 6., 7.
    x0 = 2.+u*(t0-1.)
    acceleration = -(u+w)/(t1-t0)
    x1 = x0+(t1-t0)*(u+.5*acceleration*(t1-t0))
    assert x1 > pos[-1]
    down_t = t1+(x1-pos[::-1])/w
    true_t = np.r_[up_t, down_t]
    reported = true_t + TRUTH_S
    receive_ns = np.rint(reported*1.e9).astype(np.int64)
    true_x = np.r_[pos, pos[::-1]]
    nodes = np.arange(-2., down_t[-1]+2., .025)
    mid = nodes-t0
    reference_x = np.where(nodes<t0, 2.+u*(nodes-1.),
                           np.where(nodes<=t1, x0+u*mid+.5*acceleration*mid**2,
                                    x1-w*(nodes-t1)))
    reference_rows = [dict(t=t,x=x,y=0.,z=0.) for t,x in zip(nodes,reference_x,strict=True)]
    reference = TargetReference(reference_rows, cfg['reference_gap_s'], cfg['reference_smooth_half_window_s'])
    model = VehicleObservationModel((np.eye(4),np.eye(4),np.eye(3),np.zeros(4)))
    rows = [dict(t=t,x=x,y=0.,bag_record_ns=int(r))
            for t,x,r in zip(reported,true_x,receive_ns,strict=True)]
    events, positions, slopes = select_observations(rows, 'radar', 0, cfg, reference, model)
    pairs = match_pairs(events, positions, slopes, 'radar', cfg)
    assert pairs == [(N_PAIRS-1-j,N_PAIRS+j) for j in range(N_PAIRS)], pairs
    assert len(events)==2*N_PAIRS
    sign = np.r_[np.ones(N_PAIRS), -np.ones(N_PAIRS)]
    velocity = np.r_[np.full(N_PAIRS,u), np.full(N_PAIRS,-w)]
    # Every true acquisition is covered by the smooth straight reference at the true clock value.
    for event, expected_x, v in zip(events,true_x,velocity,strict=True):
        p, vp = reference(event.reported_s-TRUTH_S)
        np.testing.assert_allclose([p[0],vp[0]],[expected_x,v],atol=1.e-10,rtol=0)
    return events, pairs, model, reference, true_x, velocity, sign


def endpoint_estimators(events, pairs, cfg, model, ref):
    raw = TemporalEKF(cfg['sensors'], cfg['filter'], ref, model)
    paired = PairedTemporalEKF(cfg['sensors'], cfg['filter'], ref, model)
    for estimator in (raw, paired):
        estimator.predict_to(events[0].receive_s)
    raw_queries = []
    pair_queries = []
    for event in events:
        raw_queries.append(event.reported_s-raw.x[0])
        raw.step(event)
    for i,j in pairs:
        gap = events[j].receive_s-events[i].receive_s
        elapsed = events[j].receive_s-paired.last_receive_s
        late = paired.x[0]+elapsed*paired.x[1]
        early = late-gap*paired.x[1]
        pair_queries.extend((events[i].reported_s-early,events[j].reported_s-late))
        paired.step_pair(events[i], events[j], cfg['sensors']['radar']['geometry_sigma'])
    assert raw.last_receive_s == paired.last_receive_s == events[-1].receive_s
    return raw.x[0],paired.x[0],raw_queries,pair_queries


def interval_samples(errors_a, errors_b, rng):
    choice = rng.integers(len(errors_a),size=(2000,len(errors_a)))
    raw = np.sqrt(np.mean(errors_a[choice]**2,axis=1))
    pair = np.sqrt(np.mean(errors_b[choice]**2,axis=1))
    return np.quantile(raw-pair,[.025,.975]).tolist()


def metrics(errors):
    return {'bias_ms': float(np.mean(errors)), 'sd_ms': float(np.std(errors,ddof=1)),
            'rmse_ms': float(np.sqrt(np.mean(errors**2)))}


def expected_static(u,w,a,c,sigma):
    velocity_power = u*u+w*w
    speed_sum = u+w
    raw_bias = -(a*(u-w)+c*speed_sum)/velocity_power
    diff_bias = -2*c/speed_sum
    raw_variance = sigma*sigma / (N_PAIRS*velocity_power)
    diff_variance = 2*sigma*sigma / (N_PAIRS*speed_sum**2)
    return {'framewise_static': {'bias_ms': 1.e3*raw_bias,'sd_ms': 1.e3*np.sqrt(raw_variance),
                                 'rmse_ms': 1.e3*np.sqrt(raw_variance+raw_bias**2)},
            'paired_static': {'bias_ms': 1.e3*diff_bias,'sd_ms': 1.e3*np.sqrt(diff_variance),
                              'rmse_ms': 1.e3*np.sqrt(diff_variance+diff_bias**2)}}


def main():
    config_path = REPRO/'inputs/filter_config.json'
    cfg = json.loads(config_path.read_text(encoding='utf8'))
    sigma = np.asarray(cfg['sensors']['radar']['sigma'])
    noise = np.random.default_rng(SEED).normal(size=(REPETITIONS,2*N_PAIRS,2))*sigma
    rng_ci = np.random.default_rng(BOOTSTRAP_SEED)
    reported = {'config_path': str(config_path.relative_to(ROOT)), 'trial_count': REPETITIONS,
                'seed': SEED, 'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_repetitions': 2000,
                'offset_truth_ms': 1000*TRUTH_S, 'drift_truth_sps': 0., 'motion_designs': {},
                'config': cfg, 'limits': 'Synthetic terminal RMSE, not recorded clock accuracy or the 50 ms gate.'}
    errors_save = {}
    for name,(u,w) in MOTIONS.items():
        events,pairs,model,ref,true_x,velocity,sign=build_trial(u,w,cfg)
        early,late=np.array(pairs).T
        q_t=np.array([event.reported_s for event in events])
        raw_nominal=np.array([ref(t)[0][0] for t in q_t])
        np.testing.assert_allclose(raw_nominal-true_x,velocity*TRUTH_S,atol=1.e-10,rtol=0)
        x_design=np.column_stack([np.ones(2*N_PAIRS),-velocity])
        h_pair=velocity[early]-velocity[late]
        assert np.linalg.matrix_rank(np.c_[x_design,sign])==2
        histories={}
        last_outbound_horizon=0.
        earliest_return_horizon=np.inf
        for shift,(a,c) in SHIFTS.items():
            values=noise.copy()
            values[:,:,0]+=true_x+a+c*sign
            d_x=values[:,:,0]-raw_nominal
            sample={'framewise_static': 1000*(-d_x@velocity/(velocity@velocity)-TRUTH_S)}
            sample['framewise_common_intercept_static']=1000*(d_x@np.linalg.pinv(x_design)[1]-TRUTH_S)
            sample['paired_static']=1000*((d_x[:,late]-d_x[:,early])@h_pair/(h_pair@h_pair)-TRUTH_S)
            np.testing.assert_allclose(sample['paired_static'],sample['framewise_common_intercept_static'],atol=1.e-11,rtol=0)
            raw_errors, pair_errors = [], []
            for j,value in enumerate(values):
                observations=[Observation(e.id,e.sensor,e.reported_s,e.receive_s,v)
                              for e,v in zip(events,value,strict=True)]
                raw,pair,rquery,pquery=endpoint_estimators(observations,pairs,cfg,model,ref)
                # Actual model queries must remain on the declared straight legs.
                pair_order = np.column_stack((early,late)).ravel()
                q = np.r_[rquery,pquery]
                direction = np.r_[sign,sign[pair_order]]
                last_outbound_horizon=max(last_outbound_horizon,q[direction>0].max())
                earliest_return_horizon=min(earliest_return_horizon,q[direction<0].min())
                assert q[direction>0].max() < 5.55 and q[direction<0].min() > 7.45
                raw_errors.append(1000*(raw-TRUTH_S))
                pair_errors.append(1000*(pair-TRUTH_S))
                if j in (0, REPETITIONS-1):
                    check=replay(dict(enumerate(observations)),pairs,cfg,ref,model,events[0].receive_s)
                    assert not any(check['reference_failures'].values())
                    assert raw==check['outputs']['raw_completed_pairs'][pairs[-1][1]]['offset_s']
                    assert pair==check['outputs']['difference'][pairs[-1][1]]['offset_s']
            sample.update(framewise_ekf=np.asarray(raw_errors),paired_ekf=np.asarray(pair_errors))
            for method,array in sample.items():
                errors_save[f'{name}__{shift}__{method}']=array
            truth=expected_static(u,w,a,c,sigma[0])
            check_stats={method: metrics(array) for method,array in sample.items()}
            for method,ideal in truth.items():
                assert abs(check_stats[method]['bias_ms']-ideal['bias_ms']) < 4*ideal['sd_ms']/np.sqrt(REPETITIONS)
            sraw=check_stats['framewise_ekf']['rmse_ms']
            spair=check_stats['paired_ekf']['rmse_ms']
            histories[shift]={'common_m':a,'direction_m':c,'analytic_static':truth,'empirical':check_stats,
                              'ekf_rmse_gain_ms':sraw-spair,'ekf_rmse_gain_percent':100*(1-spair/sraw),
                              'ekf_rmse_gain_95pct_mc_ci_ms':interval_samples(sample['framewise_ekf'],sample['paired_ekf'],rng_ci)}
        for key in ('common_plus','common_minus'):
            np.testing.assert_allclose(errors_save[f'{name}__{key}__paired_ekf'],
                                       errors_save[f'{name}__zero__paired_ekf'],rtol=0,atol=5.e-12)
        receive_times=np.array([e.receive_s for e in events])
        design={'pair_count':len(pairs),'pairs':pairs,'speed_out_mps':u,'speed_return_mps':w,
                'event_times_s':receive_times.tolist(),'last_outbound_query_s':last_outbound_horizon,
                'earliest_return_query_s':earliest_return_horizon,
                'pair_separation_range_s':[float(np.min(receive_times[late]-receive_times[early])),
                                           float(np.max(receive_times[late]-receive_times[early]))],
                'prior_receive_s':float(receive_times[0]),'terminal_receive_s':float(receive_times[-1]),'conditions':histories}
        reported['motion_designs'][name]=design
        print(json.dumps({name: {k:{'gain_ms':v['ekf_rmse_gain_ms'],'ci_ms':v['ekf_rmse_gain_95pct_mc_ci_ms'],
                                   'raw_rmse_ms':v['empirical']['framewise_ekf']['rmse_ms'],
                                   'pair_rmse_ms':v['empirical']['paired_ekf']['rmse_ms']} for k,v in histories.items()}}),flush=True)
    paths=[config_path,Path(__file__),HERE/'PROTOCOL.md', REPRO/'paired_displacement.py',
           REPRO/'experiments/paired_ekf.py',REPRO/'src/temporal_model/vehicle.py',REPRO/'src/temporal_model/model.py',
           REPRO/'src/ekf_estimator/filter.py',REPRO/'src/ekf_estimator/paired.py']
    reported['source_sha256']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (HERE/'results.json').write_text(json.dumps(reported,indent=2)+'\n', encoding='utf-8')
    np.savez_compressed(HERE/'signed_endpoint_errors_ms.npz',**errors_save)


if __name__ == '__main__':
    main()
