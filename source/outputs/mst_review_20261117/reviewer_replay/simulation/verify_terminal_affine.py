"""Verify the terminal error distributions on the straight reference branches.

The production EKFs are numerically affine while every observation query stays
in its assigned constant-speed leg. Probe every native XY observation with one
small coordinate impulse. Check that these coefficients predict all archived
Monte Carlo outputs, then compute conditional Gaussian moments analytically.
"""
import json
import numpy as np

from run_known_truth import (HERE, MOTIONS, N_PAIRS, REPETITIONS, REPRO, SEED,
                             SHIFTS, TRUTH_S, Observation, build_trial,
                             endpoint_estimators)


def calculate(cfg, archived, run):
    sigma = np.asarray(cfg['sensors']['radar']['sigma'])
    rng = np.random.default_rng(SEED)
    noise = rng.normal(size=(REPETITIONS,2*N_PAIRS,2))*sigma
    result = {}
    for motion,(u,w) in MOTIONS.items():
        events,pairs,model,ref,true_x,velocity,sign=build_trial(u,w,cfg)
        expected = np.column_stack((true_x,np.zeros_like(true_x)))

        def terminal(value):
            observations = [Observation(e.id,e.sensor,e.reported_s,e.receive_s,v)
                            for e,v in zip(events,value,strict=True)]
            return np.array(endpoint_estimators(observations,pairs,cfg,model,ref)[:2])

        baseline = terminal(expected)
        gain = np.empty((2*N_PAIRS,2,2))
        for index in np.ndindex(expected.shape):
            test = expected.copy()
            test[index] += .01
            gain[index] = (terminal(test)-baseline)/.01
        sd = 1000*np.sqrt(np.sum((gain*sigma[None,:,None])**2,axis=(0,1)))
        shifts={}
        for key,(a,c) in SHIFTS.items():
            shifts_xy = np.column_stack((a+c*sign,np.zeros_like(true_x)))
            synthetic = noise+shifts_xy
            bias = 1000*(baseline-TRUTH_S+np.einsum('ij,ijm->m',shifts_xy,gain))
            predicted = 1000*(baseline-TRUTH_S+np.einsum('nij,ijm->nm',synthetic,gain))
            actual = np.column_stack([archived[f'{motion}__{key}__{m}']
                                      for m in ('framewise_ekf','paired_ekf')])
            np.testing.assert_allclose(predicted,actual,atol=2.e-7,rtol=0)
            verification_error = float(np.max(np.abs(predicted-actual)))
            rmse = np.sqrt(bias*bias+sd*sd)
            shifts[key] = {'bias_ms':bias.tolist(),'sd_ms':sd.tolist(),'rmse_ms':rmse.tolist(),
                           'gain_ms':float(rmse[0]-rmse[1]),'gain_pct':float(100*(1-rmse[1]/rmse[0])),
                           'max_trial_reconstruction_error_ms':verification_error}
        common_gain = np.sqrt(np.mean([np.array(shifts[k]['rmse_ms'])**2
                                       for k in ('common_plus','common_minus')],axis=0))
        common_mc = np.sqrt(np.mean([np.array([run['motion_designs'][motion]['conditions'][k]['empirical'][m]['rmse_ms']
                                               for m in ('framewise_ekf','paired_ekf')])**2
                                     for k in ('common_plus','common_minus')],axis=0))
        result[motion] = {'shifts':shifts,'polarity_pooled_common_rmse_ms':common_gain.tolist(),
                          'polarity_pooled_common_empirical_rmse_ms':common_mc.tolist(),
                          'polarity_pooled_common_gain_pct':float(100*(1-common_gain[1]/common_gain[0])),
                          'polarity_pooled_common_empirical_gain_pct':float(100*(1-common_mc[1]/common_mc[0])),
                          'weights_native_xy':gain.tolist(),
                          'max_abs_y_gain_s_per_m':float(np.max(np.abs(gain[:,1,:])))}
    return result


def main():
    cfg = json.loads((REPRO/'inputs/filter_config.json').read_text(encoding='utf-8'))
    run = json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    with np.load(HERE/'signed_endpoint_errors_ms.npz',allow_pickle=False) as archive:
        result = calculate(cfg,archive,run)
    output = {'scope':'Affine approximation for the straight-leg query domains independently checked on every trial. Gaussian moments do not account for hypothetical crossings of these domains from unbounded noise tails.',
              'methods_in_order':['framewise_ekf','paired_ekf'],'motion_designs':result}
    (HERE/'terminal_affine_verification.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({motion:{k:v for k,v in rows.items() if k!='weights_native_xy'}
                      for motion,rows in result.items()},indent=2))


if __name__ == '__main__':
    main()
