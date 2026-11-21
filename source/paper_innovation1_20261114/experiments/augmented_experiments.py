"""Frozen geometry-augmented EKF controls for the reversal-calibration paper.

This is a conventional joint temporal/extrinsic EKF, used as a strong baseline.
It is not claimed to be a novel filter. No pair factor reuses raw observations.
Real-derived increments and trajectory-driven synthetic observations are separate.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REPO=ROOT/'MultiSensor_TimeSync'
sys.path[:0]=[str(REPO),str(REPO/'.deps')]
import numpy as np
from src.ekf_estimator.filter import TemporalEKF,gaussian_update
from src.ekf_estimator.paired import PairedTemporalEKF
from src.temporal_model.model import Observation,transition
from src.temporal_model.vehicle import TargetReference,VehicleObservationModel,UnsupportedReference,load_geometry
from experiments.paired_ekf import select_observations,match_pairs

METHODS=('ekf_all','ekf_same_observations','paired_ekf','augmented_all')
ALL_METHODS=('ekf_all','augmented_all')
BASE_CONFIG=ROOT/'experiment_20261114/innovation1/frozen_config.json'
FROZEN={'geometry_prior_lateral_std_m':.10,'geometry_prior_yaw_std_deg':2.,
        'geometry_process_noise':0.,'offset_increments_ms':[-60,-30,-10,10,30,60],
        'drift_increments_ms_per_s':[-.10,.10],
        'simulation_offset_ms':30.,'simulation_drift_ms_per_s':.10,
        'simulation_seeds':[1701,1702,1703,1704,1705,1706,1707,1708],
        'simulation_geometry_cases':['nominal','mixed_positive','mixed_negative'],
        'selection_policy':'Frozen previous extraction, selection and nonreused pairing; no test-set tuning',
        'metric_policy':'common pair endpoints across compared methods; separately all-event ekf_all vs augmented_all'}


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


class GeometryAugmentedEKF:
    """Independent 4-state [b,d,lateral,yaw(rad)] filter for one sensor."""
    def __init__(self,sensor,spec,filter_cfg,reference,geometry,base_lateral=0.,base_yaw_deg=0.):
        self.sensor,self.spec,self.cfg,self.reference=sensor,spec,filter_cfg,reference
        self.projector=VehicleObservationModel(geometry)
        self.base=np.array([base_lateral,np.deg2rad(base_yaw_deg)])
        self.x=np.zeros(4)
        self.P=np.diag([filter_cfg['initial_offset_std_s']**2,filter_cfg['initial_drift_std_sps']**2,
                        FROZEN['geometry_prior_lateral_std_m']**2,
                        np.deg2rad(FROZEN['geometry_prior_yaw_std_deg'])**2])
        self.last=None
        self.used=set()
        self.failed=0

    def value_jacobian(self,event,state=None):
        x=self.x if state is None else np.asarray(state)
        p,v=self.reference(event.reported_s-x[0])
        lateral,yaw=self.base+x[2:4]
        c,s=np.cos(yaw),np.sin(yaw)
        R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        center=self.projector.centers[self.sensor]
        rel=R@(p-center)
        q=center+rel+np.array([0.,lateral,0.])
        expected=self.projector._project_transformed(self.sensor,q)
        if self.sensor=='radar':
            J=np.eye(3)[:2]
        else:
            eps=1e-5
            J=np.column_stack([(self.projector._project_transformed(self.sensor,q+eps*e)-
                                self.projector._project_transformed(self.sensor,q-eps*e))/(2*eps)
                               for e in np.eye(3)])
        H=np.column_stack((-J@R@v,np.zeros(len(expected)),J@np.array([0.,1.,0.]),
                           J@np.array([-rel[1],rel[0],0.])))
        return expected,H

    def step(self,event):
        if event.id in self.used:
            raise ValueError('Each selected raw observation may be used only once')
        self.used.add(event.id)
        dt=0. if self.last is None else event.receive_s-self.last
        Ft,Qt=transition(dt,self.cfg['ekf_drift_density'],self.cfg['ekf_offset_density'])
        F=np.eye(4);F[:2,:2]=Ft
        Q=np.zeros((4,4));Q[:2,:2]=Qt
        self.x=F@self.x;self.P=F@self.P@F.T+Q;self.last=event.receive_s
        expected,H=self.value_jacobian(event)
        self.x,self.P,_=gaussian_update(self.x,self.P,event.value-expected,H,np.diag(np.square(self.spec['sigma'])))
        return {'offset_s':float(self.x[0]),'drift_sps':float(self.x[1]),
                'lateral_m':float(self.x[2]),'yaw_deg':float(np.rad2deg(self.x[3])),
                'offset_std_s':float(np.sqrt(self.P[0,0]))}


def run_filters(events,pairs,sensor,cfg,reference,geometry,case):
    model=VehicleObservationModel(geometry,case['lateral_m'],case['yaw_deg'])
    plain=TemporalEKF(cfg['sensors'],cfg['filter'],reference,model)
    same=TemporalEKF(cfg['sensors'],cfg['filter'],reference,model)
    pair=PairedTemporalEKF(cfg['sensors'],cfg['filter'],reference,model)
    augmented=GeometryAugmentedEKF(sensor,cfg['sensors'][sensor],cfg['filter'],reference,geometry,
                                   case['lateral_m'],case['yaw_deg'])
    members={i for p in pairs for i in p}
    ends={b:a for a,b in pairs}
    estimates={m:{} for m in METHODS}
    failures={m:0 for m in METHODS}
    for event in events:
        for name,estimator in [('ekf_all',plain),('augmented_all',augmented)]:
            try: estimates[name][event.id]=estimator.step(event)
            except UnsupportedReference: failures[name]+=1
        if event.id in members:
            try: estimates['ekf_same_observations'][event.id]=same.step(event)
            except UnsupportedReference: failures['ekf_same_observations']+=1
        if event.id in ends:
            try:
                estimates['paired_ekf'][event.id]=pair.step_pair(events[ends[event.id]],event,cfg['sensors'][sensor]['geometry_sigma'])
            except UnsupportedReference: failures['paired_ekf']+=1
    return {'estimates':estimates,'failures':failures,
            'all_input_events':len(events),'pair_endpoints':len(pairs),
            'augmented_final_x':augmented.x.tolist(),
            'augmented_final_min_eigenvalue':float(np.linalg.eigvalsh(augmented.P).min())}


def compare(runs,events,truth,kind,truth_drift=0.):
    result={}
    endpoint_ids=None
    for scope,methods in [('paired_endpoints',METHODS),('all_events',ALL_METHODS)]:
        ids=sorted(set.intersection(*[set(r['estimates'][m]) for r in runs for m in methods]))
        vals={}
        for m in methods:
            errors=[]
            drift_errors=[]
            for i in ids:
                values=[r['estimates'][m][i]['offset_s'] for r in runs]
                err=(values[0]-values[1] if len(values)==2 else values[0])-truth(i)
                errors.append(err)
                drift_values=[r['estimates'][m][i]['drift_sps'] for r in runs]
                drift_errors.append((drift_values[0]-drift_values[1] if len(drift_values)==2 else drift_values[0])-truth_drift)
            vals[m]={'rmse_ms':float(1000*np.sqrt(np.mean(np.square(errors)))) if ids else None,
                     'mae_ms':float(1000*np.mean(np.abs(errors))) if ids else None,
                     'bias_ms':float(1000*np.mean(errors)) if ids else None,
                     'drift_rmse_ms_per_s':float(1000*np.sqrt(np.mean(np.square(drift_errors)))) if ids else None}
        result[scope]={'common_count':len(ids),'common_ids':ids,
                       'receive_span_s':[events[ids[0]].receive_s,events[ids[-1]].receive_s] if ids else None,
                       'reported_span_s':[events[ids[0]].reported_s,events[ids[-1]].reported_s] if ids else None,
                       'methods':vals}
    return result


def information_diagnostic(events,sensor,cfg,reference,geometry):
    """Data-only local time information after projecting out geometry columns.

    For diagnosis only: no event selection, tuning, or estimator update uses it.
    Columns are scaled to b=10 ms, d=.1 ms/s, lateral=.1 m, yaw=2 degrees.
    """
    est=GeometryAugmentedEKF(sensor,cfg['sensors'][sensor],cfg['filter'],reference,geometry)
    rows=[]
    origin=events[0].receive_s
    for e in events:
        _,H=est.value_jacobian(e)
        H[:,1]=H[:,0]*(e.receive_s-origin)
        rows.append(H/np.asarray(cfg['sensors'][sensor]['sigma'])[:,None])
    A=np.vstack(rows)*np.array([.01,.0001,.1,np.deg2rad(2)])
    T,G=A[:,:2],A[:,2:]
    U,s,_=np.linalg.svd(G,full_matrices=False)
    rank=int(np.count_nonzero(s>max(G.shape)*np.finfo(float).eps*s[0])) if len(s) and s[0] else 0
    P=U[:,:rank]
    projected=T-P@(P.T@T)
    return {'definition':'whitened data-only time Jacobian projected orthogonal to local geometry nuisance; evaluated at zero b/eta, diagnostic only',
            'scaled_column_units':{'b_s':.01,'d_sps':.0001,'lateral_m':.1,'yaw_deg':2.},
            'geometry_rank':rank,'raw_time_singular_values':np.linalg.svd(T,compute_uv=False).tolist(),
            'geometry_projected_time_singular_values':np.linalg.svd(projected,compute_uv=False).tolist(),
            'all_state_singular_values':np.linalg.svd(A,compute_uv=False).tolist()}


def load_run(name):
    folder=ROOT/'paper_innovation1_20261114/new_tracks'/name
    if not (folder/'lidar_radar_targets.json').exists():
        folder=ROOT/'experiment_20261114/fresh_tracks'/name
    tracks=json.loads((folder/'lidar_radar_targets.json').read_text(encoding='utf-8'))
    camera=json.loads((folder/'camera_track.json').read_text(encoding='utf-8'))
    if tracks['epoch_ns']!=camera['epoch_ns']:raise ValueError('Epoch mismatch')
    return folder,tracks,camera


def do_run(name,quick=False):
    cfg=json.loads(BASE_CONFIG.read_text(encoding='utf-8'))
    calibration=ROOT/'experiment_20261114/innovation1/calibration'
    geometry=load_geometry(calibration)
    folder,tracks,camera=load_run(name)
    manifest={'run':name,'raw_bag':tracks['bag_file'],'data_source':'real independent sensor target extractions',
              'reference':'C16 header-time offline smoothed Hermite trajectory; not absolute time truth',
              'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [folder/'lidar_radar_targets.json',folder/'camera_track.json']},
              'sensors':{}}
    if len(tracks['lidar'])<3:
        manifest['status']='not_evaluable_no_lidar_target'
        save(OUT/'manifests'/f'{name}.json',manifest);return
    reference=TargetReference(tracks['lidar'],cfg['reference_gap_s'],cfg['reference_smooth_half_window_s'])
    nominal=VehicleObservationModel(geometry)
    cases=[{'name':'nominal','lateral_m':0.,'yaw_deg':0.}]+cfg['perturbations']
    for sensor in ['radar','camera']:
        rows=tracks['radar'] if sensor=='radar' else camera['camera']
        events,pos,slopes=select_observations(rows,sensor,tracks['epoch_ns'],cfg,reference,nominal)
        pairs=match_pairs(events,pos,slopes,sensor,cfg)
        ids=[i for p in pairs for i in p]
        assert len(ids)==len(set(ids))
        record={'extracted_count':len(rows),'selected_count':len(events),'pair_count':len(pairs),
                'pair_member_fraction':len(ids)/len(events) if events else 0,
                'selected_receive_span_s':[events[0].receive_s,events[-1].receive_s] if events else None,
                'pair_endpoint_span_s':[events[pairs[0][1]].receive_s,events[pairs[-1][1]].receive_s] if pairs else None}
        manifest['sensors'][sensor]=record
        if not events:
            continue
        record['local_information_diagnostic']=information_diagnostic(events,sensor,cfg,reference,geometry)
        base_runs={c['name']:run_filters(events,pairs,sensor,cfg,reference,geometry,c) for c in cases}
        real={'metadata':record,'metric_definition':'incremental offset error = estimate(injected)-estimate(original)-known increment; geometry sensitivity = estimate(perturbed)-estimate(nominal)',
              'geometry_sensitivity':{},'time_increments':{},'base_failures':{c:r['failures'] for c,r in base_runs.items()},
              'base_augmented_final':{c:r['augmented_final_x'] for c,r in base_runs.items()}}
        for case in cases[1:]:
            real['geometry_sensitivity'][case['name']]=compare([base_runs[case['name']],base_runs['nominal']],events,lambda i:0.,'geometry')
        injections=[{'name':f'offset_{x:g}ms','offset_s':x*1e-3,'drift_sps':0.} for x in ([-30,30] if quick else FROZEN['offset_increments_ms'])]
        if not quick:
            injections += [{'name':f'drift_{x:g}ms_per_s','offset_s':0.,'drift_sps':x*1e-3} for x in FROZEN['drift_increments_ms_per_s']]
        for inj in injections:
            deltas=np.array([inj['offset_s']+inj['drift_sps']*(e.receive_s-events[0].receive_s) for e in events])
            injected=[replace(e,reported_s=e.reported_s+deltas[e.id]) for e in events]
            comparison={}
            for case in ([cases[0]] if quick else cases):
                changed=run_filters(injected,pairs,sensor,cfg,reference,geometry,case)
                comparison[case['name']]={'metrics':compare([changed,base_runs[case['name']]],events,lambda i:deltas[i],'increment',inj['drift_sps']),
                                         'failures':changed['failures']}
            real['time_increments'][inj['name']]={'specification':inj,'comparisons':comparison}
        save(OUT/('quick' if quick else 'real_derived')/f'{name}_{sensor}.json',real)
        first=real['time_increments']['offset_30ms']['comparisons']['nominal']['metrics']['paired_endpoints']
        print(name,sensor,'pairs',len(pairs),'nominal +30ms',first['methods'],flush=True)
        if not quick:
            clean=[nominal(e,0.,cfg['sensors'][sensor],reference)[0] for e in events]
            truths=np.array([(FROZEN['simulation_offset_ms']+FROZEN['simulation_drift_ms_per_s']*(e.receive_s-events[0].receive_s))*1e-3 for e in events])
            simulation={'metadata':record,'data_kind':'C16_trajectory_driven_simulation; all non-LiDAR values regenerated; not real measurement evidence',
                        'truth':{'offset_ms':30.,'drift_ms_per_s':.10},'seeds':[]}
            for seed in FROZEN['simulation_seeds']:
                rng=np.random.default_rng(seed)
                replay=[replace(e,reported_s=e.reported_s+truths[e.id],value=z+rng.normal(0.,cfg['sensors'][sensor]['sigma'])) for e,z in zip(events,clean)]
                per_seed={'seed':seed,'cases':{}}
                for case in cases:
                    if case['name'] not in FROZEN['simulation_geometry_cases']:continue
                    run=run_filters(replay,pairs,sensor,cfg,reference,geometry,case)
                    per_seed['cases'][case['name']]={'metrics':compare([run],events,lambda i:truths[i],'absolute_simulated',FROZEN['simulation_drift_ms_per_s']*1e-3),
                                                   'failures':run['failures']}
                simulation['seeds'].append(per_seed)
            save(OUT/'trajectory_simulation'/f'{name}_{sensor}.json',simulation)
        save(OUT/'manifests'/f'{name}.json',manifest)


def flatten():
    rows=[]
    for p in sorted((OUT/'real_derived').glob('*.json')):
        run,sensor=p.stem.rsplit('_',1)
        d=json.loads(p.read_text(encoding='utf-8'))
        for geo,scopes in d['geometry_sensitivity'].items():
            for scope,metric in scopes.items():
                for method,value in metric['methods'].items():
                    rows.append({'run':run,'sensor':sensor,'source':'real_derived','metric':'geometry_sensitivity','condition':geo,'geometry':geo,'scope':scope,'method':method,'n':metric['common_count'],**value})
        for cond,block in d['time_increments'].items():
            for geo,comparison in block['comparisons'].items():
                for scope,metric in comparison['metrics'].items():
                    for method,value in metric['methods'].items():
                        rows.append({'run':run,'sensor':sensor,'source':'real_derived','metric':'increment_recovery','condition':cond,'geometry':geo,'scope':scope,'method':method,'n':metric['common_count'],**value})
    for p in sorted((OUT/'trajectory_simulation').glob('*.json')):
        run,sensor=p.stem.rsplit('_',1)
        d=json.loads(p.read_text(encoding='utf-8'))
        for seed in d['seeds']:
            for geo,block in seed['cases'].items():
                for scope,metric in block['metrics'].items():
                    for method,value in metric['methods'].items():
                        rows.append({'run':run,'sensor':sensor,'source':'trajectory_simulation','metric':'absolute_known_offset','condition':str(seed['seed']),'geometry':geo,'scope':scope,'method':method,'n':metric['common_count'],**value})
    if rows:
        with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',default=['08_cross_target_stop_reverse','05_stop_go_a','09_cross_target_holdout'])
    p.add_argument('--quick',action='store_true')
    p.add_argument('--flatten-only',action='store_true')
    args=p.parse_args()
    save(OUT/'frozen_config.json',{'base_config':json.loads(BASE_CONFIG.read_text(encoding='utf-8')),'augmentation':FROZEN,
                                  'base_config_sha256':hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest(),
                                  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if not args.flatten_only:
        for name in args.runs:
            started=time.perf_counter()
            do_run(name,args.quick)
            print('completed',name,'seconds',time.perf_counter()-started,flush=True)
    if not args.quick:flatten()


if __name__=='__main__':main()
