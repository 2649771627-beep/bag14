"""Portable helpers from the verified own epoch branch."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT
REPRO=ROOT/'core'
sys.path.insert(0,str(REPRO))
import reproduce as rep
import paired_displacement as pd
import profile_estimator as pe
from src.temporal_model.vehicle import TargetReference,VehicleObservationModel,UnsupportedReference
RUNS={'08':'08_cross_target_stop_reverse','09':'09_cross_target_holdout'}
def load(p):return json.loads(Path(p).read_text(encoding='utf8'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,x):
    def cv(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.tolist()
        raise TypeError(type(v).__name__)
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=cv)+'\n',encoding='utf8')

def pair_audit(events,pairs,ref,model,cfg,axis):
    positions=[];slopes=[];event_failures=[];q=[]
    for i,e in enumerate(events):
        try:
            p,v=ref(e.reported_s);ref(e.reported_s-cfg['candidate_time_margin_s']);ref(e.reported_s+cfg['candidate_time_margin_s']);_,slope=model(e,0.,cfg['sensors']['radar'],ref)
            positions.append(p[:2]);slopes.append(slope);q.append(float(np.sign(v@axis)))
        except UnsupportedReference as exc:positions.append([np.nan,np.nan]);slopes.append([np.nan,np.nan]);q.append(0.);event_failures.append(dict(event_id=i,reason=str(exc)))
    positions=np.asarray(positions);slopes=np.asarray(slopes);speed=np.linalg.norm(slopes,axis=1);records=[]
    for early,late in pairs:
        gap=events[late].receive_s-events[early].receive_s;distance=float(np.linalg.norm(positions[late]-positions[early]));cosine=float(slopes[early]@slopes[late]/(speed[early]*speed[late]));ratio=float(min(speed[early],speed[late])/max(speed[early],speed[late]));reasons=[]
        if not np.isfinite([distance,cosine,ratio]).all():reasons.append('reference_candidate_support')
        else:
            if min(speed[early],speed[late])<cfg['sensors']['radar']['min_slope']:reasons.append('speed_floor')
            if distance>cfg['pair_max_distance_m']:reasons.append('position_distance')
            if cosine>cfg['pair_max_cosine']:reasons.append('opposite_velocity_cosine')
            if ratio<cfg['pair_min_speed_ratio']:reasons.append('speed_ratio')
        if not cfg['pair_min_gap_s']<=gap<=cfg['pair_max_gap_s']:reasons.append('receive_time_gap')
        records.append(dict(early_id=early,late_id=late,eligible=not reasons,reasons=reasons,position_distance_m=distance if np.isfinite(distance) else None,cosine=cosine if np.isfinite(cosine) else None,speed_ratio=ratio if np.isfinite(ratio) else None,early_speed_mps=float(speed[early]) if np.isfinite(speed[early]) else None,late_speed_mps=float(speed[late]) if np.isfinite(speed[late]) else None,receive_gap_s=gap))
    return dict(pair_count=len(pairs),eligible_pairs=sum(r['eligible'] for r in records),all_pairs_still_eligible=all(r['eligible'] for r in records),records=records,event_reference_failures=event_failures,directions=q)

def corrected_rows(name,compact):
    target=load(ROOT/'inputs'/f'recording_{name}_target_epoch.json')
    info=load(ROOT/'inputs/correction_metadata.json')[name]
    assert len(target['lidar'])==len(compact['lidar'])
    assert [[r[k] for k in ['x','y','z']] for r in target['lidar']]==[[r[k] for k in ['x','y','z']] for r in compact['lidar']]
    assert [r['t'] for r in target['lidar']]==info['corrected_relative_s']
    assert target['radar']==compact['radar']
    return target['lidar'],info
