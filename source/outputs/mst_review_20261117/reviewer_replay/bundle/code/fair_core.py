"""Portable fair-selection helpers; fitting is driven by run.py."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import portable_base as base
rep=base.rep;pd=base.pd;pe=base.pe
HERE=base.ROOT
REPRO=base.REPRO
def select_reason(rows,ref,model,cfg):
    records=[];last=-np.inf
    for row in sorted(rows,key=lambda r:r['bag_record_ns']):
        receive=row['bag_record_ns']*1e-9;why=None
        if receive-last<cfg['sensors']['radar']['sample_period_s']:why='sample_period_schedule'
        elif not np.isfinite([row['t'],receive,row['x'],row['y']]).all():why='nonfinite_observation'
        else:
            try:ref(row['t']);ref(row['t']-.09);ref(row['t']+.09)
            except base.UnsupportedReference:why='reference_candidate_margin'
        if why is None:last=receive
        records.append(dict(row_id=row['row_id'],selected=why is None,reason=why))
    return records

def full_mask(ref,t,axis,q,ids):
    mask=np.zeros(len(t),bool);breaks=np.flatnonzero(np.diff(ref.times)>.35);starts=np.r_[ref.times[0],ref.times[breaks+1]];ends=np.r_[ref.times[breaks],ref.times[-1]]
    for a,b in zip(starts,ends):mask|=(t-.6>a+.001)&(t+.6<b-.001)
    reasons={int(i):[] for i in ids}
    for i in ids[~mask]:reasons[int(i)].append('incomplete_continuous_search_interval')
    for b in np.arange(-.6,.6001,.01):
        _,v,ok=pe.reference_many(ref,t-b);proj=v@axis;good=ok&(np.sign(proj)==q)&(abs(proj)>=.1);mask&=good
        for j in np.flatnonzero(~good):
            why='reference_gap' if not ok[j] else 'direction_changed' if np.sign(proj[j])!=q[j] else 'speed_below_0p10'
            record=f'{why}@b={b:.2f}'
            if record not in reasons[int(ids[j])]:reasons[int(ids[j])].append(record)
    return mask,{str(k):v for k,v in reasons.items() if v}

def common_problem(events,ref,geometry,cfg,axis,q,mask):
    _,v,ok=pe.reference_many(ref,np.array([e.reported_s for e in events]));energy=float(np.sum((v[ok]@axis)**2)/np.sum(v[ok,:2]**2))
    support=dict(axis=axis.tolist(),direction=q.tolist(),mask=mask.tolist(),principal_energy_fraction=energy,selected_events=len(events),retained_events=int(mask.sum()),positive_events=int(sum(q[mask]>0)),negative_events=int(sum(q[mask]<0)),policy='Pre-fit intersection of whole-search support under both epoch conventions, same candidate events, frozen original axis and direction')
    return pe.ProfileProblem(events,ref,geometry,'radar',cfg['sensors']['radar']['sigma'],support=support)

def finish_state(replay,pairs,events,terminal):
    late=pairs[-1][1];epoch=events[late].receive_s;dt=terminal-epoch;assert dt>=-1e-7
    return {method:dict(last_pair_endpoint_id=late,last_update_receive_s=epoch,shared_terminal_receive_s=terminal,prediction_only_dt_s=dt,last_endpoint_offset_s=v[late]['offset_s'],drift_sps=v[late]['drift_sps'],shared_terminal_offset_s=v[late]['offset_s']+v[late]['drift_sps']*dt,interpretation='Offset propagated to the shared terminal using the unchanged filter constant-drift transition; no additional observation or covariance claim') for method,v in replay['outputs'].items()}

def prepare(name,cfg,geometry,identity):
    oldevents,oldpairs,oldref,oldrows=rep.load_recording(name,cfg,geometry,identity[name]);tracks=base.load(REPRO/'inputs'/f'recording_{name}.json');corrected,correction=base.corrected_rows(name,tracks);newref=base.TargetReference(corrected,cfg['reference_gap_s'],cfg['reference_smooth_half_window_s']);model=base.VehicleObservationModel(geometry)
    newevents,p,v=rep.select_observations(tracks['radar'],'radar',0,cfg,newref,model);newpairs=rep.match_pairs(newevents,p,v,'radar',cfg);rowmap={(r['t'],r['bag_record_ns']*1e-9):r['row_id'] for r in tracks['radar']};newrows=[rowmap[e.reported_s,e.receive_s] for e in newevents]
    native={}
    for events,rows in [(oldevents,oldrows),(newevents,newrows)]:
        for e,row in zip(events,rows):
            if row in native:assert e.reported_s==native[row].reported_s and e.receive_s==native[row].receive_s and np.array_equal(e.value,native[row].value)
            native[row]=e
    unionrows=sorted(native,key=lambda row:native[row].receive_s);events=[replace(native[row],id=i) for i,row in enumerate(unionrows)];index={row:i for i,row in enumerate(unionrows)}
    oldtuples={(oldrows[a],oldrows[b]) for a,b in oldpairs};newtuples={(newrows[a],newrows[b]) for a,b in newpairs}
    pairsets={label:sorted([(index[a],index[b]) for a,b in pairs],key=lambda p:p[1]) for label,pairs in [('old_frozen',oldtuples),('new_original_rules',newtuples),('identical_pair_intersection',oldtuples&newtuples)]}
    axis=np.asarray(identity[name]['axis']);times=np.array([e.reported_s for e in events]);_,vv,ok=pe.reference_many(oldref,times);direction=np.sign(vv@axis).astype(int)
    selected_common=sorted(set(oldrows)&set(newrows),key=lambda row:native[row].receive_s);common_events=[events[index[row]] for row in selected_common];ci=np.array([e.id for e in common_events]);ct=np.array([e.reported_s for e in common_events]);cq=direction[ci]
    hm,hr=full_mask(oldref,ct,axis,cq,ci);pm,pr=full_mask(newref,ct,axis,cq,ci);mask=hm&pm
    common={label:common_problem(common_events,ref,geometry,cfg,axis,cq,mask) for label,ref in [('header_epoch',oldref),('target_point_epoch',newref)]}
    member_union=sorted({i for pairs in pairsets.values() for pair in pairs for i in pair});prior=min(events[i].receive_s for i in member_union);terminal=max(events[late].receive_s for pairs in pairsets.values() for _,late in pairs)
    schedule={label:select_reason(tracks['radar'],ref,model,cfg) for label,ref in [('header_epoch',oldref),('target_point_epoch',newref)]}
    assert [r['row_id'] for r in schedule['header_epoch'] if r['selected']]==oldrows and [r['row_id'] for r in schedule['target_point_epoch'] if r['selected']]==newrows
    frozen=dict(source=name,correction=correction,union_radar_row_ids=unionrows,old_selected_radar_row_ids=oldrows,new_selected_radar_row_ids=newrows,only_header_selected_rows=sorted(set(oldrows)-set(newrows)),only_target_selected_rows=sorted(set(newrows)-set(oldrows)),selection_reason_records=schedule,
      pairsets={label:[dict(early_event_id=a,late_event_id=b,early_radar_row_id=unionrows[a],late_radar_row_id=unionrows[b]) for a,b in pairs] for label,pairs in pairsets.items()},prior_receive_s=prior,shared_terminal_receive_s=terminal,axis=axis.tolist(),direction=direction.tolist(),
      common_profile=dict(candidate_event_ids=ci.tolist(),candidate_radar_row_ids=selected_common,header_support_mask=hm.tolist(),target_support_mask=pm.tolist(),intersection_mask=mask.tolist(),retained_event_ids=ci[mask].tolist(),retained_radar_row_ids=[selected_common[j] for j in np.flatnonzero(mask)],header_rejection_reasons=hr,target_rejection_reasons=pr))
    return dict(frozen=frozen,events=events,pairsets=pairsets,oldref=oldref,newref=newref,model=model,common=common,native_new_problem=pe.ProfileProblem(newevents,newref,geometry,'radar',cfg['sensors']['radar']['sigma']))
