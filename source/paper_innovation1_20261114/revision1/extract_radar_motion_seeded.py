"""Motion-seeded same-radar tracking; preserves stopped parts of an active track.

Frozen before injected-time evaluation: seed speed>=0.15 m/s; track timeout
0.45 s; position jump <=0.28+2.5*dt m; prefer the prior ID; otherwise nearest
last-reported-velocity prediction with the same gate. Ambiguity uses the old
0.36 m near-best margin and 0.75 m lateral separation. Original ROI/background
and final centered same-stream temporal support are unchanged. No LiDAR
position or cross-sensor timing residual is consulted by this tracker.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import argparse,hashlib,json,sys,time

REVISION=Path(__file__).resolve().parent
sys.path.insert(0,str(REVISION))
import extract_radar_motion as motion
old,m,np=motion.old,motion.m,motion.np
PROTOCOL={'seed_min_recorded_xy_speed_mps':.15,'track_timeout_s':.45,
          'jump_slack_m':.28,'jump_speed_limit_mps':2.5,
          'ambiguity_near_best_margin_m':.36,'ambiguity_lateral_separation_m':.75,
          'prediction':'last accepted raw XY + last accepted recorded XY velocity * same-sensor header dt',
          'priority':'same ID within displacement gate, else closest prediction within displacement and prediction gate, else no association',
          'seed_policy':'frontmost moving foreground return with original lateral ambiguity rule',
          'selection_freeze':'specified before this source was run; no injected truth or LiDAR association error used',
          'postprocessing':'original centered temporal support applied after tracking'}

def association(rows,t,state,gates):
    """Choose from already ROI/background-filtered (position,velocity,obj) tuples."""
    dt=None if state is None else t-state['t']
    expired=state is None or dt>PROTOCOL['track_timeout_s']
    if not expired and dt<=0:return None,'nonincreasing_header',state,{}
    if expired:
        moving=sorted([r for r in rows if np.linalg.norm(r[1][:2])>=.15],key=lambda r:(float(r[0][0]),int(r[2].id)))
        if not moving:return None,'awaiting_moving_seed',None,{'timeout':state is not None}
        chosen=moving[0]
        if any(r[0][0]-chosen[0][0]<=gates.radar_ambiguity_x_margin_m
               and abs(r[0][1]-chosen[0][1])>gates.radar_ambiguity_y_separation_m for r in moving[1:]):
            return None,'ambiguous_seed',None,{'motion_candidates':len(moving)}
        return chosen,'motion_seed',None,{'motion_candidates':len(moving),'timeout':state is not None}
    gate=.28+2.5*dt
    prediction=state['p']+state['v']*dt
    close=[r for r in rows if np.linalg.norm(r[0][:2]-state['p'])<=gate]
    same=[r for r in close if int(r[2].id)==state['id']]
    if same:
        chosen=min(same,key=lambda r:np.linalg.norm(r[0][:2]-prediction))
        return chosen,'same_id_continue',state,{'dt_s':dt,'jump_gate_m':gate,'prediction_error_m':float(np.linalg.norm(chosen[0][:2]-prediction))}
    eligible=sorted([r for r in close if np.linalg.norm(r[0][:2]-prediction)<=gate],
                    key=lambda r:(float(np.linalg.norm(r[0][:2]-prediction)),int(r[2].id)))
    if not eligible:return None,'no_continuation_within_gate',state,{'dt_s':dt,'jump_gate_m':gate}
    chosen=eligible[0]
    best=float(np.linalg.norm(chosen[0][:2]-prediction))
    if any(np.linalg.norm(r[0][:2]-prediction)-best<=gates.radar_ambiguity_x_margin_m
           and abs(r[0][1]-chosen[0][1])>gates.radar_ambiguity_y_separation_m for r in eligible[1:]):
        return None,'ambiguous_id_change',state,{'dt_s':dt,'jump_gate_m':gate,'eligible':len(eligible)}
    return chosen,'nearest_prediction_id_change',state,{'dt_s':dt,'jump_gate_m':gate,'prediction_error_m':best}

def extract_seeded(conn,store,epoch,rotation,translation,gates):
    candidates,reasons,counts,trace=[],Counter(),Counter(),[]
    occupancies,background_frames=Counter(),0
    state,segment=None,-1
    for receive_ns,msg in old.records(conn,store,old._RADAR_TOPIC):
        counts['raw_frames']+=1
        stamp=old.header_ns(msg)
        if not stamp or msg.header.frame_id!='radar0_link':reasons['invalid_stamp_or_frame']+=1;continue
        if len(msg.targets)!=msg.radar_header.target_count:reasons['target_count_mismatch']+=1;continue
        rows=[r for r in old.radar_rows(msg,rotation,translation) if old._inside(r[0][0],gates.radar_x_m) and old._inside(r[0][1],gates.radar_y_m)]
        cells=old._grid(np.array([r[0][:2] for r in rows]),gates.radar_grid_m)
        if (receive_ns-epoch)*1e-9<gates.background_initial_s:
            occupancies.update(set(cells));background_frames+=1;reasons['background_initialization']+=1;continue
        if not background_frames:raise ValueError('No radar background frames')
        rows=[r for r,cell in zip(rows,cells) if occupancies[cell]/background_frames<gates.background_fraction]
        t=(stamp-epoch)*1e-9
        picked,decision,prior,details=association(rows,t,state,gates)
        counts[decision]+=1
        record={'header_ns':stamp,'t':t,'decision':decision,'foreground_candidate_count':len(rows),
                'prior_id':state['id'] if state else None,'prior_t':state['t'] if state else None,**details}
        if picked is None:
            state=prior
            reasons[decision]+=1
            trace.append(record);continue
        p,v,obj=picked
        if decision=='motion_seed':segment+=1
        state={'t':t,'p':p[:2].copy(),'v':v[:2].copy(),'id':int(obj.id)}
        stopped=float(np.linalg.norm(v[:2]))<.15
        counts['accepted_below_seed_speed']+=int(stopped)
        record.update(segment=segment,radar_id=int(obj.id),xy_m=p[:2].tolist(),speed_mps=float(np.linalg.norm(v[:2])),retained_stop=stopped)
        trace.append(record)
        candidates.append({'t':t,'header_ns':stamp,'bag_record_ns':receive_ns,'x':float(p[0]),'y':float(p[1]),'radar_id':int(obj.id),
            'quality':{'association':'motion_seeded_same_sensor_track_then_temporal_support','track_association_decision':decision,
                'derived_track_segment':segment,'retained_below_seed_speed':stopped,
                'device_stamp_us':int(msg.radar_header.device_stamp),'candidate_count':len(rows),
                'raw_longitudinal_m':float(obj.longitudinal_position),'raw_lateral_m':float(obj.lateral_position),
                'velocity_xy_mps':list(map(float,v[:2])),'rcs_dbsm':old._finite(obj.rcs),
                'move_status':int(obj.move_status),'maintain_status':int(obj.maintain_status),'track_age_frames':int(obj.track_age),
                'existence_probability':old._finite(obj.existence_probability),
                'position_std_codes':[int(obj.longitudinal_std_code),int(obj.lateral_std_code)]}})
    counts['before_temporal_support']=len(candidates)
    accepted=old._confirm(candidates,gates,reasons)
    counts['after_temporal_support']=len(accepted)
    accepted_headers={r['header_ns'] for r in accepted}
    for r in trace:r['accepted_after_temporal_support']=r['header_ns'] in accepted_headers
    counts['final_below_seed_speed']=sum(np.linalg.norm(r['quality']['velocity_xy_mps'])<.15 for r in accepted)
    return accepted,reasons,{k:int(v) for k,v in counts.items()},trace

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',default=motion.RUNS)
    parser.add_argument('--raw-root',type=Path,required=True)
    args=parser.parse_args()
    out=REVISION/'radar_motion_seeded_tracks';out.mkdir(exist_ok=True)
    manifest=[]
    for name in args.runs:
        start=time.perf_counter();folder,frozen,_=m.load_run(name)
        bag=args.raw_root/name/'bag/bag_0.db3';store=old.message_store(args.raw_root/name/'setup');gates=old.ExtractionGates()
        with old.open_bag(bag) as conn:
            epoch=old.bag_epoch(conn)
            if epoch!=frozen['epoch_ns']:raise ValueError('Epoch mismatch')
            R,tr=old.radar_to_lidar_transform(conn,store)
            rows,reasons,counts,trace=extract_seeded(conn,store,epoch,R,tr,gates)
        result=deepcopy(frozen);result['radar']=rows
        result['summary']['radar']=old._stream_summary(rows,reasons)
        result['radar_motion_seeded_revision']={'protocol':PROTOCOL,'counts':counts,'unchanged_rules':asdict(gates),
            'frozen_source':str(folder/'lidar_radar_targets.json'),'frozen_source_sha256':hashlib.sha256((folder/'lidar_radar_targets.json').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'old_extractor_sha256':hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
            'lidar_source':'copied exactly from frozen source','limitation':'ID can be reused; radar kinematics alone cannot establish cross-sensor target identity; recorded velocity and status are not ground truth'}
        result['association_scope']='motion-seeded same radar tracking, independent of C16 values/estimated time/injection truth'
        result['gate_provenance']='revision1 prespecified motion seed, timeout, same-ID then prediction gates plus old ROI/background and centered support'
        m.save(out/name/'lidar_radar_targets.json',result)
        m.save(out/name/'association_trace.json',trace)
        brief={'run':name,'radar_rows':len(rows),'counts':counts,'elapsed_s':time.perf_counter()-start}
        manifest.append(brief);m.save(out/'manifest.json',manifest)
        print(json.dumps(brief),flush=True)

if __name__=='__main__':main()
