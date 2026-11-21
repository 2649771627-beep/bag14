"""Read-only radar candidate extraction with a fixed motion eligibility rule.

Preserves the frozen C16 trajectory verbatim. ROI, background occupancy,
ambiguity and same-stream temporal support are exactly the old extractor's.
Only radar candidates with recorded XY velocity norm >=0.15 m/s are eligible
for frontmost selection. Neither cross-sensor residuals nor clock injections
participate in association. Outputs are separate from all previous sources.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib,json,sys,time,argparse

REVISION=Path(__file__).resolve().parent
PAPER=REVISION.parent
sys.path.insert(0,str(PAPER/'experiments'))
import augmented_experiments as m
from src.data_loader import lidar_radar_targets as old
np=m.np
RUNS=['01_cross_slow','03_range_slow','04_range_normal','05_stop_go_a','06_stop_go_b',
      '08_cross_target_stop_reverse','09_cross_target_holdout']
MIN_SPEED=.15

def extract_motion(connection,store,epoch,rotation,translation,gates):
    candidates,reasons=[],Counter()
    occupancies,background_frames=Counter(),0
    counts=Counter()
    for receive_ns,message in old.records(connection,store,old._RADAR_TOPIC):
        counts['raw_frames']+=1
        stamp=old.header_ns(message)
        if not stamp or message.header.frame_id!='radar0_link':
            reasons['invalid_stamp_or_frame']+=1;continue
        if len(message.targets)!=message.radar_header.target_count:
            reasons['target_count_mismatch']+=1;continue
        rows=[(p,v,obj) for p,v,obj in old.radar_rows(message,rotation,translation)
              if old._inside(p[0],gates.radar_x_m) and old._inside(p[1],gates.radar_y_m)]
        cells=old._grid(np.array([r[0][:2] for r in rows]),gates.radar_grid_m)
        if (receive_ns-epoch)*1e-9<gates.background_initial_s:
            occupancies.update(set(cells));background_frames+=1
            reasons['background_initialization']+=1;continue
        if not background_frames:raise ValueError('No radar background frames')
        rows=[row for row,cell in zip(rows,cells) if occupancies[cell]/background_frames<gates.background_fraction]
        counts['foreground_candidates']+=len(rows)
        if not rows:
            reasons['no_nearby_candidate']+=1;continue
        rows.sort(key=lambda row:(float(row[0][0]),int(row[2].id)))
        moving=[row for row in rows if np.linalg.norm(row[1][:2])>=MIN_SPEED]
        counts['motion_candidates']+=len(moving)
        counts['below_motion_threshold_candidates']+=len(rows)-len(moving)
        counts['foreground_frames']+=1
        if moving:
            counts['frames_with_motion_candidate']+=1
            if np.linalg.norm(rows[0][1][:2])<MIN_SPEED:
                counts['frames_original_frontmost_static_but_motion_candidate_present']+=1
        if not moving:
            reasons['no_motion_eligible_candidate']+=1;continue
        p,v,obj=moving[0]
        if any(alt[0][0]-p[0]<=gates.radar_ambiguity_x_margin_m
               and abs(alt[0][1]-p[1])>gates.radar_ambiguity_y_separation_m for alt in moving[1:]):
            reasons['ambiguous_crossing_returns']+=1;continue
        candidates.append({'t':(stamp-epoch)*1e-9,'header_ns':stamp,'bag_record_ns':receive_ns,
            'x':float(p[0]),'y':float(p[1]),'radar_id':int(obj.id),
            'quality':{'association':'motion_eligible_frontmost_same_sensor_roi_then_temporal_support',
                'motion_eligibility_speed_mps':float(np.linalg.norm(v[:2])),
                'minimum_recorded_xy_speed_mps':MIN_SPEED,
                'device_stamp_us':int(message.radar_header.device_stamp),
                'candidate_count':len(moving),'foreground_candidate_count':len(rows),
                'raw_longitudinal_m':float(obj.longitudinal_position),'raw_lateral_m':float(obj.lateral_position),
                'velocity_xy_mps':list(map(float,v[:2])),'rcs_dbsm':old._finite(obj.rcs),
                'move_status':int(obj.move_status),'maintain_status':int(obj.maintain_status),
                'track_age_frames':int(obj.track_age),'existence_probability':old._finite(obj.existence_probability),
                'position_std_codes':[int(obj.longitudinal_std_code),int(obj.lateral_std_code)]}})
    counts['before_temporal_support']=len(candidates)
    accepted=old._confirm(candidates,gates,reasons)
    counts['after_temporal_support']=len(accepted)
    return accepted,reasons,dict(counts)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',default=RUNS)
    p.add_argument('--raw-root',type=Path,required=True)
    args=p.parse_args()
    out=REVISION/'radar_motion_tracks'
    out.mkdir(exist_ok=True)
    manifest=[]
    for name in args.runs:
        started=time.perf_counter()
        folder,frozen,_=m.load_run(name)
        bag=args.raw_root/name/'bag/bag_0.db3'
        store=old.message_store(args.raw_root/name/'setup')
        gates=old.ExtractionGates()
        with old.open_bag(bag) as conn:
            epoch=old.bag_epoch(conn)
            if epoch!=frozen['epoch_ns']:raise ValueError('Epoch mismatch')
            rotation,translation=old.radar_to_lidar_transform(conn,store)
            rows,reasons,counts=extract_motion(conn,store,epoch,rotation,translation,gates)
        result=deepcopy(frozen)
        result['radar']=rows
        result['summary']['radar']=old._stream_summary(rows,reasons)
        result['radar_motion_revision']={'minimum_recorded_xy_speed_mps':MIN_SPEED,
            'changed_rule':'after existing ROI/background eligibility, exclude speed <0.15 before frontmost/ambiguity and original temporal support',
            'unchanged_rules':asdict(gates),'counts':counts,
            'frozen_source':str(folder/'lidar_radar_targets.json'),
            'frozen_source_sha256':hashlib.sha256((folder/'lidar_radar_targets.json').read_bytes()).hexdigest(),
            'lidar_source':'copied exactly from frozen source, no LiDAR extraction performed',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'old_extractor_sha256':hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
            'velocity_caveat':'recorded target XY velocity is unvalidated; candidate eligibility is not ground-truth association',
            'no_truth_tuning':'fixed threshold announced before any injection ablation; no cross-sensor residual or injected clock used for extraction'}
        result['association_scope']='independent radar candidate velocity/ROI/background/frontmost/ambiguity/temporal support; no cross-sensor residual or aligned timestamp'
        result['gate_provenance']='old development-08 geometry/support rules plus predeclared revision1 speed>=0.15m/s; all seven runs retained'
        m.save(out/name/'lidar_radar_targets.json',result)
        summary={'run':name,'bag':str(bag),'radar_rows':len(rows),'lidar_rows':len(result['lidar']),
            'counts':counts,'elapsed_s':time.perf_counter()-started}
        manifest.append(summary)
        m.save(out/'manifest.json',manifest)
        print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
