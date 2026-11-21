"""Explicit, provenance-bound epoch conversion for the audited own C16 schema.

This never infers point-time units or header semantics from a field name. The
caller supplies the recorded-driver audit. It does not deskew coordinates,
reselect clusters, repair clocks, or change radar observations.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone

CONVENTION='audited_c16_scan_start_plus_cluster_median_seconds'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def convert(document, *, maximum_point_offset_s=0.11, header_tolerance_s=1e-9):
    if document.get('reference_epoch_conversion'):
        raise ValueError('Input already carries an epoch conversion; refusing a second application.')
    if not math.isfinite(maximum_point_offset_s) or maximum_point_offset_s<=0:
        raise ValueError('Maximum point offset must be positive and finite.')
    epoch=document['epoch_ns']
    if isinstance(epoch,bool) or not isinstance(epoch,int):
        raise ValueError('epoch_ns must be an integer nanosecond epoch.')
    rows=document['lidar']
    if not rows:raise ValueError('No retained LiDAR observations.')
    transformed=copy.deepcopy(document)
    ledger=[];previous_header=None;previous_target=None
    for index,(original,row) in enumerate(zip(rows,transformed['lidar'])):
        stamp=original['header_ns']
        if isinstance(stamp,bool) or not isinstance(stamp,int):
            raise ValueError(f'Row {index}: header_ns must be integer nanoseconds.')
        header=(stamp-epoch)*1e-9
        old=original['t'];dt=original.get('quality',{}).get('point_time_field_median_raw')
        if isinstance(old,bool) or not isinstance(old,(int,float)) or not math.isfinite(old):
            raise ValueError(f'Row {index}: invalid source trajectory time.')
        if abs(old-header)>header_tolerance_s:
            raise ValueError(f'Row {index}: source t does not equal header time; cannot apply this convention.')
        if isinstance(dt,bool) or not isinstance(dt,(int,float)) or not math.isfinite(dt):
            raise ValueError(f'Row {index}: missing or non-finite cluster point-time median.')
        if not 0<=dt<=maximum_point_offset_s:
            raise ValueError(f'Row {index}: point-time median is outside the explicitly declared seconds domain.')
        target=old+dt
        if previous_header is not None and not header>previous_header:
            raise ValueError(f'Row {index}: source header epochs are not strictly increasing.')
        if previous_target is not None and not target>previous_target:
            raise ValueError(f'Row {index}: target epochs are not strictly increasing; no sorting or dropping is allowed.')
        for name in ['x','y','z']:
            if not math.isfinite(original[name]):raise ValueError(f'Row {index}: non-finite coordinate.')
        row['t_header_s']=old
        row['t_target_s']=target
        row['point_time_median_s']=dt
        row['t']=target
        ledger.append({'source_row_index':index,'header_ns':stamp,'t_header_s':old,'point_time_median_s':dt,'t_target_s':target})
        previous_header=header;previous_target=target
    if transformed.get('radar')!=document.get('radar'):
        raise AssertionError('Radar observations changed.')
    for before,after in zip(rows,transformed['lidar']):
        restored=copy.deepcopy(after)
        restored['t']=restored.pop('t_header_s')
        restored.pop('t_target_s');restored.pop('point_time_median_s')
        if restored!=before:raise AssertionError('A field other than LiDAR time/explicit epoch metadata changed.')
    transformed['reference_epoch_conversion']={
        'convention':CONVENTION,
        'point_time_unit':'second',
        'maximum_point_offset_s':maximum_point_offset_s,
        'header_tolerance_s':header_tolerance_s,
        'rows':len(rows),
        'clock_quality':'HostEstimated according to supplied archived-driver audit; not physical clock truth',
        'coordinates_deskewed':False,
        'observations_reselected':False,
        'requires_downstream_pair_and_full_search_support_revalidation':True,
    }
    return transformed,ledger

def run(input_path, output_path, provenance_path, maximum_point_offset_s):
    source=Path(input_path).resolve();destination=Path(output_path).resolve();provenance=Path(provenance_path).resolve()
    if not source.is_file() or not provenance.is_file():raise ValueError('Input and driver-semantics audit must exist.')
    if source==destination:raise ValueError('Source overwrite is forbidden.')
    if destination.exists() or destination.with_suffix('.manifest.json').exists():
        raise FileExistsError('Destination or manifest already exists. Choose a fresh result path.')
    document=json.loads(source.read_text(encoding='utf-8-sig'))
    result,ledger=convert(document,maximum_point_offset_s=maximum_point_offset_s)
    result['reference_epoch_conversion'].update({'source_sha256':digest(source),'audit_sha256':digest(provenance)})
    destination.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with destination.open('x',encoding='utf8') as stream:stream.write(payload)
    manifest={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'input':str(source),'input_sha256':digest(source),
        'output':str(destination),'output_sha256':digest(destination),
        'provenance':str(provenance),'provenance_sha256':digest(provenance),
        'implementation_sha256':digest(__file__),
        'conversion':result['reference_epoch_conversion'],
        'radar_canonical_sha256_before':canonical_digest(document.get('radar')),
        'radar_canonical_sha256_after':canonical_digest(result.get('radar')),
        'lidar_source_row_ledger':ledger,
    }
    with destination.with_suffix('.manifest.json').open('x',encoding='utf8') as stream:
        json.dump(manifest,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--driver-semantics-audit',required=True)
    parser.add_argument('--convention',choices=[CONVENTION],required=True)
    parser.add_argument('--maximum-point-offset-s',type=float,default=.11)
    args=parser.parse_args()
    report=run(args.input,args.output,args.driver_semantics_audit,args.maximum_point_offset_s)
    print(json.dumps({k:report[k] for k in ['input_sha256','output_sha256','conversion']},ensure_ascii=False))
