"""Standalone own08/09 epoch-correction reproduction; no bag or network access."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import portable_base as base
import fair_core as fair
import prepare_target_epochs as adapter

def integrity():
    manifest=base.load(ROOT/'manifest.json')
    for relative,expected in manifest['package_sha256'].items():
        path=(ROOT/relative).resolve();assert path.is_relative_to(ROOT)
        if base.sha(path)!=expected:raise ValueError('Package checksum mismatch: '+relative)
    base.rep.check_integrity()
    modules={'reproduce':base.rep,'paired_displacement':base.pd,'profile_estimator':base.pe,'portable_base':base,'fair_core':fair,'epoch_adapter':adapter}
    modules.update({'vehicle':sys.modules[base.TargetReference.__module__],'temporal_filter':sys.modules[base.pd.TemporalEKF.__module__],'paired_filter':sys.modules[base.pd.PairedTemporalEKF.__module__]})
    paths={name:Path(module.__file__).resolve() for name,module in modules.items()}
    assert all(path.is_relative_to(ROOT) for path in paths.values())
    return {name:path.relative_to(ROOT).as_posix() for name,path in paths.items()}

def offsets_equal(actual,expected):
    maximum=0.
    for method,values in actual.items():
        assert set(map(str,values))==set(expected[method])
        for i,r in values.items():
            for key in ['offset_s','drift_sps','offset_std_s','prior_offset_s']:
                maximum=max(maximum,abs(r[key]-expected[method][str(i)][key]))
    assert maximum<1e-8,maximum
    return maximum

def fits_equal(actual,expected):
    maximum=0.
    for method,fit in actual.items():
        old=expected[method];assert fit['accepted']==old['accepted'] and fit['reasons']==old['reasons'];assert fit['support']['retained_events']==old['support']['retained_events']
        if 'profile_cost' in fit:
            np.testing.assert_allclose(fit['profile_grid_s'],old['profile_grid_s'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(fit['profile_cost'],old['profile_cost'],atol=1e-7,rtol=1e-6);maximum=max(maximum,float(np.max(abs(np.array(fit['profile_cost'])-old['profile_cost']))))
            assert abs(fit['offset_s']-old['offset_s'])<1e-5
            np.testing.assert_allclose(fit['block_inflated_offset_std_s'],old['block_inflated_offset_std_s'],rtol=1e-4,atol=1e-5)
    return maximum

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('replayed_results'));parser.add_argument('--matrix',action='store_true',help='Also replay all5space×3time conditions on identical-pair intersections');parser.add_argument('--figures',action='store_true');parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    module_paths=integrity()
    if args.check_only:print('Package hashes and local module isolation verified.');return
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cfg=base.load(base.REPRO/'inputs/filter_config.json');geometry=(np.eye(4),np.array(base.load(base.REPRO/'inputs/radar_from_lidar.json')),np.eye(3),np.zeros(4));identity=base.load(base.REPRO/'inputs/event_identity.json');gold=base.load(ROOT/'inputs/expected_own.json');runs={};verification={}
    for name in base.RUNS:
        source=base.load(ROOT/'inputs'/f'epoch_source_{name}.json');converted,ledger=adapter.convert(source);target=base.load(ROOT/'inputs'/f'recording_{name}_target_epoch.json')
        assert [r['t'] for r in converted['lidar']]==[r['t'] for r in target['lidar']]
        assert converted['radar']==source['radar']==target['radar']
        assert [[r[k] for k in ['x','y','z']] for r in converted['lidar']]==[[r[k] for k in ['x','y','z']] for r in target['lidar']]
        events,pairs,oldref,selected=base.rep.load_recording(name,cfg,geometry,identity[name]);newref=base.TargetReference(target['lidar'],cfg['reference_gap_s'],cfg['reference_smooth_half_window_s']);frozen,members=base.pd.freeze_events(events,pairs,oldref,selected,identity[name]);model=base.VehicleObservationModel(geometry)
        fixed={};checks={'exact_epoch_conversion_rows':len(ledger),'fixed_nominal_max_state_error':{},'fair_nominal_max_state_error':{},'profile_max_cost_error':{}}
        for epoch,ref in [('header_epoch',oldref),('target_point_epoch',newref)]:
            replay=base.pd.replay(members,pairs,cfg,ref,model,frozen['prior_receive_s']);checks['fixed_nominal_max_state_error'][epoch]=offsets_equal(replay['outputs'],gold['fixed'][name][epoch]['outputs']);assert replay['reference_failures']==gold['fixed'][name][epoch]['reference_failures'];fixed[epoch]=replay
        prepared=fair.prepare(name,cfg,geometry,identity);pf=prepared['frozen'];gf=gold['fair'][name]['frozen']
        for key in ['pairsets','old_selected_radar_row_ids','new_selected_radar_row_ids','prior_receive_s','shared_terminal_receive_s','axis','direction','common_profile']:assert pf[key]==gf[key],(name,key)
        fair_nominal={};profiles={}
        for label,pairlist in prepared['pairsets'].items():
            keys=sorted(i for pair in pairlist for i in pair);observations={i:prepared['events'][i] for i in keys};fair_nominal[label]={};checks['fair_nominal_max_state_error'][label]={}
            for epoch,ref in [('header_epoch',prepared['oldref']),('target_point_epoch',prepared['newref'])]:
                replay=base.pd.replay(observations,pairlist,cfg,ref,prepared['model'],pf['prior_receive_s']);expected=gold['fair'][name]['nominal'][label][epoch];checks['fair_nominal_max_state_error'][label][epoch]=offsets_equal(replay['outputs'],expected['replay']['outputs']);terminal=fair.finish_state(replay,pairlist,prepared['events'],pf['shared_terminal_receive_s'])
                for method,state in terminal.items():assert abs(state['shared_terminal_offset_s']-expected['terminal_states'][method]['shared_terminal_offset_s'])<1e-8
                fair_nominal[label][epoch]=dict(replay=replay,terminal_states=terminal)
        for epoch,problem in prepared['common'].items():
            fits={mode:problem.fit(mode) for mode in ['geometry','direction']};checks['profile_max_cost_error'][epoch]=fits_equal(fits,gold['fair'][name]['common_profiles'][epoch]['fits']);profiles[epoch]=fits
        newfits={mode:prepared['native_new_problem'].fit(mode) for mode in ['geometry','direction']};checks['profile_max_cost_error']['new_native']=fits_equal(newfits,gold['fair'][name]['new_native_support_profile']['fits'])
        matrix={}
        if args.matrix:
            pairlist=prepared['pairsets']['identical_pair_intersection'];keys=sorted(i for pair in pairlist for i in pair);observations={i:prepared['events'][i] for i in keys};mini=dict(axis_lidar_xy=pf['axis'][:2],members=[dict(event_id=i,direction=pf['direction'][i]) for i in keys])
            for epoch,ref in [('header_epoch',prepared['oldref']),('target_point_epoch',prepared['newref'])]:
                matrix[epoch]={}
                for case,(amplitude,directional) in base.pd.SHIFTS.items():
                    matrix[epoch][case]={}
                    for delta in [0,-30,30]:
                        modified=base.pd.perturb(observations,pairlist,mini,amplitude,directional,delta);replay=base.pd.replay(modified,pairlist,cfg,ref,prepared['model'],pf['prior_receive_s']);terminal=fair.finish_state(replay,pairlist,prepared['events'],pf['shared_terminal_receive_s']);matrix[epoch][case][str(delta)]=dict(replay=replay,terminal_states=terminal)
                        expected=gold['matrix_terminal'][name][epoch][case][str(delta)]
                        for method,state in terminal.items():assert abs(state['shared_terminal_offset_s']-expected[method]['shared_terminal_offset_s'])<1e-8
            checks['all_matrix_terminal_offsets_verified']=True
        runs[name]=dict(frozen=pf,fixed_nominal=fixed,fair_nominal=fair_nominal,common_profiles=profiles,new_native_profile=newfits,matrix=matrix);verification[name]=checks
        print(name,'verified:',len(pairs),'original pairs;',len(prepared['pairsets']['new_original_rules']),'reselected;',len(prepared['pairsets']['identical_pair_intersection']),'identical pairs;',len(pf['common_profile']['retained_event_ids']),'common profile observations.',flush=True)
    result=dict(runs=runs,verification=verification,runtime={'python':sys.version,'numpy':np.__version__,'elapsed_s':time.perf_counter()-start,'local_module_paths':module_paths},matrix_evaluated=args.matrix,original_position_policy_unchanged=True,innovation2_changed=False);base.save(output/'results.json',result);write_report(result,output)
    if args.figures:figures(result,output)
    print('All requested reproductions matched the saved results. Output:',output)

def write_report(result,output):
    lines=['# Standalone own08/09 replay verified','', 'No bag, raw video, workspace code or network access was used. The exported target epochs were independently regenerated by the supplied adapter. All copied-core hashes, frame identities, pairsets, shared-support rows, terminal offsets and profile decisions matched the saved results.','', '|Recording|Epoch convention|Model|Same observations|Offset(ms)|Engineering scale(ms)|Accepted|Reasons|','|---|---|---|---:|---:|---:|---|---|']
    for name,r in result['runs'].items():
        for epoch,fits in r['common_profiles'].items():
            for method,f in fits.items():lines.append(f"|{name}|{epoch}|{method}|{f['support']['retained_events']}|{1000*f['offset_s']:.3f}|{1000*f['block_inflated_offset_std_s']:.3f}|{f['accepted']}|{', '.join(f['reasons'])}|")
    lines+=['','An estimated offset numerically below50ms is not a50ms-gate pass: that gate concerns an engineering identifiability scale. Direction-centre profiles remain rejected. The target-point epoch is a scan-phase representative without deskewing or independently certified physical clock truth.','',f"Full identical-pair perturbation matrix replayed: {result['matrix_evaluated']}. Runtime: {result['runtime']['elapsed_s']:.2f}s."]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def figures(result,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,(name,r) in zip(axes,result['runs'].items()):
        for epoch,fits in r['common_profiles'].items():
            for method,f in fits.items():
                c=np.array(f['profile_cost']);ax.plot(np.array(f['profile_grid_s'])*1000,c-c.min(),label=epoch+' '+method)
        ax.set(xlabel='Offset (ms)',ylabel='Cost above own minimum',title='Own'+name+': identical common observations');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(output/'common_profiles.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
