"""Offline robust temporal profiles with explicit geometric/feature nuisances.

The reference is fixed, not a jointly estimated target state (innovation 2).
Every retained observation is used once. No pair factor is appended.
No hardware clock truth is available for the recorded observations.
"""
from pathlib import Path
import json

import numpy as np

from src.temporal_model.vehicle import VehicleObservationModel

# Chosen from declared physical search/diagnostic scales before revised RMSE evaluation.
CONFIG={
 'offset_search_s':[-0.6,0.6], 'coarse_grid_step_s':0.01,
 'reference_margin_s':0.001, 'direction_min_speed_mps':0.10,
 'principal_energy_fraction_min':0.60, 'min_events':30,
 'min_per_direction':10, 'huber_delta_sigma':1.5, 'irls_iterations':50,
 'irls_tolerance':1e-6, 'max_translation_nuisance_m':0.30,
 'max_yaw_nuisance_deg':6., 'max_direction_center_m':0.30,
 'max_linearization_rms_sigma':0.25,
 'profile_near_min_cost_delta':2.0,
 'diagnostic_block_s':1.0, 'max_block_inflated_offset_std_s':0.05,
 'min_retained_temporal_information_fraction':0.02,
 'max_median_abs_residual_sigma':2.5,
 'max_outlier_fraction_3sigma':0.25,
 'boundary_margin_s':0.02,
 'max_split_disagreement_s':0.10,
 'nuisance_rank_rtol':1e-8,
 'drift_mode':'fixed_zero; only constant offset estimated in this revision',
 'interpretation':'engineering validity diagnostics, not calibrated confidence or physical-clock certification',
}
METHODS=('plain','geometry','direction')

def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def reference_many(ref,t):
 t=np.asarray(t);right=np.searchsorted(ref.times,t,side='right')
 valid=(right>0)&(right<len(ref.times))
 right=np.clip(right,1,len(ref.times)-1);left=right-1
 dt=ref.times[right]-ref.times[left];valid&=dt<=ref.max_gap
 u=(t-ref.times[left])/dt
 p0,p1=ref.positions[left],ref.positions[right]
 v0,v1=ref.velocities[left],ref.velocities[right]
 p=(2*u**3-3*u*u+1)[:,None]*p0+(u**3-2*u*u+u)[:,None]*dt[:,None]*v0+(-2*u**3+3*u*u)[:,None]*p1+(u**3-u*u)[:,None]*dt[:,None]*v1
 v=(6*u*u-6*u)[:,None]*(p0-p1)/dt[:,None]+(3*u*u-4*u+1)[:,None]*v0+(3*u*u-2*u)[:,None]*v1
 return p,v,valid

def project_many(model,sensor,p):
 center=model.centers[sensor]
 q=(p-center)@model.rotation.T+center+model.translation
 if sensor=='radar':return q[:,:2],np.ones(len(p),dtype=bool)
 cp=q@model.camera_from_lidar[:3,:3].T+model.camera_from_lidar[:3,3]
 a,b=cp[:,0]/cp[:,2],cp[:,1]/cp[:,2]
 radius=np.hypot(a,b);theta=np.arctan(radius)
 distorted=theta*(1+sum(d*theta**(2*(k+1)) for k,d in enumerate(model.D)))
 factor=np.divide(distorted,radius,out=np.ones_like(radius),where=radius>1e-12)
 return (model.K[0,0]*a*factor+model.K[0,2])[:,None],cp[:,2]>.1

def prediction_design(ref,model,sensor,t,offset,axis,direction,mode):
 p,v,valid=reference_many(ref,t-offset)
 y,front=project_many(model,sensor,p);valid&=front
 m=y.shape[1]
 if mode=='plain':return y,np.zeros((len(t)*m,0)),valid
 eps=1e-4
 J=np.stack([(project_many(model,sensor,p+eps*e)[0]-project_many(model,sensor,p-eps*e)[0])/(2*eps) for e in np.eye(3)],axis=-1)
 # Shared XY feature/geometry translation plus small yaw, both in LiDAR axes.
 # These nuisance coefficients are not reported as verified mounting calibration.
 relative=p-model.centers[sensor]
 yawvec=np.stack([-relative[:,1],relative[:,0],np.zeros(len(t))],axis=1)
 columns=[J[:,:,0],J[:,:,1],np.einsum('nmk,nk->nm',J,yawvec)]
 if mode=='direction':columns.append(np.einsum('nmk,k->nm',J,axis)*direction[:,None])
 return y,np.stack(columns,axis=-1).reshape(len(t)*m,-1),valid

def robust_profile(residual,G,delta=1.5,max_iter=50):
 """Eliminate linear nuisance by IRLS. Rows already whitened by declared sigma."""
 w=np.ones(len(residual));eta=np.zeros(G.shape[1])
 converged=False
 for iteration in range(max_iter):
  if G.shape[1]:eta=np.linalg.lstsq(G*np.sqrt(w)[:,None],residual*np.sqrt(w),rcond=CONFIG['nuisance_rank_rtol'])[0]
  rr=residual-G@eta
  new=np.minimum(1.,delta/np.maximum(np.abs(rr),1e-12))
  if np.max(np.abs(new-w))<CONFIG['irls_tolerance']:w=new;converged=True;break
  w=new
 absolute=np.abs(rr)
 rho=np.where(absolute<=delta,.5*rr*rr,delta*(absolute-.5*delta))
 return float(np.sum(rho)),eta,rr,w,{'converged':converged,'iterations':iteration+1}

class ProfileProblem:
 def __init__(self,events,reference,geometry,sensor,sigma,case=None,support=None):
  self.events=events;self.ref=reference;self.sensor=sensor;self.sigma=np.asarray(sigma)
  self.model=VehicleObservationModel(geometry,**(case or {}))
  self.t=np.array([e.reported_s for e in events]);self.r=np.array([e.receive_s for e in events])
  self.z=np.array([e.value for e in events])
  self.ids=np.array([e.id for e in events])
  if support is None:self.support=self.make_support()
  else:self.support=support
  self.axis=np.array(self.support['axis'])
  self.direction=np.array(self.support['direction'],dtype=float)
  self.mask=np.array(self.support['mask'],dtype=bool)

 def make_support(self):
  if not len(self.events):
   return {'axis':[1.,0.,0.],'principal_energy_fraction':0.,'direction':[],'mask':[],
           'selected_events':0,'retained_events':0,'positive_events':0,'negative_events':0}
  p,v,ok=reference_many(self.ref,self.t)
  if not ok.any():
   return {'axis':[1.,0.,0.],'principal_energy_fraction':0.,'direction':[0.]*len(self.t),'mask':[False]*len(self.t),
           'selected_events':len(self.t),'retained_events':0,'positive_events':0,'negative_events':0}
  _,S,V=np.linalg.svd(v[ok,:2],full_matrices=False)
  axis=np.r_[V[0],0.];axis*=1 if axis[np.argmax(np.abs(axis))]>0 else -1
  energy=float(S[0]**2/np.sum(S*S)) if np.sum(S*S)>0 else 0.
  dot=v@axis;q=np.sign(dot)
  low,high=CONFIG['offset_search_s']
  # Full continuous reference support, not merely valid grid samples across gaps.
  breaks=np.flatnonzero(np.diff(self.ref.times)>self.ref.max_gap)
  starts=np.r_[self.ref.times[0],self.ref.times[breaks+1]]
  ends=np.r_[self.ref.times[breaks],self.ref.times[-1]]
  mask=np.zeros(len(self.t),dtype=bool)
  for a,b in zip(starts,ends):
   mask|=(self.t-high>a+CONFIG['reference_margin_s'])&(self.t-low<b-CONFIG['reference_margin_s'])
  # Fixed conservative direction domain: retained sign is invariant throughout search.
  # Checking the 10-ms grid is a numerical screening rule, not a continuous proof.
  for b in np.arange(low,high+.001,CONFIG['coarse_grid_step_s']):
   _,vb,valid=reference_many(self.ref,self.t-b)
   projection=vb@axis
   mask&=valid&(np.sign(projection)==q)&(np.abs(projection)>=CONFIG['direction_min_speed_mps'])
  return {'axis':axis.tolist(),'principal_energy_fraction':energy,
          'direction':q.tolist(),'mask':mask.tolist(),
          'selected_events':len(self.t),'retained_events':int(mask.sum()),
          'positive_events':int(np.sum(mask&(q>0))),'negative_events':int(np.sum(mask&(q<0))),
          'policy':'common whole-search support, grid-checked direction stability, independent of measured values and injected truth'}

 def cost(self,b,mode,mask=None,details=False):
  mask=self.mask if mask is None else mask
  y,G,valid=prediction_design(self.ref,self.model,self.sensor,self.t[mask],b,self.axis,self.direction[mask],mode)
  if not valid.all():return (np.inf,None,None,None,None,{'converged':False}) if details else np.inf
  residual=((self.z[mask]-y)/self.sigma).ravel()
  Gw=G/np.tile(self.sigma,int(mask.sum()))[:,None]
  cost,eta,rr,w,inner=robust_profile(residual,Gw,CONFIG['huber_delta_sigma'],CONFIG['irls_iterations'])
  return (cost,eta,rr,w,Gw,inner) if details else cost

 def optimize(self,mode,mask=None):
  low,high=CONFIG['offset_search_s'];step=CONFIG['coarse_grid_step_s']
  grid=np.arange(low,high+.1*step,step)
  costs=np.array([self.cost(b,mode,mask) for b in grid]);best=int(np.argmin(costs))
  if not np.isfinite(costs).any():return None,grid,costs
  a,c=grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]
  ratio=(np.sqrt(5)-1)/2
  x1=c-ratio*(c-a);x2=a+ratio*(c-a)
  f1=self.cost(x1,mode,mask);f2=self.cost(x2,mode,mask)
  for _ in range(26):
   if f1<f2:c,x2,f2=x2,x1,f1;x1=c-ratio*(c-a);f1=self.cost(x1,mode,mask)
   else:a,x1,f1=x1,x2,f2;x2=a+ratio*(c-a);f2=self.cost(x2,mode,mask)
  b=(a+c)/2
  return b,grid,costs

 def fit(self,mode='direction',split_check=True):
  n=int(self.mask.sum());reasons=[]
  if n<CONFIG['min_events']:
   return {'accepted':False,'reasons':['insufficient_common_support'],'offset_s':None,'support':self.support,'mode':mode}
  b,grid,costs=self.optimize(mode)
  if b is None:return {'accepted':False,'reasons':['no_valid_profile'],'offset_s':None,'support':self.support,'mode':mode}
  cost,eta,rr,w,G,inner=self.cost(b,mode,details=True)
  if not inner['converged']:reasons.append('nuisance_solver_not_converged')
  near=np.flatnonzero(costs<=cost+CONFIG['profile_near_min_cost_delta'])
  if len(near)>1 and np.any(np.diff(near)>1):reasons.append('profile_multimodal')
  mask=self.mask;t=self.t[mask];q=self.direction[mask]
  eps=1e-4
  def full_prediction(offset):
   y,D,_=prediction_design(self.ref,self.model,self.sensor,t,offset,self.axis,q,mode)
   return (y.ravel()+D@eta)/np.tile(self.sigma,n)
  h=(full_prediction(b+eps)-full_prediction(b-eps))/(2*eps)
  A=G*np.sqrt(w)[:,None];hw=h*np.sqrt(w)
  if A.shape[1]:
   U,s,_=np.linalg.svd(A,full_matrices=False)
   rank=int(np.sum(s>CONFIG['nuisance_rank_rtol']*s[0])) if len(s) else 0
   hp=hw-U[:,:rank]@(U[:,:rank].T@hw)
  else:rank=0;hp=hw
  information=float(hp@hp);raw=float(hw@hw)
  fraction=information/max(raw,1e-30)
  r=self.r[mask];blocks=np.floor((r-r.min())/CONFIG['diagnostic_block_s']).astype(int)
  independent_blocks=len(np.unique(blocks))
  residual_scale=max(1.,float(1.4826*np.median(np.abs(rr-np.median(rr)))))
  std=float(residual_scale*np.sqrt(n/max(1,independent_blocks))/np.sqrt(information)) if information>1e-16 else None
  outlier=float(np.mean(np.abs(rr)>3));med=float(np.median(np.abs(rr)))
  if min(b-CONFIG['offset_search_s'][0],CONFIG['offset_search_s'][1]-b)<CONFIG['boundary_margin_s']:reasons.append('search_boundary')
  if std is None or std>CONFIG['max_block_inflated_offset_std_s']:reasons.append('weak_temporal_information')
  if fraction<CONFIG['min_retained_temporal_information_fraction']:reasons.append('temporal_nuisance_aliasing')
  if med>CONFIG['max_median_abs_residual_sigma'] or outlier>CONFIG['max_outlier_fraction_3sigma']:reasons.append('model_residual_mismatch')
  nonlinear_rms=0.
  if mode!='plain':
   if np.linalg.norm(eta[:2])>CONFIG['max_translation_nuisance_m'] or abs(eta[2])>np.deg2rad(CONFIG['max_yaw_nuisance_deg']):reasons.append('outside_local_geometry_model')
   if mode=='direction' and abs(eta[3])>CONFIG['max_direction_center_m']:reasons.append('outside_center_bias_model')
   p,_,_=reference_many(self.ref,t-b);center=self.model.centers[self.sensor]
   c,s=np.cos(eta[2]),np.sin(eta[2]);R=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
   exact=(p-center)@R.T+center+np.r_[eta[:2],0.]
   if mode=='direction':exact+=eta[3]*q[:,None]*self.axis
   exact_y,valid=project_many(self.model,self.sensor,exact)
   nominal,D,_=prediction_design(self.ref,self.model,self.sensor,t,b,self.axis,q,mode)
   difference=(exact_y.ravel()-nominal.ravel()-D@eta)/np.tile(self.sigma,n)
   nonlinear_rms=float(np.sqrt(np.mean(difference*difference))) if np.isfinite(difference).all() else None
   if not valid.all() or nonlinear_rms is None or nonlinear_rms>CONFIG['max_linearization_rms_sigma']:reasons.append('linearization_invalid')
  if mode=='direction':
   if self.support['principal_energy_fraction']<CONFIG['principal_energy_fraction_min']:reasons.append('no_dominant_motion_axis')
   if min(self.support['positive_events'],self.support['negative_events'])<CONFIG['min_per_direction']:reasons.append('insufficient_reversal_support')
  splits=[]
  if split_check:
   for parity in [0,1]:
    local=mask.copy();local[mask]=(blocks%2)==parity
    if int(local.sum())>=CONFIG['min_events']//2:
     sb=self.optimize(mode,local)[0]
     if sb is not None:splits.append(float(sb))
   if len(splits)<2:reasons.append('insufficient_split_support')
   elif abs(splits[0]-splits[1])>CONFIG['max_split_disagreement_s']:reasons.append('split_instability')
  # Direction-bias robustness comparison and temporal rank are data diagnostics.
  # Constant-offset output is not an estimate of native absolute clock truth.
  return {'mode':mode,'accepted':not reasons,'reasons':reasons,'offset_s':float(b),
          'drift_sps':0.,'drift_estimated':False,'robust_cost':cost,
          'nuisance_coefficients':eta.tolist(),'nuisance_rank':rank,
          'inner_solver':inner,'linearization_rms_sigma':nonlinear_rms,
          'median_abs_residual_sigma':med,'outlier_fraction_3sigma':outlier,
          'projected_information':information,'retained_information_fraction':fraction,
          'block_inflated_offset_std_s':std,'temporal_block_count':independent_blocks,
          'split_offsets_s':splits,'support':self.support,
          'profile_grid_s':grid.tolist(),'profile_cost':[float(c) if np.isfinite(c) else None for c in costs],
          'profile_candidate_valid':np.isfinite(costs).tolist(),
          'evaluated_receive_span_s':[float(r.min()),float(r.max())],
          'evaluated_header_span_s':[float(t.min()),float(t.max())],
          'output_interpretation':'conditional offline constant-offset fit; rejected fits retained for audit; diagnostic uncertainty is not a confidence interval'}

