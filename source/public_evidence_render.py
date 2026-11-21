"""Radar Ghost G01-C/left supplements drawn with the archived figure routines."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

WIDTH=16.5/2.54
COLORS={'geometry':'#245b8a','direction':'#b04c24','difference':'#b04c24','raw_completed_pairs':'#245b8a','raw_stream':'#747474'}
LABELS={'difference':'Difference EKF','raw_completed_pairs':'Framewise, same completed members','raw_stream':'Streaming framewise (auxiliary)'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.labelsize':10,'axes.titlesize':10,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'lines.linewidth':1.55,
 'axes.linewidth':.7,'savefig.facecolor':'white','pdf.fonttype':42,'ps.fonttype':42,
 'figure.constrained_layout.use':True})
def clean(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2,linewidth=.6)
 ax.tick_params(width=.7,length=3);ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
def save(fig, name, sources, description, extra=None):
    fig.savefig(output_dir / (name + ".png"), dpi=300)
    plt.close(fig)

def context(data):
 members={str(r['event_id']):r for r in data['frozen']['members']}
 ids=[str(p['late_id']) for p in data['frozen']['pairs']]
 times=np.array([members[i]['receive_s'] for i in ids])
 assert np.all(np.diff(times)>0)
 return ids,times
def offsets(data,case,injection,method,ids):
 runs=data['replays'][case][str(injection)]['outputs'][method]
 return np.array([runs[i]['offset_s'] for i in ids])*1000
def method_legend(methods):return [Line2D([],[],color=COLORS[m],lw=1.55,label=LABELS[m]) for m in methods]
def polarity_legend(unit='cm',amplitude=8):return [Line2D([],[],color='black',ls=ls,label=f'{sign}{amplitude} {unit}') for sign,ls in [('+','-'),('−','--')]]
def response(data,number,source,include_nominal=False,include_stream=False):
 ids,t=context(data);methods=['difference','raw_completed_pairs']+(['raw_stream'] if include_stream else [])
 base={m:offsets(data,'nominal',0,m,ids) for m in methods}
 n=3 if include_nominal else 2
 fig,axs=plt.subplots(n,1,figsize=(WIDTH,(18.0 if n==3 else 15.8)/2.54),sharex=True,layout='constrained')
 fig.get_layout_engine().set(rect=(0,0,1,.87 if include_stream else .90))
 idx=0
 if include_nominal:
  for m in methods:axs[0].plot(t,base[m],'.-',color=COLORS[m],ms=4)
  axs[0].set_title('(a) Nominal conditional offset',loc='left',pad=5);axs[0].set_ylabel('Offset estimate (ms)');idx=1
 for ax,kind,letter in zip(axs[idx:],['common','direction'],['b','c'] if include_nominal else ['a','b']):
  ax.axhline(0,color='black',lw=.6)
  for m in methods:
   for sign,style in [('plus','-'),('minus','--')]:
    y=offsets(data,kind+'_'+sign,0,m,ids)-base[m]
    ax.plot(t,y,style,color=COLORS[m],marker='o',ms=3,alpha=.65 if m=='raw_stream' else 1,lw=1.1 if m=='raw_stream' else 1.55)
    recorded=data['metrics'][kind+'_'+sign][m]['displacement_change']['rmse_ms']
    np.testing.assert_allclose(np.sqrt(np.mean(y*y)),recorded,rtol=0,atol=1e-10)
  ax.set_title(f'({letter}) '+('Common displacement' if kind=='common' else 'Direction-changing displacement'),loc='left',pad=5)
  ax.set_ylabel('Offset change (ms)')
 for ax in axs:clean(ax)
 axs[-1].set_xlabel('Completed-pair endpoint (s)')
 fig.legend(handles=method_legend(methods),loc='upper center',bbox_to_anchor=(.5,1.001),ncol=1,frameon=False)
 fig.legend(handles=polarity_legend(),loc='upper center',bbox_to_anchor=(.5,.915 if include_stream else .94),ncol=2,frameon=False)
 save(fig,f'figure_{number}_'+('ghost18_response' if source=='ghost18' else 'ghost01_response'),[source],f'{source}: same fixed pair endpoints; responses not absolute errors',{'methods':methods,'polarity':'solid +8 cm; dashed -8 cm','layout':f'{n} vertical panels','fixed_pairs':len(ids)})

def injection(data):
 ids,t=context(data);fig,axs=plt.subplots(2,1,figsize=(WIDTH,13.5/2.54),sharex=True,layout='constrained')
 fig.get_layout_engine().set(rect=(0,0,1,.925))
 for ax,m,letter in zip(axs,['difference','raw_completed_pairs'],['a','b']):
  base=offsets(data,'nominal',0,m,ids)
  for ms,c,style in [(-30,'#245b8a','--'),(30,'#b04c24','-')]:
   y=offsets(data,'nominal',ms,m,ids)-base-ms
   ax.plot(t,y,color=c,ls=style,marker='o',ms=3.2)
   np.testing.assert_allclose(np.sqrt(np.mean(y*y)),data['metrics']['nominal'][m]['recovery_error'][str(ms)]['rmse_ms'],rtol=0,atol=1e-10)
  ax.axhline(0,color='black',lw=.6);clean(ax);ax.set_ylabel('Increment recovery error (ms)')
  ax.set_title(f'({letter}) '+LABELS[m],loc='left',pad=6)
 axs[-1].set_xlabel('Completed-pair endpoint (s)')
 handles=[Line2D([],[],color='#245b8a',ls='--',marker='o',ms=3,label='−30 ms injection'),Line2D([],[],color='#b04c24',ls='-',marker='o',ms=3,label='+30 ms injection')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False)
 save(fig,'figure_5_injection_response',['ghost01'],'G01-C left nominal-coordinate timestamp-increment recovery',{'conditions':'nominal spatial coordinates; -30 and +30 ms reported-time injections','meaning':'estimated increment minus injection; no absolute-clock truth'})

def render(data, destination):
    global output_dir
    output_dir = destination
    response(data, 4, "ghost01", include_nominal=True)
    injection(data)
    return [destination / name for name in ("figure_4_ghost01_response.png", "figure_5_injection_response.png")]
