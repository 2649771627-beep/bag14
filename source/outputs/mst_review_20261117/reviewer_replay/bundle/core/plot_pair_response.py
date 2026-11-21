"""Plot the signed offset response at every fixed pair endpoint."""

import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHODS = {'raw_stream': ('Framewise EKF', '#596773', 1.05),
           'raw_completed_pairs': ('Framewise EKF, completed pairs', '#376bb2', 1.55),
           'difference': ('Difference EKF', '#af521d', 1.95)}


def plot(results):
    data = json.loads((results / 'experiment.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 5.9), sharex='col', sharey=True, layout='constrained')
    fig.get_layout_engine().set(rect=(0, 0, 1, .88))
    for col, (name, run) in enumerate(data['runs'].items()):
        frozen = run['frozen']
        receive = {str(row['event_id']): row['receive_s'] for row in frozen['members']}
        endpoints = [str(pair['late_id']) for pair in frozen['pairs']]
        time = [receive[i] for i in endpoints]
        gaps = np.flatnonzero(np.diff(time) > 2.) + 1
        segments = np.split(time, gaps)
        baseline = run['replays']['nominal']['0']['outputs']
        for row, kind in enumerate(('common', 'direction')):
            ax = axes[row, col]
            condition = 'Common displacement' if kind == 'common' else 'Direction-dependent displacement'
            ax.set_title(f'Bag {name[:2]} · {condition} · {len(endpoints)} pairs', fontsize=10, loc='left')
            ax.axhline(0., color='#7e7e7e', linewidth=.8, zorder=0)
            for method, (_, color, width) in METHODS.items():
                for polarity, linestyle in (('plus', '-'), ('minus', '--')):
                    outcomes = run['replays'][f'{kind}_{polarity}']['0']['outputs'][method]
                    shift = [1000 * (outcomes[i]['offset_s'] - baseline[method][i]['offset_s']) for i in endpoints]
                    for segment_t, segment_shift in zip(segments, np.split(shift, gaps)):
                        ax.plot(segment_t, segment_shift, color=color, linewidth=width, linestyle=linestyle,
                                alpha=.93, marker='o', markersize=2.2)
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(-145, 145)
        axes[1, col].set_xlabel('Pair endpoint receive time (s from bag epoch)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Change in offset (ms)')
    handles = [Line2D([0], [0], color=c, lw=w, label=label) for label, c, w in METHODS.values()]
    handles.extend(Line2D([0], [0], color='#3e3e3e', lw=1.3, linestyle=style, label=label)
                   for label, style in (('+8 cm', '-'), ('−8 cm', '--')))
    fig.suptitle('Time-offset response to displacement of fixed radar observations', y=.987, fontsize=14)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .946), ncol=5, frameon=False, fontsize=9)
    fig.savefig(results / 'offset_response.png', dpi=210)


