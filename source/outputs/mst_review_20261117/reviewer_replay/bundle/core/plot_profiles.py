"""Render the six recorded objective profiles and a view around the minima."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'08_near_constant': '08  Near-constant speed', '08_variable': '08  Variable speed',
          '09_full': '09  Whole recording'}
MODES = {'geometry': ('Shared XY and yaw', '#24579b'),
         'direction': ('Shared XY and yaw + direction centre', '#b75124')}


def plot(results):
    data = json.loads((results / 'comparison.json').read_text(encoding='utf-8'))['groups']
    output = results / 'cost_profiles.png'
    if output.exists():
        raise FileExistsError(output)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3), sharex=True, sharey='row', layout='constrained')
    for col, (name, title) in enumerate(LABELS.items()):
        group = data[name]
        axes[0, col].set_title(f'{title}\nN = {len(group["group"]["ids"])} radar observations')
        for mode, (label, color) in MODES.items():
            fit = group['fits'][mode]
            x = np.asarray(fit['profile_grid_s']) * 1000
            delta = np.asarray(fit['profile_cost'], dtype=float) - fit['robust_cost']
            for ax in axes[:, col]:
                ax.plot(x, delta, label=label, color=color, lw=1.7)
                ax.plot(fit['offset_s'] * 1000, 0, marker='o' if fit['accepted'] else 'x',
                        color=color, ms=5, clip_on=False, linestyle='none')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            ax.set_xlim(-600, 600)
        axes[0, col].set_ylim(-5, 485)
        axes[1, col].set_ylim(-.2, 8.)
        axes[1, col].axhline(2, color='#616161', ls=':', lw=1)
        axes[1, col].set_xlabel('Candidate time offset (ms)')
    axes[0, 0].set_ylabel('Cost above fitted minimum\nFull range')
    axes[1, 0].set_ylabel('Cost above fitted minimum\nDetail near minima')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    fig.supxlabel('Circle = accepted fit. Cross = rejected fit. Dotted line = cost rise of 2.', fontsize=9)
    with output.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=200)
    plt.close(fig)
    print(output)


