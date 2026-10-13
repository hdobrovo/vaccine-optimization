import itertools
import numpy as np
import matplotlib.pyplot as plt

from model import initial_state, run_ordering

window, final_day = 30, 350
gamma_values = np.logspace(-4, -1, 20)
groups = ['Sll', 'Slh', 'Shl', 'Shh']
y0 = initial_state()

panels = [('Hospitalizations', 12, 'Cumulative Hosp.', 'ro-'),
          ('Deaths', 10, 'Cumulative Deaths', 'ko-'),
          ('Infections', 9, 'Cumulative Infections', 'bo-'),
          ('Vaccinated', 7, 'People Vaccinated', 'go-')]

for order in itertools.permutations(range(4)):
    curves = {idx: [] for _, idx, _, _ in panels}
    for gamma in gamma_values:
        end = run_ordering(y0, order, gamma, window, final_day)
        for _, idx, _, _ in panels:
            curves[idx].append(end[idx])

    fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.6))
    for ax, (title, idx, ylabel, style) in zip(axes.flat, panels):
        ax.loglog(gamma_values, curves[idx], style, lw=1.2, ms=2.5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Vaccination Rate (γ)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    name = '→'.join(groups[i] for i in order)
    plt.suptitle(name, fontsize=9, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('perm_' + '_'.join(groups[i] for i in order).lower() + '.png', dpi=300)
    plt.close()
