
import itertools
import numpy as np
import matplotlib.pyplot as plt

from age_model import initial_state, run_ordering, n

window, final_day = 20, 150
gamma_values = np.logspace(-4, -1, 20)
ages = ['18-29', '30-49', '50-64', '65+']
y0 = initial_state()

panels = [('Hospitalizations', 7, 'Cumulative Hosp.', 'ro-'),
          ('Deaths', 6, 'Cumulative Deaths', 'ko-'),
          ('Infections', 5, 'Cumulative Infections', 'bo-'),
          ('Vaccinated', 3, 'People Vaccinated', 'go-')]

for order in itertools.permutations(range(4)):
    curves = {block: [] for _, block, _, _ in panels}
    for gamma in gamma_values:
        end = run_ordering(y0, order, gamma, window, final_day)
        for _, block, _, _ in panels:
            curves[block].append(end[block * n:(block + 1) * n].sum())

    fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.6))
    for ax, (title, block, ylabel, style) in zip(axes.flat, panels):
        ax.loglog(gamma_values, curves[block], style, lw=1.2, ms=2.5)
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xlabel('Vaccination Rate (γ)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3)

    name = ' → '.join(ages[i] for i in order)
    plt.suptitle(name, fontsize=9, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('age_' + '_'.join(ages[i] for i in order).replace('+', 'plus') + '.png', dpi=300)
    plt.close()
