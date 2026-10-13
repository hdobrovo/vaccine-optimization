import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from age_model import initial_state, model, n, labels

window, final_day = 30, 350
gamma_values = np.logspace(-4, -1, 20)
order = (4, 3, 2, 1, 0)
y0 = initial_state_age()

panels = [('Hospitalizations', 7, 'Cumulative Hosp.', 'ro-'),
          ('Deaths', 6, 'Cumulative Deaths', 'ko-'),
          ('Infections', 5, 'Cumulative Infections', 'bo-'),
          ('Vaccinated', 3, 'People Vaccinated', 'go-')]

curves = {block: [] for _, block, _, _ in panels}
for gamma in gamma_values:
    y, t = y0.copy(), 0
    for grp in order:
        gvec = np.zeros(n)
        gvec[grp] = gamma
        y = odeint(model, y, np.linspace(t, t + window, 50), args=(gvec,))[-1]
        t += window
    y = odeint(model, y, np.linspace(t, final_day, 400), args=(np.zeros(n),))[-1]
    for _, block, _, _ in panels:
        curves[block].append(y[block * n:(block + 1) * n].sum())

plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.6))
for ax, (title, block, ylabel, style) in zip(axes.flat, panels):
    ax.loglog(gamma_values, curves[block], style, lw=1.2, ms=2.5)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('Vaccination Rate (γ)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(' → '.join(labels[i] for i in order), fontsize=9, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('age_65plus_first.png', dpi=300)
plt.close()
