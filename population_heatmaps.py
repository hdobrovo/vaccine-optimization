import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from model import model, initial_state

gamma_values = np.logspace(-6, 0, 20)
days = np.arange(20, 301, 20)
window = 5

def outcome_grid(proportions, order):
    y0 = initial_state(proportions)
    hosp = np.zeros((len(days), len(gamma_values)))
    deaths = np.zeros((len(days), len(gamma_values)))
    inf = np.zeros((len(days), len(gamma_values)))

    for gi, gamma in enumerate(gamma_values):
        ts, ys, y, t = [], [], y0, 0
        for grp in list(order) + [None]:
            stop = t + window if grp is not None else days.max()
            gvec = [0, 0, 0, 0]
            if grp is not None:
                gvec[grp] = gamma
            seg = odeint(model, y, np.linspace(t, stop, 60), args=(gvec,))
            ts.extend(np.linspace(t, stop, 60))
            ys.extend(seg)
            y, t = seg[-1], stop
        ts, ys = np.array(ts), np.array(ys)
        for di, day in enumerate(days):
            row = ys[np.argmin(np.abs(ts - day))]
            hosp[di, gi], deaths[di, gi], inf[di, gi] = row[12], row[10], row[9]
    return hosp, deaths, inf

def plot_ratios(order, name, fname):
    even = outcome_grid([0.25, 0.25, 0.25, 0.25], order)
    high = outcome_grid([0.125, 0.375, 0.125, 0.375], order)
    low = outcome_grid([0.375, 0.125, 0.375, 0.125], order)
    eps = 1e-10

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    rows = [(high, '75'), (low, '25')]
    cols = ['Hospitalizations', 'Deaths', 'Infections']
    for r, (case, pct) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(case[c] / (even[c] + eps), aspect='auto', origin='lower',
                           cmap='RdBu_r', vmin=0.5, vmax=1.5, interpolation='bilinear')
            ax.set_title(f'{cols[c]}\n({pct}% / 50% high risk)', fontsize=10, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Ratio')
            ax.set_xticks(np.arange(0, len(gamma_values), 4))
            ax.set_xticklabels([f'{g:.1e}' for g in gamma_values[::4]], rotation=45)
            ax.set_yticks(np.arange(0, len(days), 3))
            ax.set_yticklabels(days[::3])
            if c == 0:
                ax.set_ylabel('Days')
            if r == 1:
                ax.set_xlabel('Vaccination Rate (γ)')
    plt.tight_layout()
    plt.savefig(fname, format='pdf', bbox_inches='tight')
    plt.show()
    print(f'done: {name}')

plot_ratios((3, 2, 1, 0), 'best',  'ratio best.pdf')
plot_ratios((0, 1, 2, 3), 'worst', 'ratio worst.pdf')
plot_ratios((1, 0, 3, 2), 'mixed', 'ratio mixed.pdf')
