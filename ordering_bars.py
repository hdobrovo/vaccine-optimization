
import itertools
import numpy as np
import matplotlib.pyplot as plt

from model import initial_state, run_ordering

gamma = np.logspace(-6, 0, 20)[14]        # ~0.026 / day
window, final_day = 20, 150

groups = ['Sll', 'Slh', 'Shl', 'Shh']
orders = list(itertools.permutations(range(4)))
labels = ['→'.join(groups[i] for i in o) for o in orders]
y0 = initial_state()

hosp = np.zeros(len(orders))
deaths = np.zeros(len(orders))
inf = np.zeros(len(orders))
for k, order in enumerate(orders):
    end = run_ordering(y0, order, gamma, window, final_day)
    inf[k], deaths[k], hosp[k] = end[9], end[10], end[12]

for arr, ylabel, title, color in [
    (hosp, 'Cumulative Hospitalizations', 'Total Hospitalizations by Vaccination Order', 'firebrick'),
    (deaths, 'Cumulative Deaths', 'Total Deaths by Vaccination Order', 'dimgray'),
    (inf, 'Cumulative Infections', 'Total Infections by Vaccination Order', 'steelblue'),
]:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(np.arange(len(orders)), arr, color=color)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Vaccination Order')
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(orders)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"bar_{title.split()[1].lower()}.png", dpi=200)
    plt.show()
