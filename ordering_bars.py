import itertools
import numpy as np
import matplotlib.pyplot as plt

from model import initial_state, run_ordering

gamma = np.logspace(-6, 0, 20)[14]
final_day = 350
windows = [14, 30]

groups = ['Sll', 'Slh', 'Shl', 'Shh']
orders = list(itertools.permutations(range(4)))
labels = ['→'.join(groups[i] for i in o) for o in orders]
x = np.arange(len(orders))
y0 = initial_state()

results = {w: {9: [], 10: [], 12: []} for w in windows}
for w in windows:
    for order in orders:
        end = run_ordering(y0, order, gamma, w, final_day)
        for idx in (9, 10, 12):
            results[w][idx].append(end[idx])
    for idx in (9, 10, 12):
        results[w][idx] = np.array(results[w][idx])

for idx, ylabel, tag, color in [
    (12, 'Cumulative Hospitalizations', 'hospitalizations', 'firebrick'),
    (10, 'Cumulative Deaths', 'deaths', 'dimgray'),
    (9, 'Cumulative Infections', 'infections', 'steelblue'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, w in zip(axes, windows):
        arr = results[w][idx]
        spread = (arr.max() - arr.min()) / arr.max() * 100
        ax.bar(x, arr, color=color)
        ax.set_title(f'{w}-day windows  (spread: {spread:.1f}%)', fontweight='bold')
        ax.set_xlabel('Vaccination Order')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f'Total {tag.capitalize()} by Vaccination Order', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'bar_{tag}.png', dpi=200)
    plt.close()

for w in windows:
    for idx, name in [(10, 'deaths'), (12, 'hosp'), (9, 'infections')]:
        a = results[w][idx]
        print(f'{w}d {name:11s} {a.min():.4g} to {a.max():.4g}  spread {(a.max()-a.min())/a.max()*100:.1f}%')
