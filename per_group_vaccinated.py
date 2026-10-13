import itertools
import numpy as np
import matplotlib.pyplot as plt

from model import initial_state, run_ordering, VAC

gamma = np.logspace(-6, 0, 20)[14]
window, final_day = 30, 350
groups = ['Sll', 'Slh', 'Shl', 'Shh']
y0 = initial_state()

orders = list(itertools.permutations(range(4)))
deaths = [run_ordering(y0, o, gamma, window, final_day)[10] for o in orders]
best, worst = orders[int(np.argmin(deaths))], orders[int(np.argmax(deaths))]

print(f"{'group':6}" + ''.join(f'{g:>14}' for g in groups) + f"{'total':>14}")
data = {}
for order in (best, worst):
    v = np.array(run_ordering(y0, order, gamma, window, final_day)[VAC])
    data['→'.join(groups[i] for i in order)] = v
    print(f"{'→'.join(groups[i] for i in order):40} "
          + ' '.join(f'{x:12.3g}' for x in v) + f' {v.sum():12.4g}')

x = np.arange(4)
w = 0.38
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for k, (name, v) in enumerate(data.items()):
    ax.bar(x + (k - 0.5) * w, v / 1e6, w, label=name)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Total vaccinated (millions)')
ax.set_xlabel('Group')
ax.legend(title='Ordering', fontsize=8)
plt.tight_layout()
plt.savefig('vaccinated_by_group.png', dpi=200)
plt.close()
