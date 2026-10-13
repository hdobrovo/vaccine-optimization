import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from model import model, initial_state, delta

t = np.linspace(0, 300, 1000)
dt = t[1] - t[0]
y0 = initial_state()

gammas_scan = np.logspace(-6, 0, 100)
labels = ['su', 'shi', 'shl', 'shh']

results = []
for i in range(4):
    infected, hospitalized, deaths = [], [], []
    for g in gammas_scan:
        gvec = [0, 0, 0, 0]
        gvec[i] = g
        sol = odeint(model, y0, t, args=(gvec,))
        Il, Ih, H = sol[:, 4], sol[:, 5], sol[:, 6]
        infected.append(np.sum(Il + Ih) * dt)
        hospitalized.append(np.sum(H) * dt)
        deaths.append(np.sum(delta * (Ih + H)) * dt)
    results.append((infected, hospitalized, deaths))

vals = [v for r in results for s in r for v in s if v > 0]
ylo, yhi = min(vals) / 3, max(vals) * 3

for i in range(4):
    infected, hospitalized, deaths = results[i]
    plt.figure(figsize=(4.4, 3.2))
    plt.plot(gammas_scan, infected, 'o-', ms=2.5, label='Infected (person-days)')
    plt.plot(gammas_scan, hospitalized, 's-', ms=2.5, label='Hospitalized (person-days)')
    plt.plot(gammas_scan, deaths, '^-', ms=2.5, label='Deaths (people)')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(ylo, yhi)
    plt.xlabel('Vaccination Rate (d$^{-1}$)')
    plt.ylabel('Cumulative Total')
    plt.legend(fontsize=8)
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{labels[i]} only.png', dpi=200)
    plt.show()
