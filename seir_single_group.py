import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

N = 330e6
mu = 2.74e-5
Bl, Bh = 0.1981, 0.3301
al, ah = 1 / 9.5, 1 / 12
rho, delta = 0.0432, 7.23e-4
sigma = 1 / 5

t = np.linspace(0, 300, 1000)
dt = t[1] - t[0]

y0 = [N * 0.25, N * 0.25, N * 0.25, N * 0.25, 0, 0, 1e3, 1e3, 0]

def model(y, t, gammas):
    Sll, Slh, Shl, Shh, El, Eh, Il, Ih, H = y
    I = Il + Ih
    exp_l = Bl * Sll * I / N + Bh * Shl * I / N
    exp_h = Bl * Slh * I / N + Bh * Shh * I / N
    return [-Bl * Sll * I / N - gammas[0] * Sll - mu * Sll,
            -Bl * Slh * I / N - gammas[1] * Slh - mu * Slh,
            -Bh * Shl * I / N - gammas[2] * Shl - mu * Shl,
            -Bh * Shh * I / N - gammas[3] * Shh - mu * Shh,
            exp_l - sigma * El - mu * El,
            exp_h - sigma * Eh - mu * Eh,
            sigma * El - al * Il - mu * Il,
            sigma * Eh - ah * Ih - rho * Ih - delta * Ih - mu * Ih,
            rho * Ih - ah * H - delta * H - mu * H]

gammas_scan = np.logspace(-6, 0, 100)
labels = ['su', 'shi', 'shl', 'shh']

results = []
for i in range(4):
    infected, hospitalized, deaths = [], [], []
    for g in gammas_scan:
        gvec = [0, 0, 0, 0]
        gvec[i] = g
        sol = odeint(model, y0, t, args=(gvec,))
        Il, Ih, H = sol[:, 6], sol[:, 7], sol[:, 8]
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
    plt.savefig(f'seir_{labels[i]}_only.png', dpi=200)
    plt.close()
