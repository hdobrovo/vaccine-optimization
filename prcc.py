import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

N = 330e6
base = dict(Lam=2.74e-5 * N / 4, mu=2.74e-5, Bl=0.4066, Bh=0.6778,
            al=1 / 9.5, ah=1 / 12, rho=0.0432, delta=7.23e-4)


def model(y, t, Bl, Bh, al, ah, rho, delta, Lam, mu):
    Sll, Slh, Shl, Shh, Il, Ih, H, V, R = y[:9]
    I = Il + Ih
    inf_l = Bl * Sll * I / N + Bh * Shl * I / N
    inf_h = Bl * Slh * I / N + Bh * Shh * I / N
    return [Lam - Bl * Sll * I / N - mu * Sll, Lam - Bl * Slh * I / N - mu * Slh,
            Lam - Bh * Shl * I / N - mu * Shl, Lam - Bh * Shh * I / N - mu * Shh,
            inf_l - al * Il - mu * Il, inf_h - ah * Ih - rho * Ih - delta * Ih - mu * Ih,
            rho * Ih - ah * H - delta * H - mu * H, 0, al * Il + ah * (Ih + H) - mu * R,
            inf_l + inf_h, delta * (Ih + H), al * Il + ah * (Ih + H), rho * Ih]


def partial_corr(C):
    inv = np.linalg.inv(np.corrcoef(C, rowvar=False))
    p = C.shape[1]
    P = np.eye(p)
    for i in range(p):
        for j in range(i + 1, p):
            P[i, j] = P[j, i] = -inv[i, j] / np.sqrt(inv[i, i] * inv[j, j])
    return P


names = ['Lam', 'mu', 'Bl', 'Bh', 'al', 'ah', 'rho', 'delta']
t = np.linspace(0, 100, 1000)
y0 = [N * 0.25] * 4 + [1e3, 1e3, 0, 0, 0, 2e3, 0, 0, 0]

samples = []
for _ in range(1000):
    p = {k: random.uniform(v * 0.9, v * 1.1) for k, v in base.items()}
    sol = odeint(model, y0, t, args=(p['Bl'], p['Bh'], p['al'], p['ah'],
                                     p['rho'], p['delta'], p['Lam'], p['mu']))
    samples.append([p[k] for k in names] + [sol[-1, 10], sol[-1, 9], sol[-1, 12]])

prcc = partial_corr(np.array(samples))
deaths_prcc, inf_prcc, hosp_prcc = prcc[8, :8], prcc[9, :8], prcc[10, :8]

labels = [r'$\Lambda$', r'$\mu$', r'$\beta_\ell$', r'$\beta_h$',
          r'$\alpha_\ell$', r'$\alpha_h$', r'$\rho$', r'$\delta$']
x = np.arange(8)

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
for ax, vals, title in zip(axes, [deaths_prcc, inf_prcc, hosp_prcc],
                           ['Deaths', 'Infections', 'Hospitalizations']):
    ax.bar(x, vals, color=['red' if v > 0 else 'blue' for v in vals])
    ax.set_title(f'PRCC: {title}\nvs Parameters', fontweight='bold')
    ax.set_ylabel('PRCC Coefficient')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.axhline(0, color='black', alpha=0.3)
    ax.set_ylim(-1, 1)
plt.tight_layout(pad=2.0)
plt.savefig('prcc.pdf', format='pdf', bbox_inches='tight')
plt.show()

for i, name in enumerate(names):
    print(f'{name:6} deaths {deaths_prcc[i]:6.2f}  inf {inf_prcc[i]:6.2f}  hosp {hosp_prcc[i]:6.2f}')
