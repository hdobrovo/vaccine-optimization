import numpy as np
from scipy.integrate import odeint

N = 330e6
mu = 0#2.74e-5
Bl, Bh = 0.1981, 0.3301
al, ah = 1 / 9.5, 1 / 12
rho, delta = 0.0432, 7.23e-4

VAC = slice(13, 17)

def initial_state(proportions=(0.25, 0.25, 0.25, 0.25)):
    S = [N * p for p in proportions]
    return S + [1e3, 1e3, 0, 0, 0, 2e3, 0, 0, 0, 0, 0, 0, 0]

def model(y, t, gammas):
    Sll, Slh, Shl, Shh, Il, Ih, H, V, R = y[:9]
    S = [Sll, Slh, Shl, Shh]
    I = Il + Ih
    inf_l = Bl * Sll * I / N + Bh * Shl * I / N
    inf_h = Bl * Slh * I / N + Bh * Shh * I / N
    vac = [gammas[i] * S[i] for i in range(4)]
    dSll = -Bl * Sll * I / N - vac[0] - mu * Sll
    dSlh = -Bl * Slh * I / N - vac[1] - mu * Slh
    dShl = -Bh * Shl * I / N - vac[2] - mu * Shl
    dShh = -Bh * Shh * I / N - vac[3] - mu * Shh
    dIl = inf_l - al * Il - mu * Il
    dIh = inf_h - ah * Ih - rho * Ih - delta * Ih - mu * Ih
    dH = rho * Ih - ah * H - delta * H - mu * H
    dV = sum(vac) - mu * V
    dR = al * Il + ah * (Ih + H) - mu * R
    return [dSll, dSlh, dShl, dShh, dIl, dIh, dH, dV, dR,
            inf_l + inf_h, delta * (Ih + H), al * Il + ah * (Ih + H), rho * Ih] + vac

def run_ordering(y0, order, gamma, window, final_day):
    y, t = y0, 0
    for grp in order:
        gvec = [0, 0, 0, 0]
        gvec[grp] = gamma
        y = odeint(model, y, np.linspace(t, t + window, 50), args=(gvec,))[-1]
        t += window
    if t < final_day:
        y = odeint(model, y, np.linspace(t, final_day, 400), args=([0, 0, 0, 0],))[-1]
    return y
