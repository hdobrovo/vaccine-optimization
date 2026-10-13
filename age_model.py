import numpy as np
from scipy.integrate import odeint

N = 330e6
n = 5
mu = 0#2.74e-5

labels = ['0-17', '18-29', '30-49', '50-64', '65+']
age_fracs = np.array([0.22, 0.16, 0.26, 0.19, 0.17])
hosp_factor = np.array([0.336, 1.0, 1.5, 3.1, 8.4])
death_factor = np.array([0.155, 1.0, 3.5, 25.0, 144.0])
recov_factor = np.array([1.0, 1.0, 1.1, 1.3, 1.6])

beta = 0.4066 * np.ones(n)
alpha = (1 / 9.5) / recov_factor
rho = 0.0432 * hosp_factor
delta = 7.23e-4 * death_factor

def initial_state():
    S = N * age_fracs - 250
    I = np.full(n, 250.0)
    z = np.zeros(n)
    return np.concatenate([S, I, z, z, z, I.copy(), z, z])

def model(y, t, gamma):
    S, I, H, V, R = (y[i * n:(i + 1) * n] for i in range(5))
    foi = beta * I.sum() / N
    new_inf = foi * S
    new_hosp = rho * I
    vac = gamma * S
    dS = -new_inf - vac - mu * S
    dI = new_inf - alpha * I - rho * I - delta * I - mu * I
    dH = new_hosp - alpha * H - delta * H - mu * H
    dV = vac
    dR = alpha * I + alpha * H - mu * R
    return np.concatenate([dS, dI, dH, dV, dR, new_inf, delta * (I + H), new_hosp])

def run_ordering(y0, order, gamma, window, final_day):
    y, t = y0.copy(), 0.0
    for grp in order:
        gvec = np.zeros(n)
        gvec[grp] = gamma
        y = odeint(model, y, np.linspace(t, t + window, 50), args=(gvec,))[-1]
        t += window
    if t < final_day:
        y = odeint(model, y, np.linspace(t, final_day, 400), args=(np.zeros(n),))[-1]
    return y

def outcomes(y):
    return y[5 * n:6 * n].sum(), y[6 * n:7 * n].sum(), y[7 * n:8 * n].sum()
