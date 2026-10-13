import numpy as np
import matplotlib.pyplot as plt

from age_model import (
    initial_state as initial_state_age,
    run_ordering as run_ordering_age,
    n as n_age,
)
from model import (
    initial_state as initial_state_risk,
    run_ordering as run_ordering_risk,
)


# Simulation settings
window = 30
final_day = 350
gamma_values = np.logspace(-4, -1, 20)

# Age groups in age_model.py:
# [0-17, 18-29, 30-49, 50-64, 65+]
age_order = (4, 3, 2, 1, 0)

# Risk groups in model.py:
# [Sll, Slh, Shl, Shh]
risk_order = (3, 2, 1, 0)

y0_age = initial_state_age()
y0_risk = initial_state_risk()

# Each entry contains:
# (panel title, age-model block, risk-model index, y-axis label)
panels = [
    ("Hospitalizations", 7, 12, "Cumulative hospitalizations"),
    ("Deaths",           6, 10, "Cumulative deaths"),
    ("Infected",         5,  9, "Cumulative infections"),
    ("Vaccinated",       3,  7, "People vaccinated"),
]

age_curves = np.empty((len(panels), len(gamma_values)))
risk_curves = np.empty_like(age_curves)

# Run both models at each vaccination rate.
for j, gamma in enumerate(gamma_values):
    end_age = run_ordering_age(
        y0_age.copy(), age_order, gamma, window, final_day
    )
    end_risk = run_ordering_risk(
        y0_risk.copy(), risk_order, gamma, window, final_day
    )

    for i, (_, age_block, risk_index, _) in enumerate(panels):
        start = age_block * n_age
        stop = (age_block + 1) * n_age

        age_curves[i, j] = end_age[start:stop].sum()
        risk_curves[i, j] = end_risk[risk_index]

# Plot the four outcomes in one figure.
plt.rcParams.update({"font.size": 10})

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

for i, (ax, (title, _, _, ylabel)) in enumerate(
    zip(axes.flat, panels)
):
    ax.loglog(
        gamma_values,
        age_curves[i],
        color="tab:blue",
        linestyle="-",
        linewidth=2,
        marker=None,
        label="Age: oldest to youngest",
    )

    ax.loglog(
        gamma_values,
        risk_curves[i],
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        marker=None,
        label="Risk: Shh → Shl → Slh → Sll",
    )

    ax.set_title(title)
    ax.set_xlabel(r"Vaccination rate, $\gamma$ (day$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=9)

# Use one shared legend for the entire figure.
handles, legend_labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    frameon=False,
)

fig.suptitle(
    "Age-structured and risk-structured vaccination strategies",
    fontsize=13,
    y=0.99,
)

fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig("combined_ordering_scan.png", dpi=300, bbox_inches="tight")

plt.show()
