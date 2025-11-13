# ============================================================
# RUN HOWARD ITERATION AND PLOT POLICY FUNCTIONS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from utilities import tauchen
from howards import howard_iteration  # <-- uses your existing functions

# ============================================================
# SETUP
# ============================================================

# Model parameters
beta, R, rho = 0.95, 1.05, 2.0

# Wealth grid
N_w = 250
w_grid = np.linspace(0.0, 250, N_w)

# Income process via Tauchen
y_grid, Pi = tauchen(mu=30, phi=0.7, sigma=10, n_states=15, m=4)
N_y = len(y_grid)

print(f"Income grid: min={y_grid.min():.1f}, max={y_grid.max():.1f}")
print(f"Transition matrix: {Pi.shape}, rows sum to {Pi.sum(axis=1).mean():.3f}")

# ============================================================
# RUN HOWARD ITERATION
# ============================================================

a_prime, V = howard_iteration(w_grid, y_grid, Pi, beta, R, rho,
                              max_iter=50, tol=1e-6)

# ============================================================
# RECOVER CONSUMPTION AND SAVINGS FUNCTIONS
# ============================================================

# c(w,s) = w - a'(w,s)/R
c_policy = w_grid.reshape(-1,1) - a_prime / R

# savings = a'(w,s)
s_policy = a_prime


# ============================================================
# PLOTTING
# ============================================================
# Set your figure path
output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"


# Choose representative income states
idxs = [0, N_y//4, N_y//2, 3*N_y//4, N_y-1]
labels = [f"y-state {i} (y={y_grid[i]:.1f})" for i in idxs]

# ------------------------------------------------------------
# Plot Value Functions
# ------------------------------------------------------------
plt.figure(figsize=(12,8))
for i, lab in zip(idxs, labels):
    plt.plot(w_grid, V[:, i], label=lab)

plt.title("Value Function V(w,s) by Income State", fontsize = 22)
plt.xlabel("Cash on hand (w)", fontsize = 18)
plt.ylabel("Value", fontsize = 18)
plt.ylim(-.5, .1)
plt.grid(True)
plt.legend(fontsize=12)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(f"{output_path}/figures/fig_v_by_income.pdf", dpi=600)
plt.show()

# ------------------------------------------------------------
# Plot Consumption Policies
# ------------------------------------------------------------
plt.figure(figsize=(12,8))
for i, lab in zip(idxs, labels):
    plt.plot(w_grid, c_policy[:, i], label=lab)

plt.title("Consumption Policy c(w,s)", fontsize = 22)
plt.xlabel("Cash on hand (w)", fontsize = 18)
plt.ylabel("Consumption (c)", fontsize = 18)
plt.xlim(0,200)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_path}/figures/fig_c_by_income.pdf", dpi=600)
plt.show()

# ------------------------------------------------------------
# Plot Savings Policies
# ------------------------------------------------------------
plt.figure(figsize=(12,8))
for i, lab in zip(idxs, labels):
    plt.plot(w_grid, s_policy[:, i], label=lab)

plt.title("Asset Policy a'(w,s)", fontsize = 22)
plt.xlabel("Cash on hand (w)", fontsize = 18)
plt.ylabel("Next Period assets (a')", fontsize = 18)
plt.xlim(0,200)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"{output_path}/figures/fig_s_by_income.pdf", dpi=600)
plt.show()
