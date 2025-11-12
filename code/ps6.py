#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
myred = '#8B0000'
myblue = '#003366'

# Parameters
delta = 0.10
r = 0.05
rho = 2
mu=100
phi = 0.7
sigma_u = 10.0
mu_y = 100.0
ny = 7 # states in Tauchen
na = 1000
a_max = 200.0
tol = 1e-6
maxiter = 2000
r_std = 3

# utility
def u(c, rho):
    """CRRA utility"""
    return (np.maximum(c, 1e-10) ** (1 - rho) / (1 - rho))

def u_prime(c, rho):
    """Marginal utility"""
    return np.maximum(c, 1e-10) ** (-rho)

def u_prime_inv(mu, rho):
    """Inverse marginal utility: given u'(c), return c."""
    return np.maximum(mu, 1e-10) ** (-1.0 / rho)

# Tauchen Method
def tauchen(phi, sigma_u, n=7, m=3, mu=100.0):
    """
    Tauchen method to discretize an AR(1). Returns:
    y_grid: Grid of discretized income values (n x 1)
    P: Transition probability matrix (n x n)
    """
    std_y = sigma_u / np.sqrt(1 - phi**2)
    y_min, y_max = mu - m * std_y, mu + m * std_y
    y_grid = np.linspace(y_min, y_max, n)
    step = (y_max - y_min) / (n - 1)

    P = np.zeros((n, n))
    for i, y_i in enumerate(y_grid):
        mu_cond = mu + phi * (y_i - mu)
        P[i, 0]      = norm.cdf((y_grid[0] - mu_cond + step/2) / sigma_u)
        P[i, -1]     = 1 - norm.cdf((y_grid[-1] - mu_cond - step/2) / sigma_u)
        for j in range(1, n - 1):
            z_upper = (y_grid[j]   - mu_cond + step/2) / sigma_u
            z_lower = (y_grid[j]   - mu_cond - step/2) / sigma_u
            P[i, j] = norm.cdf(z_upper) - norm.cdf(z_lower)

    # Ensure nonnegative income and prob. sum to 1 for each row of P
    y_grid = np.maximum(y_grid, 0.0)
    P /= P.sum(axis=1, keepdims=True)
    return y_grid, P


# EGM
def solve_egm(delta=delta, r=r, rho=rho, phi=phi, sigma_u=sigma_u,
              ny=ny, mu=mu_y, a_max=a_max, na=na,
              tol=tol, maxiter=maxiter, r_std=r_std):
    """Solve the consumption-savings problem using the Endogenous Grid Method (EGM)."""

    # discretize income process
    y_grid, P = tauchen(phi, sigma_u, n=ny, m=r_std, mu=mu)

    # asset and cash-on-hand grids
    A = np.linspace(0.0, a_max, na)
    X = np.linspace(0.0, (1 + r) * a_max + np.max(y_grid), na)

    # initial guess: consume all cash-on-hand
    c = np.tile(X, (ny, 1))
    factor = (1 + r) / (1 + delta)

    for it in range(maxiter):
        c_old = c.copy()

        # Expected marginal utility
        X_next = (1 + r) * A[None, :] + y_grid[:, None]
        X_next = np.clip(X_next, X[0], X[-1])
        c_next = np.array([np.interp(X_next[iy], X, c[iy]) for iy in range(ny)])
        Eu_prime = P @ u_prime(c_next, rho)

        # Invert Euler equation
        c_endo = u_prime_inv(factor * Eu_prime, rho)
        X_endo = c_endo + A[None, :]

        # Interpolate onto exogenous grid with borrowing constraint
        for iy in range(ny):
            Xs, Cs = np.sort(X_endo[iy]), np.sort(c_endo[iy])
            c[iy] = np.where(X <= Xs[0], X, np.interp(X, Xs, Cs))

        diff = np.max(np.abs(c - c_old))
        if diff < tol:
            break

    print(f"EGM converged in {it+1} iterations (diff={diff:.2e})")
    return {"X_grid": X, "A_grid": A, "y_grid": y_grid, "P": P, "c_policy": c}


# Fig. 1: Cons. Policy for two sigmas with PHI=0
fig, ax = plt.subplots(figsize=(8, 5))

for sigma in [10, 15]:
    sol = solve_egm(phi=0.0, sigma_u=sigma)
    X_grid, c_policy, y_grid = sol["X_grid"], sol["c_policy"], sol["y_grid"] 
    # pick mid income path (check any income is the same)
    idx_mid = (ny - 1) // 2 #-1 for zero indexing
    max_row_diff = np.max(
        np.max(c_policy, axis=0) - np.min(c_policy, axis=0))
    print(f"For phi=0, max diff. across inc states is {max_row_diff}")
    color = myblue if sigma == 10 else myred
    ax.plot(X_grid, c_policy[idx_mid, :], lw=2, color=color, label=rf"$\sigma = {sigma}$")

# linear policy
c_linear = (100 + r * X_grid) / (1 + r)
ax.plot(X_grid, c_linear, 'k--', lw=2, label=r"$c = (100 + rx)/(1 + r)$")

# mean income
ax.axvline(100, color='gray', linestyle=':', lw=2, label=r"$\mu = 100$")

ax.set_xlabel("Cash-on-hand")
ax.set_ylabel("Consumption")
ax.legend()
ax.set_xlim(0, 180)
ax.set_ylim(0, 180)
ax.grid(True)
plt.tight_layout()
plt.savefig('fig1.png', dpi=300)
plt.show()

# Fig. 3: Consumption policy by income state with PHI=.7
fig, ax = plt.subplots(figsize=(8, 5))

sol = solve_egm(phi=0.7, sigma_u=10, rho=2, mu=100, r=0.02, delta=0.05)
X_grid, c_policy, y_grid = sol["X_grid"], sol["c_policy"], sol["y_grid"]

n_shades = len(y_grid)
reds = [mcolors.to_rgba(myred, alpha=1 - 0.6 * (i / (n_shades - 1))) for i in range(n_shades)]
for iy, y_val in enumerate(y_grid):
    ax.plot(X_grid, c_policy[iy, :], lw=2, color=reds[iy], label=rf"$y \approx {y_val:.0f}$")
    
ax.set_xlabel("Cash-on-hand")
ax.set_ylabel("Consumption")
ax.grid(True)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("fig3.png", dpi=300)
plt.show()

# Simulate discrete paths of income and apply existing EGM solutions
def simulate_income(P, y_grid, T=200, seed=123):
    """Simulate discrete income process given transition matrix P."""
    np.random.seed(seed)
    n = len(y_grid)
    y_idx = np.zeros(T, dtype=int)
    y_idx[0] = n // 2  # start from avg income
    for t in range(1, T):
        y_idx[t] = np.random.choice(n, p=P[y_idx[t-1], :])
    return y_grid[y_idx]

def simulate_paths(sol, T=200, r=r):
    """Simulate assets and consumption given EGM policy."""
    X_grid, y_grid, P, c_policy = (
        sol["X_grid"], sol["y_grid"], sol["P"], sol["c_policy"]
    )
    y = simulate_income(P, y_grid, T=T)
    a = np.zeros(T+1)
    c = np.zeros(T)
    for t in range(T):
        X = (1 + r) * a[t] + y[t]
        # interpolate using nearest income grid
        iy = np.argmin(np.abs(y_grid - y[t]))
        c[t] = np.interp(X, X_grid, c_policy[iy, :])
        a[t+1] = X - c[t] # assets follow from cash - consumption
    return a[:-1], c, y

# loop over phi (and hence policies)
phi_dict = {
    0.0: solve_egm(phi=0.0, sigma_u=10.0, mu=100, rho=2),
    0.7: solve_egm(phi=0.7, sigma_u=10.0, mu=100, rho=2, r=0.02, delta=0.05)
}
fig_names = {0.0: "fig2.png", 0.7: "fig4.png"}

for phi_val, sol in phi_dict.items():
    a, c, y = simulate_paths(sol, T=200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(y, color='gray', lw=2, label='Income (discrete)')
    ax.plot(c - 40, color=myred, lw=2, label='Consumption - 40')
    ax.plot(a, color=myblue, lw=2, label='Assets')
    ax.set_xlabel("Time")
    ax.set_ylabel("Amount")
    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_names[phi_val], dpi=300)
    plt.show()

# Policy iteration
def policy_iteration(delta=delta, r=r, rho=rho, ny=ny, na=na, a_max=a_max,
                     phi=phi, sigma_u=sigma_u, mu_y=mu_y,
                     tol=tol, maxiter=maxiter, n_cand=200):
    # Discretize income
    y_grid, P = tauchen(phi, sigma_u, n=ny, mu=mu_y)
    A = np.linspace(0.0, a_max, na)
    beta = 1/(1+delta)

    # Initial guess: consume all cash-on-hand
    c = np.tile(A[None, :] + y_grid[:, None], (1, 1))
    V = np.zeros_like(c)

    for it in range(maxiter):
        c_old = c.copy()
        V_old = V.copy()

        ### Policy Evaluation
        # set grid of assets next period
        a_next = (1 + r) * (A[None, :] + y_grid[:, None] - c)  # ny x na
        a_next = np.maximum(a_next, 0.0) # credit constraints

        # Value function: u() + beta * EV
        EV = np.zeros_like(c)
        for jy in range(ny):
            # Interpolate V at all a_next
            EV += P[:, jy][:, None] * np.interp(a_next, A, V[jy, :])
        V = u(c, rho) + beta * EV

        ### Policy Improvement
        # candidate consumption grids for all (iy, ia)
        cash_max = A[None, :] + y_grid[:, None]  # ny x na
        cash_grid = np.linspace(0, 1, n_cand)[None, None, :] * cash_max[:, :, None]
        # next assets for all candidate consumption
        a_next_cand = np.maximum((1 + r) * (cash_max[:, :, None] - cash_grid), 0.0)
        # expected continuation value for all candidates
        EV_cand = np.zeros_like(a_next_cand)
        for jy in range(ny):
            # interp V at all candidate next assets
            EV_cand += P[:, jy][:, None, None] * np.interp(a_next_cand, A, V[jy, :])
        # value function for each candidate consumption
        V_total = u(cash_grid, rho) + beta * EV_cand
        # optimal consumption: pick consumption that maximizes V
        c = np.take_along_axis(cash_grid, np.argmax(V_total, axis=2)[..., None], axis=2)[..., 0]
        diff_C = np.max(np.abs(c - c_old))
        diff_V = np.max(np.abs(V - V_old))
        print(f"Iter {it}: diff_C={diff_C:.2e}, diff_V={diff_V:.2e}")
        if diff_C < tol: # convergence
            print(f"Policy iteration converged in {it+1} iterations")
            break

    return {"A_grid": A, "y_grid": y_grid, "c_policy": c, "V": V, "P": P}

# Solve for phi=.7 via policy iteration
sol_pi = policy_iteration() 
A_grid, y_grid, c_policy, V = sol_pi["A_grid"], sol_pi["y_grid"], sol_pi["c_policy"], sol_pi["V"]

# Plot value function for different income states
fig, ax = plt.subplots(figsize=(8,5))
blues = [mcolors.to_rgba(myblue, alpha=1 - 0.6*(i/(ny-1))) for i in range(ny)]
for iy, y_val in enumerate(y_grid):
    assets = A_grid
    ax.plot(assets, V[iy,:], color=blues[iy], lw=2, label=f"$y={y_val:.1f}$")

ax.set_xlabel("Assets")
ax.set_ylabel("Value function")
ax.set_xlim(0, 200)
ax.grid(True)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("fig5.png", dpi=300)
plt.show()