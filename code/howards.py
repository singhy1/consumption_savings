# Yash Singh 
# Goal: Howard's Improvement Algorithm 

import numpy as np 
from utilities import utility, tauchen
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"
# ============================================================
# SETUP
# ============================================================

# Model parameters
beta, R, rho = 0.95, 1.05, 2.0

# Wealth grid
N_w = 100
w_grid = np.linspace(0.0, 100.0, N_w)

# Income process (Tauchen)
y_grid, Pi = tauchen(mu=30, phi=0.7, sigma=10, n_states=15, m=4)
N_y = len(y_grid)

print(f"Income grid: min={y_grid.min():.1f}, max={y_grid.max():.1f}")
print(f"Transition matrix: {Pi.shape}, rows sum to {Pi.sum(axis=1).mean():.3f}")

# ============================================================
# POLICY EVALUATION
# ============================================================

def policy_evaluation(a_prime, w_grid, y_grid, Pi, beta, R, rho):
    """Solve (I - βP)V = u for V given policy a_prime."""
    N_w, N_y = len(w_grid), len(y_grid)
    N = N_w * N_y
    
    # Build utility vector
    u = np.zeros(N)
    for i_y in range(N_y):
        for i_w in range(N_w):
            c = max(w_grid[i_w] - a_prime[i_w, i_y] / R, 1e-10)
            u[i_y * N_w + i_w] = utility(c, rho)
    
    # Build transition matrix (sparse)
    P = lil_matrix((N, N))
    for i_y in range(N_y):
        for i_w in range(N_w):
            current_idx = i_y * N_w + i_w
            w_prime = a_prime[i_w, i_y] + y_grid
            
            for i_y_prime in range(N_y):
                wp = w_prime[i_y_prime]
                
                # Interpolate on wealth grid
                if wp <= w_grid[0]:
                    P[current_idx, i_y_prime * N_w] = Pi[i_y, i_y_prime]
                elif wp >= w_grid[-1]:
                    P[current_idx, i_y_prime * N_w + N_w - 1] = Pi[i_y, i_y_prime]
                else:
                    i_low = np.searchsorted(w_grid, wp) - 1
                    i_low = max(0, min(i_low, N_w - 2))
                    weight = (wp - w_grid[i_low]) / (w_grid[i_low + 1] - w_grid[i_low])
                    
                    P[current_idx, i_y_prime * N_w + i_low] += Pi[i_y, i_y_prime] * (1 - weight)
                    P[current_idx, i_y_prime * N_w + i_low + 1] += Pi[i_y, i_y_prime] * weight
    
    # Solve linear system
    V_flat = spsolve(eye(N, format='csr') - beta * P.tocsr(), u)
    return V_flat.reshape(N_y, N_w).T

# ============================================================
# POLICY IMPROVEMENT
# ============================================================

def policy_improvement(V, w_grid, y_grid, Pi, beta, R, rho):
    """Find optimal policy given value function V."""
    N_w, N_y = len(w_grid), len(y_grid)
    a_prime = np.zeros((N_w, N_y))
    
    for i_y in range(N_y):
        for i_w in range(N_w):
            w = w_grid[i_w]
            
            def objective(a):
                c = w - a / R
                if c <= 1e-10 or a < 0 or a > R * w:
                    return 1e10
                
                u_val = utility(c, rho)
                cont_val = sum(Pi[i_y, j] * np.interp(a + y_grid[j], w_grid, V[:, j]) 
                              for j in range(N_y))
                return -(u_val + beta * cont_val)
            
            result = minimize_scalar(objective, bounds=(0, R * w), method='bounded')
            a_prime[i_w, i_y] = result.x
    
    return a_prime

# ============================================================
# HOWARD'S ALGORITHM
# ============================================================

def howard_iteration(w_grid, y_grid, Pi, beta, R, rho, max_iter=50, tol=1e-6):
    """Solve using Howard's policy iteration."""
    N_w, N_y = len(w_grid), len(y_grid)
    a_prime = np.zeros((N_w, N_y))
    
    print(f"\nHoward's Algorithm: N_w={N_w}, N_y={N_y}, β={beta}, R={R}, ρ={rho}")
    print("="*60)
    
    for it in range(max_iter):
        V = policy_evaluation(a_prime, w_grid, y_grid, Pi, beta, R, rho)
        a_prime_new = policy_improvement(V, w_grid, y_grid, Pi, beta, R, rho)
        
        diff = np.abs(a_prime_new - a_prime).max()
        print(f"Iter {it+1:2d}: ||Δa'|| = {diff:.2e}")
        
        if diff < tol:
            print(f"Converged in {it+1} iterations!\n")
            return a_prime_new, V
        
        a_prime = a_prime_new
    
    print("Warning: Did not converge\n")
    return a_prime, V

# ============================================================
# RUN
# ============================================================

a_prime_opt, V_opt = howard_iteration(w_grid, y_grid, Pi, beta, R, rho)

# Consumption policy
c_opt = w_grid[:, None] - a_prime_opt / R

print(f"Avg savings rate: {(a_prime_opt / (R * w_grid[:, None])).mean():.1%}")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Savings
for i_y in [0, N_y//2, N_y-1]:
    axes[0].plot(w_grid, a_prime_opt[:, i_y], label=f'y={y_grid[i_y]:.0f}')
axes[0].plot(w_grid, R*w_grid, 'k--', alpha=0.3)
axes[0].set(xlabel='Cash-on-hand', ylabel='Assets saved', title='Savings Policy')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Consumption
for i_y in [0, N_y//2, N_y-1]:
    axes[1].plot(w_grid, c_opt[:, i_y], label=f'y={y_grid[i_y]:.0f}')
axes[1].plot(w_grid, w_grid, 'k--', alpha=0.3)
axes[1].set(xlabel='Cash-on-hand', ylabel='Consumption', title='Consumption Policy')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Value function
for i_y in [0, N_y//2, N_y-1]:
    axes[2].plot(w_grid, V_opt[:, i_y], label=f'y={y_grid[i_y]:.0f}')
axes[2].set(xlabel='Cash-on-hand', ylabel='Value', title='Value Function')
axes[2].legend()
axes[2].grid(alpha=0.3) 
plt.tight_layout()
plt.savefig(f"{output_path}/figures/howard_results.pdf")
