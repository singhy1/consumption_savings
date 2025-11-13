# Yash Singh 
# Goal: Howard's Improvement Algorithm 

import numpy as np 
from utilities import utility
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize_scalar


output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"

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

