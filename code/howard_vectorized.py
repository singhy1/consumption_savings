# Yash Singh 
# Vectorized Howard's Algorithm with Explicit Matrix Operations
# Uses @ operator for clarity and minimizes indexing

import numpy as np 
from scipy.sparse import lil_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import spsolve
from numba import jit
from utilities import utility


def interp_linear(x, x_grid, y_grid):
    """Linear interpolation."""
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]
    
    i = np.searchsorted(x_grid, x, side='right') - 1
    weight = (x - x_grid[i]) / (x_grid[i+1] - x_grid[i])
    return (1 - weight) * y_grid[i] + weight * y_grid[i+1]


def build_transition_matrix(a_prime, w_grid, y_grid, Pi):
    """
    Build transition matrix P given policy a_prime.
    
    P[(i,j), (i',j')] = Pr(next state is (w_i', y_j') | current state is (w_i, y_j))
    
    Parameters:
    -----------
    a_prime : ndarray (N_w, N_y)
        Policy function: a_prime[i,j] = savings at state (w_i, y_j)
    w_grid : ndarray (N_w,)
        Wealth grid
    y_grid : ndarray (N_y,)
        Income grid
    Pi : ndarray (N_y, N_y)
        Income transition matrix: Pi[j,j'] = Pr(y'=y_j' | y=y_j)
    
    Returns:
    --------
    P : sparse matrix (N, N) where N = N_w * N_y
        Transition matrix in CSR format
    """
    from scipy.sparse import lil_matrix
    
    N_w, N_y = len(w_grid), len(y_grid)
    N = N_w * N_y
    P = lil_matrix((N, N))
    
    # For each current state (w_i, y_j)
    for j in range(N_y):
        for i in range(N_w):
            curr_idx = j * N_w + i  # Current state index
            
            # For each possible next income y_j'
            for j_next in range(N_y):
                if Pi[j, j_next] < 1e-12:  # Skip negligible probabilities
                    continue
                
                # Next wealth: w' = a' + y'
                w_next = a_prime[i, j] + y_grid[j_next]
                
                # Find bracketing indices and weight
                i_low, i_high, weight = get_interp_weights(w_next, w_grid)
                
                # Fill transition probabilities
                next_idx_low = j_next * N_w + i_low
                next_idx_high = j_next * N_w + i_high
                
                P[curr_idx, next_idx_low] += Pi[j, j_next] * (1 - weight)
                P[curr_idx, next_idx_high] += Pi[j, j_next] * weight
    
    return P.tocsr()

def policy_evaluation(a_prime, w_grid, y_grid, Pi, beta, R, rho):
    """
    Solve for value function V given policy a_prime.
    
    Solves: (I - β P) V = u
    
    Parameters:
    -----------
    a_prime : ndarray (N_w, N_y)
        Policy function
    w_grid : ndarray (N_w,)
        Wealth grid
    y_grid : ndarray (N_y,)
        Income grid
    Pi : ndarray (N_y, N_y)
        Income transition matrix
    beta : float
        Discount factor
    R : float
        Gross interest rate
    rho : float
        Risk aversion parameter
    
    Returns:
    --------
    V : ndarray (N_w, N_y)
        Value function
        
    Mathematical Formula:
    --------------------
    V = (I - β P)^(-1) u
    
    where:
    - u[k] = u(w_i - a'[i,j]/R) for state k = (w_i, y_j)
    - P[k,l] = Pr(next state is l | current state is k, following policy a')
    """
    from scipy.sparse import eye
    from scipy.sparse.linalg import spsolve
    
    N_w, N_y = len(w_grid), len(y_grid)
    N = N_w * N_y
    
    # Step 1: Compute consumption for all states
    # c[i,j] = w[i] - a'[i,j]/R
    c = w_grid[:, np.newaxis] - a_prime / R  # Broadcasting: (N_w,1) - (N_w,N_y)
    c = np.maximum(c, 1e-10)  # Ensure positive
    
    # Step 2: Compute utility for all states
    u_matrix = utility(c, rho)  # Shape: (N_w, N_y)
    
    # Step 3: Flatten to vector (match state ordering k = j*N_w + i)
    u_vec = np.zeros(N)
    for j in range(N_y):
        for i in range(N_w):
            u_vec[j * N_w + i] = u_matrix[i, j]
    
    # Step 4: Build transition matrix P
    P = build_transition_matrix(a_prime, w_grid, y_grid, Pi)
    
    # Step 5: Solve (I - βP) V = u
    A = eye(N, format='csr') - beta * P
    V_vec = spsolve(A, u_vec)
    
    # Step 6: Reshape back to (N_w, N_y)
    V = np.zeros((N_w, N_y))
    for j in range(N_y):
        for i in range(N_w):
            V[i, j] = V_vec[j * N_w + i]
    
    return V

def policy_improvement(V, w_grid, y_grid, Pi, beta, R, rho, n_a=150):
    
    N_w, N_y = len(w_grid), len(y_grid)
    a_prime = np.zeros((N_w, N_y))
    
    for j in range(N_y):
        for i in range(N_w):
            w = w_grid[i]
            a_grid = np.linspace(0, R * w, n_a)
            
            # Utility
            c = np.maximum(w - a_grid/R, 1e-10)
            u = utility(c, rho)
            
            # Expected value
            ev = np.array([
                Pi[j, :] @ np.array([interp_linear(a + y_grid[jn], w_grid, V[:, jn]) 
                                     for jn in range(N_y)])
                for a in a_grid
            ])
            
            
            a_prime[i, j] = a_grid[np.argmax(u + beta * ev)]
    
    return a_prime

