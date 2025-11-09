# all utility function for macro 1 

import numpy as np 
import pandas as pd 
from scipy.stats import norm
import statsmodels.api as sm


# CRRA utility 

def utility(x, gamma):
    if gamma == 1: 
        return np.log(x)
    else: 
        return (x**(1-gamma)) / (1-gamma)

def u_prime(x, gamma):
    return x**(-gamma)

def inv_uprime(x, gamma):
    return x ** (-1/gamma)


def discretize_series(series, grid):
    """
    For each element in 'series', assign the closest value from 'grid'.
    """
    series_discretized = np.array([grid[np.argmin(np.abs(grid - x))] for x in series])
    return series_discretized


def simulate_markov_chain(P, z_grid, pi, T=400, seed=None):
    """
    Simulates a discrete Markov chain given a transition matrix, grid, and stationary distribution.

    Parameters
    ----------
    P : np.ndarray
        Transition matrix (n x n), where each row sums to 1.
    z_grid : np.ndarray
        Array of state values corresponding to each Markov state.
    pi : np.ndarray
        Stationary distribution (n,).
    T : int, optional
        Number of periods to simulate (default 400).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    z_sim : np.ndarray
        Simulated series of length T (state values).
    s_sim : np.ndarray
        Simulated sequence of state indices.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(z_grid)
    assert P.shape == (n, n), "Transition matrix and grid size mismatch."

    # Find long-run mean and closest state
    long_run_mean = np.dot(pi, z_grid)
    init_state = np.argmin(np.abs(z_grid - long_run_mean))

    # Simulate
    s_sim = np.zeros(T, dtype=int)
    s_sim[0] = init_state

    for t in range(1, T):
        s_prev = s_sim[t-1]
        s_sim[t] = np.random.choice(n, p=P[s_prev, :])

    # Map indices to actual values
    z_sim = z_grid[s_sim]

    return z_sim


def stationary_stats(P, z_grid, tol=1e-10):
    """
    Computes the stationary distribution, mean, and standard deviation 
    of a discrete Markov chain.

    Parameters
    ----------
    P : np.ndarray
        Transition matrix (n x n), where each row sums to 1.
    z_grid : np.ndarray
        Array of state values corresponding to rows/columns of P.
    tol : float, optional
        Tolerance for convergence.

    Returns
    -------
    pi : np.ndarray
        Stationary distribution (n x 1) summing to 1.
    mean_z : float
        Stationary mean of the Markov chain.
    std_z : float
        Stationary standard deviation of the Markov chain.
    """
    n = len(z_grid)
    assert P.shape == (n, n), "Transition matrix and grid size mismatch."

    # Find stationary distribution (left eigenvector associated with eigenvalue 1)
    eigvals, eigvecs = np.linalg.eig(P.T)
    
    # Find the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigvals - 1))
    
    # Check if eigenvalue is actually close to 1
    if np.abs(eigvals[idx] - 1) > tol:
        raise ValueError(f"No eigenvalue close to 1 found. Closest: {eigvals[idx]}")
    
    pi = np.real(eigvecs[:, idx])
    
    # Normalize to sum to 1
    pi = np.abs(pi)  # Take absolute value instead of clipping
    pi = pi / pi.sum()
    
    # Verify it's a valid distribution
    if not np.allclose(P.T @ pi, pi, atol=tol):
        raise ValueError("Found eigenvector is not a valid stationary distribution.")

    # Compute mean and std
    mean_z = np.dot(pi, z_grid)
    std_z = np.sqrt(np.dot(pi, (z_grid - mean_z)**2))

    return pi, mean_z, std_z



def tauchen(mu, phi, sigma, n_states=7, m=3):
    """
    Discretize an AR(1): z' = mu + phi*z + eps, eps ~ N(0, sigma^2)
    using Tauchen's method.

    Returns:
        grid (np.ndarray): grid of z values
        Pi (np.ndarray): transition matrix
    """

    # Unconditional mean and std
    unconditional_mean = mu / (1 - phi)
    unconditional_std = sigma / np.sqrt(1 - phi**2)

    # Symmetric grid around unconditional mean
    z_min = unconditional_mean - m * unconditional_std
    z_max = unconditional_mean + m * unconditional_std
    grid = np.linspace(z_min, z_max, n_states)
    h = (grid[1] - grid[0]) / 2  # half-step size

    # Initialize transition matrix
    Pi = np.zeros((n_states, n_states))

    # First and last columns (tails)
    Pi[:, 0] = norm.cdf((grid[0] + h - mu - phi * grid) / sigma)
    Pi[:, -1] = 1 - norm.cdf((grid[-1] - h - mu - phi * grid) / sigma)

    # Interior columns
    for j in range(1, n_states - 1):
        Pi[:, j] = (norm.cdf((grid[j] + h - mu - phi * grid) / sigma)
                    - norm.cdf((grid[j] - h - mu - phi * grid) / sigma))
    
    Pi = Pi / Pi.sum(axis=1, keepdims=True)

    return grid, Pi



def ar1_k_step_forecast(last_pi, ar1_results, k):
    """
    Compute k-step ahead forecast for an AR(1) model.

    Parameters
    ----------
    last_pi : float
        Most recent observed value of the series (pi_T)
    ar1_results : dict
        Output from ar1_estimate() with keys: 'mu', 'alpha'
    k : int
        Number of steps ahead to forecast

    Returns
    -------
    forecast : float
        k-step ahead forecast
    """
    mu = ar1_results['mu']
    alpha = ar1_results['alpha']

    if alpha == 1:  # handle unit root case
        forecast = last_pi + k * mu
    else:
        forecast = mu * (1 - alpha**k) / (1 - alpha) + alpha**k * last_pi

    return forecast


def ar1_estimate(X, y, spec = "ar1"):
    """
    Runs an OLS regression with two specifications:
    - AR(1): y_t ~ const + X
    - Random Walk: delta_y_t = y_t - y_{t-1} ~ const
    
    Returns the constant term, the coefficient of the first column in X (alpha),
    and the residual variance.
    
    Parameters
    ----------
    X : pd.DataFrame or pd.Series
        Independent variables (can be one or more columns)
    y : pd.Series
        Dependent variable
    
    Returns
    -------
    dict
        {
            'const': estimated intercept,
            'alpha': estimated coefficient on first column of X,
            'resid_var': variance of residuals
        }
    """

    if spec == "ar1": 

        # Add constant
        X_with_const = sm.add_constant(X)
    
        # Fit OLS
        model = sm.OLS(y, X_with_const).fit()
    
        # Residuals and variance
        residuals = model.resid
        resid_var = np.var(residuals, ddof=1)
    
        # Get intercept and first coefficient (assumes first column in X is alpha)
        const = model.params['const']
        alpha = model.params[X.columns[0]]
    
    
    elif spec == "random":
        delta = y.diff().dropna()
        X_const = sm.add_constant(pd.DataFrame(index=delta.index, data=np.zeros(len(delta))))  
        model = sm.OLS(delta, X_const).fit()

        
        residuals = model.resid
        resid_var = np.var(residuals, ddof=1)
        const = model.params['const']
        alpha = None  # no slope in RW specification

    return {'mu': const, 'alpha': alpha, 'resid_var': resid_var}
