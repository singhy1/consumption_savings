# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities import tauchen, stationary_stats, simulate_markov_chain  # adjust import path
from deaton_model import DeatonModel

def generate_deaton_table(model, phi_values, y_mean, filepath, filename="deaton_table", T=10000, n_sims=30, seed=42):
    """
    Generate Deaton (1991) Table I: Standard deviations of consumption and income
    for different AR(1) autocorrelation parameters.
    
    Parameters:
    -----------
    model : object
        The consumption-savings model instance
    phi_values : list
        List of AR(1) autocorrelation coefficients to test
    y_mean : float
        Target unconditional mean of income (same across all phi values)
    filepath : str
        Path to save table
    filename : str
        Filename for saved table (without extension)
    T : int
        Number of periods to simulate for computing statistics
    n_sims : int
        Number of simulations to average over for each phi
    seed : int
        Random seed for reproducibility
    """
    results = {
        'phi': [],
        'sd_y': [],
        'est_sd_y': [],
        'est_sd_c': [],
        'ratio': []
    }
    
    # Store original parameters
    original_phi = model.phi
    original_mu = model.mu
    original_y_grid = model.y_grid.copy()
    original_Pi = model.Pi.copy()
    
    print("\n" + "="*60)
    print("GENERATING DEATON TABLE I")
    print(f"Target unconditional mean: {y_mean}")
    print(f"Number of simulations per phi: {n_sims}")
    print("="*60)
    
    for phi in phi_values:
        print(f"\nSolving model for φ = {phi:.1f}...")
        
        # Adjust mu to keep unconditional mean constant
        mu_adjusted = y_mean * (1 - phi)
        print(f"  Adjusted μ = {mu_adjusted:.2f}")
        
        # Update AR(1) process with new phi and adjusted mu
        model.phi = phi
        model.mu = mu_adjusted
        model.y_grid, model.Pi = tauchen(mu=mu_adjusted, phi=phi, sigma=model.sigma, 
                                         n_states=model.S, m=model.m)
        
        # Re-solve the model
        model.solve(max_iter=1000)
        
        # Run multiple simulations and average
        sd_y_sims = []
        est_sd_c_sims = []
        income_means = []
        
        for sim in range(n_sims):
            # Use different seed for each simulation
            sim_seed = seed + sim
            
            # Simulate lifecycle
            data = model.simulate_lifecycle(T=T, seed=sim_seed)
            
            income = data['income']
            consumption = data['consumption']
            
            # Store statistics for this simulation
            sd_y_sims.append(income.std())
            est_sd_c_sims.append(consumption.std())
            income_means.append(income.mean())
        
        # Average across simulations
        sd_y = np.mean(sd_y_sims)
        est_sd_c = np.mean(est_sd_c_sims)
        mean_income = np.mean(income_means)
        
        # 2. est sd(y): Theoretical unconditional std of AR(1) process
        if abs(phi) < 0.9999:  # Check for numerical stability
            est_sd_y = model.sigma / np.sqrt(1 - phi**2)
        else:
            est_sd_y = np.nan
        
        # 4. ratio 3/2
        ratio = est_sd_c / est_sd_y if est_sd_y > 0 else np.nan
        
        # Store results
        results['phi'].append(phi)
        results['sd_y'].append(sd_y)
        results['est_sd_y'].append(est_sd_y)
        results['est_sd_c'].append(est_sd_c)
        results['ratio'].append(ratio)
        
        print(f"  Average simulated mean income: {mean_income:.2f}")
        print(f"  Average sd(y) = {sd_y:.1f}")
        print(f"  est sd(y) = {est_sd_y:.1f}")
        print(f"  Average est sd(c) = {est_sd_c:.1f}")
        print(f"  ratio = {ratio:.2f}")
    
    # Restore original parameters
    model.phi = original_phi
    model.mu = original_mu
    model.y_grid = original_y_grid
    model.Pi = original_Pi
    
    # Re-solve with original parameters
    print("\nRestoring original model parameters and re-solving...")
    model.solve(max_iter=1000)
    
    # Generate LaTeX table in true AER style (matching Deaton's table)
    latex_table = r"""\begin{table}[htbp]\centering
\caption{Standard Deviations of Consumption and Income for AR(1) Income, Parameter $\phi$}
\label{tab:deaton_table1}
\begin{tabular}{l""" + "c" * len(phi_values) + r"""}
\hline\hline
AR coeff $\phi$ """
    
    # Add phi values as column headers
    for phi in phi_values:
        latex_table += f"& {phi:.1f} "
    latex_table += r""" \\
\hline
"""
    
    # Row 1: sd(y)
    latex_table += "1. sd($y$) "
    for sd_y in results['sd_y']:
        latex_table += f"& {sd_y:.1f} "
    latex_table += r""" \\
"""
    
    # Row 2: est sd(y)
    latex_table += "2. est sd($y$) "
    for est_sd_y in results['est_sd_y']:
        latex_table += f"& {est_sd_y:.1f} "
    latex_table += r""" \\
"""
    
    # Row 3: est sd(c)
    latex_table += "3. est sd($c$) "
    for est_sd_c in results['est_sd_c']:
        latex_table += f"& {est_sd_c:.1f} "
    latex_table += r""" \\
"""
    
    # Row 4: ratio 3/2
    latex_table += "ratio 3/2 "
    for ratio in results['ratio']:
        latex_table += f"& {ratio:.2f} "
    latex_table += r""" \\
"""
    
    latex_table += r"""\hline\hline
\end{tabular}
\end{table}
"""
    
    # Save table
    with open(f"{filepath}/{filename}.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to: {filepath}/{filename}.tex")
    
    return results


# Initialize the model with parameters
model = DeatonModel(
    phi=0.7,        # AR(1) coefficient (this will be overwritten in the loop)
    rho=2,          # CRRA parameter
    r=0.02,         # interest rate
    mu=30,          # AR(1) mean parameter
    delta=0.05,     # discount factor
    sigma=10,       # AR(1) standard deviation
    S=15)           # number of income states

# Solve the model with initial parameters
model.solve(max_iter=1000)

# Now generate the Deaton table
phi_values = [-0.4, 0.0, 0.3, 0.5, 0.7, 0.8]

results = generate_deaton_table(
    model=model,
    phi_values=phi_values,
    y_mean=100,  # Target unconditional mean for all processes
    filepath="C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output/tables/",
    filename="deaton_table1",
    T=10000,
    n_sims=2,  # Average over n simulations
    seed=42
)