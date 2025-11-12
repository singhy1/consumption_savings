import numpy as np 
from utilities import utility, u_prime, inv_uprime, tauchen, simulate_markov_chain, stationary_stats
from scipy.interpolate import PchipInterpolator 
import matplotlib.pyplot as plt
import pandas as pd 


class DeatonModel:
    """
    Solves the Deaton (1991) consumption-savings problem using the Endogenous Grid Method (EGM).
    """
    
    def __init__(self, r=0.05, delta=0.10, rho=2, mu=100, sigma=10, 
                 N=100, a_max=100, S=10, m=4, phi=0.0):
        """
        Initialize the Deaton consumption-savings model.
        
        Parameters:
        -----------
        r : float
            Real interest rate
        delta : float
            Rate of time preference
        rho : float
            Coefficient of relative risk aversion
        mu : float
            Mean income level
        sigma : float
            Standard deviation of income
        N : int
            Number of asset grid points
        a_max : float
            Maximum asset level
        S : int
            Number of income states
        phi : float
            Persistence of income process (0 for IID)
        """
        # Deaton's parameters
        self.r = r
        self.R = 1 + r
        self.delta = delta
        self.beta = 1/(1+delta)
        self.rho = rho
        
        # Asset grid
        self.N = N
        self.a_min = 0
        self.a_max = a_max
        self.aprime_grid = np.linspace(self.a_min, self.a_max, self.N)
        
        # Income states
        self.S = S
        self.m = m
        self.mu = mu
        self.phi = phi
        self.sigma = sigma
        self.y_grid, self.Pi = tauchen(mu=mu, phi=phi, sigma=sigma, 
                                               n_states=S, m=m)

        # cash on hand grid 
        self.wprime= self.aprime_grid[:,np.newaxis] + self.y_grid[np.newaxis,:]
        
        # Initialize consumption function - consume everything
        self.c_old = self.wprime.copy()
        
        # Tracking
        self.converged = False
        self.n_iterations = 0
        
    def solve(self, max_iter=10, tol=1e-6, verbose=True):
        """
        Solve the model using EGM.
        
        Parameters:
        -----------
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress
            
        Returns:
        --------
        c_old : np.ndarray
            Consumption policy function
        """
        for iteration in range(max_iter):
            if verbose:
                print(f"\nIteration {iteration}")
            
            # Step 4: Compute RHS of Euler equation
            RHS = np.zeros((self.N, self.S))
            
            for n in range(self.N):
                for s in range(self.S):
                    expected_marginal_utility = 0.0
                    
                    for s_prime in range(self.S):
                        
                        c_prime = self.c_old[n, s_prime]
                        
                        # Marginal utility
                        u_prime_val = u_prime(c_prime, self.rho)
                        
                        # Weight by transition probability
                        expected_marginal_utility += self.Pi[s, s_prime] * u_prime_val
                    
                    RHS[n, s] = self.beta * self.R * expected_marginal_utility

            # Step 5: Invert to get current consumption
            c_new = inv_uprime(RHS, self.rho)
            
            # Step 6: Recover endogenous wealth grid
            w_endo = self.aprime_grid[:, None] / self.R + c_new
            
            # Step 7: Interpolate back to exogenous grid
            c_interp = np.zeros_like(self.c_old)
            
            for s in range(self.S):
                # Sort by endogenous wealth (maintaining the pairing!)
                sort_idx = np.argsort(w_endo[:, s])
                w_endo_sorted = w_endo[sort_idx, s]
                c_new_sorted = c_new[sort_idx, s]
                
                # Add borrowing constraint point only if it's below the minimum endogenous wealth
                # At the constraint: w = y(s), c = y(s) (consume all income, save nothing)
                if self.y_grid[s] < w_endo_sorted[0]:
                    w_with_constraint = np.concatenate([[self.y_grid[s]], w_endo_sorted])
                    c_with_constraint = np.concatenate([[self.y_grid[s]], c_new_sorted])
                else:
                    # Constraint is not binding in this region
                    w_with_constraint = w_endo_sorted
                    c_with_constraint = c_new_sorted
                
                # Check that we have enough points for interpolation
                if len(w_with_constraint) < 2:
                    c_interp[:, s] = self.wprime[:, s]  # Fallback: consume everything
                    continue
                
                # Interpolate to exogenous grid
                interp_func = PchipInterpolator(w_with_constraint, 
                                               c_with_constraint, 
                                               extrapolate=False)
                
                c_interp[:, s] = interp_func(self.wprime[:, s])
                
                # Fill NaNs from extrapolation regions
                below_range = self.wprime[:, s] < w_with_constraint[0]
                above_range = self.wprime[:, s] > w_with_constraint[-1]
            
                c_interp[below_range, s] = self.wprime[below_range, s]  # Consume everything at constraint
                c_interp[above_range, s] = c_with_constraint[-1]   # Use last value (or linear)
                
                # Enforce constraints: consumption cannot exceed wealth or be negative
                c_interp[:, s] = np.clip(c_interp[:, s], 0, self.wprime[:, s])

            # Check convergence
            diff = np.max(np.abs(c_interp - self.c_old))
            if verbose:
                print(f"Max difference: {diff:.6f}")
            
            if diff < tol:
                if verbose:
                    print("Converged!")
                self.converged = True
                self.n_iterations = iteration + 1
                break
                
            self.c_old = c_interp.copy()

        if verbose and not self.converged:
            print(f"\nWarning: Did not converge after {max_iter} iterations")
        
        return self.c_old
    
    def plot_consumption_function(self, filepath, filename, save=True):
        """
        Plot the consumption function for all income states.
    
        Parameters:
        -----------
        save : bool
            If True, save the figure to figures/ folder
        filename : str
            Name of the file to save
        """
        if self.c_old is None:
            raise ValueError("Model not solved yet. Call solve() first.")
    
        # Create figures directory if it doesn't exist
    
        plt.figure(figsize=(10, 6))
        for s in range(self.S):
            plt.plot(self.wprime[:, s], self.c_old[:, s], 'b-', alpha=0.7, linewidth=1)

        # Add 45-degree line
        w_range = np.linspace(0, np.max(self.wprime), 100)
        plt.plot(w_range, w_range, 'k--', alpha=0.5, label='45° line')

        plt.xlabel('Cash on Hand (w)')
        plt.ylabel('Consumption (c)')
        plt.xlim(0, 350)
        plt.ylim(0, 350)
        plt.title('Consumption Function (Deaton Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        
        plt.savefig(f"{filepath}/{filename}.pdf")
        
    def simulate_lifecycle(self, T=200, seed=42):
        """
        Simulate lifecycle paths for income, consumption, and assets.

        Parameters:
        -----------
        T : int
            Number of periods to simulate
        seed : int
            Random seed for reproducibility
    
        Returns:
        --------
        dict with keys: 'income', 'consumption', 'assets'
        """
        if self.c_old is None:
            raise ValueError("Model not solved yet. Call solve() first.")

        # Step 1: Simulate income process
        pi_stat, _, _ = stationary_stats(self.Pi, self.y_grid)
        income = simulate_markov_chain(self.Pi, self.y_grid, pi_stat, T=T, seed=seed)

        # Step 2: Initialize arrays
        consumption = np.zeros(T)
        assets = np.zeros(T + 1)
        assets[0] = 0.0  # Start with zero assets

        # Step 3: Simulate consumption and assets using policy function
        for t in range(T):
            # Find income state (closest grid point)
            s = np.argmin(np.abs(self.y_grid - income[t]))
    
            # Cash on hand
            w = assets[t] + income[t]
    
            # Look up consumption policy for this state
            w_grid = self.wprime[:, s]
            c_policy = self.c_old[:, s]
    
            # Interpolate if necessary
            w_clamped = np.clip(w, w_grid.min(), w_grid.max())
            c_t = np.interp(w_clamped, w_grid, c_policy)
            c_t = np.clip(c_t, 0, w)
            consumption[t] = c_t
    
            # Update assets via budget constraint
            assets[t + 1] = self.R * (w - c_t)
            assets[t + 1] = max(assets[t + 1], 0.0)

        # Trim last asset entry
        assets = assets[:-1]

        # Print summary statistics
        print("\n" + "="*60)
        print("LIFECYCLE SIMULATION SUMMARY")
        print("="*60)
        print(f"Income:       mean={income.mean():.2f}, std={income.std():.2f}")
        print(f"Consumption:  mean={consumption.mean():.2f}, std={consumption.std():.2f}")
        print(f"Assets:       mean={assets.mean():.2f}, std={assets.std():.2f}")

        return {
            'income': income,
            'consumption': consumption,
            'assets': assets
        }


    def save_lifecycle_plot(self, data, filepath, filename):
        """
        Plot and save lifecycle simulation results.

        Parameters:
        -----------
        data : dict
            Dictionary with 'income', 'consumption', 'assets' arrays
        filepath : str
            Path to save figure
        filename : str
            Filename for saved figure (without extension)
        """
        income = data['income']
        consumption = data['consumption']
        assets = data['assets']
        T = len(income)

        plt.figure(figsize=(12, 7))

        time = np.arange(T)
        plt.plot(time, income, label='Income', 
         color='black', linewidth=1.5, linestyle='-')
        plt.plot(time, consumption - 40, label='Consumption - 40', 
         color='black', linewidth=1.5, linestyle='--')
        plt.plot(time, assets, label='Assets', 
         color='black', linewidth=1.5, linestyle=':')

        plt.xlabel("Period", fontsize=12)
        plt.ylabel("Level", fontsize=12)
        plt.title("Lifecycle Simulation (Deaton 1991)", fontsize=13)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, T)
        plt.ylim(0, 150)
        plt.tight_layout()

        plt.savefig(f"{filepath}/{filename}.pdf", dpi=300)
        print(f"Figure saved to: {filepath}/{filename}.pdf")
        plt.close()

    def save_lifecycle_table(self, data, filepath, filename):
        """
        Generate and save AER-style LaTeX summary table.

        Parameters:
        -----------
        data : dict
            Dictionary with 'income', 'consumption', 'assets' arrays
        filepath : str
            Path to save table
        filename : str
            Filename for saved table (without extension)
        """
        income = data['income']
        consumption = data['consumption']
        assets = data['assets']

        # Create AER-style LaTeX table
        latex_table = r"""\begin{table}[htbp]
    \centering
    \caption{Lifecycle Simulation: Summary Statistics}
    \label{tab:lifecycle_summary}
    \begin{tabular}{@{\extracolsep{5pt}}lcc}
    \hline\hline
    \\[-1.8ex]
     & Mean & Std. Dev. \\
    \hline
    \\[-1.8ex]
    """
    
        latex_table += f"Income & {income.mean():.2f} & {income.std():.2f} \\\\\n"
        latex_table += f"Consumption & {consumption.mean():.2f} & {consumption.std():.2f} \\\\\n"
        latex_table += f"Assets & {assets.mean():.2f} & {assets.std():.2f} \\\\\n"
    
        latex_table += r"""\\[-1.8ex]
    \hline\hline
    \end{tabular}
    \begin{tablenotes}[para,flushleft]
    \small
    \textit{Note:} Statistics computed from a simulated lifecycle of """ + f"{len(income)}" + r""" periods.
    The consumer faces an AR(1) income process and makes optimal consumption-saving decisions.
    \end{tablenotes}
    \end{table}
    """

        with open(f"{filepath}/{filename}.tex", "w") as f: 
            f.write(latex_table)

        print(f"LaTeX table saved to: {filepath}/{filename}.tex")