import numpy as np 
from utilities import utility, u_prime, inv_uprime, tauchen
from scipy.interpolate import PchipInterpolator 
import matplotlib.pyplot as plt


class DeatonModel:
    """
    Solves the Deaton (1991) consumption-savings problem using the Endogenous Grid Method (EGM).
    """
    
    def __init__(self, r=0.05, delta=0.10, gamma=2, y_mean=100, y_std=10, 
                 N=100, a_max=100, S=10, phi=0.0):
        """
        Initialize the Deaton consumption-savings model.
        
        Parameters:
        -----------
        r : float
            Real interest rate
        delta : float
            Rate of time preference
        gamma : float
            Coefficient of relative risk aversion
        y_mean : float
            Mean income level
        y_std : float
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
        self.beta = (1 + r) / (1 + delta)
        self.gamma = gamma
        
        # Asset grid
        self.N = N
        self.a_min = 0
        self.a_max = a_max
        self.aprime_grid = np.linspace(self.a_min, self.a_max, self.N)
        
        # Income states
        self.S = S
        self.y_mean = y_mean
        self.y_std = y_std
        self.y_grid_iid, self.Pi_iid = tauchen(mu=y_mean, phi=phi, sigma=y_std, 
                                               n_states=S, m=9)
        
        # Create w' grid
        self.wprime = np.zeros((self.N, self.S))
        for n in range(self.N):
            for s in range(self.S):
                self.wprime[n, s] = self.aprime_grid[n] + self.y_grid_iid[s]
        
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
                        # Direct lookup - no interpolation needed!
                        c_prime = self.c_old[n, s_prime]
                        
                        # Marginal utility
                        u_prime_val = u_prime(c_prime, self.gamma)
                        
                        # Weight by transition probability
                        expected_marginal_utility += self.Pi_iid[s, s_prime] * u_prime_val
                    
                    RHS[n, s] = self.beta * self.R * expected_marginal_utility

            # Step 5: Invert to get current consumption
            c_new = inv_uprime(RHS, self.gamma)
            
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
                if self.y_grid_iid[s] < w_endo_sorted[0]:
                    w_with_constraint = np.concatenate([[self.y_grid_iid[s]], w_endo_sorted])
                    c_with_constraint = np.concatenate([[self.y_grid_iid[s]], c_new_sorted])
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
    
    def plot_consumption_function(self, save=True, filename='consumption_function.png'):
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
        plt.xlim(0, 250)
        plt.ylim(0, 250)
        plt.title('Consumption Function (Deaton Model)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        
        plt.savefig('figures/consumption_function.png', dpi=300, bbox_inches='tight')
        
    
        plt.show()
# Usage
if __name__ == "__main__":
    model = DeatonModel()
    model.solve(max_iter=10, verbose=True)
    model.plot_consumption_function()