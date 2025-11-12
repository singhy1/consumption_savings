
from deaton_model import DeatonModel
from utilities import tauchen, stationary_stats

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures" 

# testing the tauchen method 

import numpy as np

# Simple example with small numbers so we can see what's happening
N = 4  # number of asset grid points
S = 3  # number of income states

# Create sample grids
aprime_grid = np.array([0, 10, 20, 30])  # shape (4,)
y_grid = np.array([5, 15, 25])            # shape (3,)

aprime_column = aprime_grid[:, np.newaxis]  # shape (4,1)
#print(aprime_column)

y_row = y_grid[np.newaxis, :]                # shape (1, 3)
#print(y_row)

wprime_fast = aprime_column + y_row
print(wprime_fast)