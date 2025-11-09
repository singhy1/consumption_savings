
from deaton_model import DeatonModel
from utilities import tauchen, stationary_stats

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures" 

# testing the tauchen method 


phi=0.7
rho=2
r=0.02 
mu=30
delta=0.05
S=15
sigma = 10

y_grid, Pi = tauchen(mu=mu, phi=phi, sigma=sigma, 
                                               n_states=S, m=4) 
print("income grid:", y_grid)
print("transition matrix:", Pi)

pi, y_mean, y_std = stationary_stats(Pi, y_grid)

print(pi)