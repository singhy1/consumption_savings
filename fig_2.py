from deaton_model import DeatonModel
from deaton_model import simulate_lifecycle
import matplotlib.pyplot as plt
import numpy as np

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures"

# Figure 2 
model = DeatonModel(r=0.05, delta=0.15, rho=3, mu=100, sigma=10)
model.solve(max_iter=1000, verbose=True)

#model.plot_consumption_function(filepath=figure_path, filename='c_func_rho=3_sigma=10')


#simulate_lifecycle(model,T=200, filepath=figure_path, filename="fig2")

# Simulate and plot lifecycle
results = model.plot_lifecycle(T=200, seed=42, 
                               filepath=figure_path, 
                               filename='fig2')