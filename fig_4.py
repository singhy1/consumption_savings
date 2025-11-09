from deaton_model import DeatonModel
from deaton_model import simulate_lifecycle
import matplotlib.pyplot as plt
import numpy as np

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures"

# # Figure 4 
# model = DeatonModel(phi=.7, rho=2, r=.02, mu=30, delta=.05, S=15)
# model.solve(max_iter=1000)

# #simulate_lifecycle(model,T=200, filepath=figure_path, filename="fig4")


# model.simulate_lifecycle(T=200, seed=42, filepath=figure_path, filename='fig4')

from deaton_model import DeatonModel

# Create and solve model
model = DeatonModel(phi=0.7, rho=2, r=0.02, mu=30, delta=0.05, S=15)
model.solve(max_iter=1000)

# Simulate and plot lifecycle
results = model.plot_lifecycle(T=200, seed=42, 
                               filepath=figure_path, 
                               filename='fig4')