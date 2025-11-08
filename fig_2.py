from deaton_model import DeatonModel
from deaton_model import simulate_lifecycle
import matplotlib.pyplot as plt
import numpy as np

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures"

# Figure 2 
model = DeatonModel(r=0.05, delta=0.10, rho=3, mu=100, sigma=10)
model.solve(max_iter=3, verbose=True)

model.plot_consumption_function(filepath=figure_path, filename='c_func_rho=3_sigma=10')


simulate_lifecycle(model,T=200, filepath=figure_path, filename="fig2")

# Figure 4 
model1 = DeatonModel(rho=2, r=.02, delta=.05,mu=100, sigma=10, phi=.7, S=15)
model1.solve(max_iter=3)

simulate_lifecycle(model1,T=200, filepath=figure_path, filename="fig4")
