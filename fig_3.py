from deaton_model import DeatonModel
import matplotlib.pyplot as plt
import numpy as np

# Set your figure path
figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures"

model = DeatonModel(phi=.7, rho=2, r=.02, delta=.05, S=15)
model.solve(max_iter=1000)
model.plot_consumption_function(filepath=figure_path, filename="fig3")