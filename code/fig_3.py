from deaton_model import DeatonModel
import matplotlib.pyplot as plt
import numpy as np

# Set your figure path
output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"

# Create model with CORRECTED parameters
model1 = DeatonModel(rho=2, r=0.02, delta=0.10, mu=30, sigma=10, phi=.7,S=15)

model1.solve(max_iter=1000)
model1.plot_consumption_function(filepath=f"{output_path}/figures", filename="fig3")