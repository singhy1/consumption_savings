from deaton_model import DeatonModel
import matplotlib.pyplot as plt
import numpy as np

output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"

# Figure 2 
model = DeatonModel(r=0.05, delta=0.15, rho=3, mu=100, sigma=10)
model.solve(max_iter=1000, verbose=True)

#model.plot_consumption_function(filepath=figure_path, filename='c_func_rho=3_sigma=10')

# Simulate the data
data = model.simulate_lifecycle(T=200, seed=42)

# Save the plot
model.save_lifecycle_plot(data, filepath=f"{output_path}/figures", filename="fig2")

# Save the table
model.save_lifecycle_table(data, filepath=f"{output_path}/tables", filename="table1")

