from deaton_model import DeatonModel
import matplotlib.pyplot as plt
import numpy as np

output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"

from deaton_model import DeatonModel

# Create and solve model
model = DeatonModel(phi=0.7, rho=2, r=0.02, mu=30, delta=0.05, S=15)
model.solve(max_iter=1000)

# Simulate the data
data = model.simulate_lifecycle(T=200, seed=42)

# Save the plot
model.save_lifecycle_plot(data, filepath=f"{output_path}/figures", filename="fig4")

# Save the table
model.save_lifecycle_table(data, filepath=f"{output_path}/tables", filename="table2")
