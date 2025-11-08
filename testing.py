# Yash Singh 

from deaton_model import DeatonModel
import numpy as np 

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures" 

# Parameterization 1 

model1 = DeatonModel(rho=2, sigma=10)
model1.solve(max_iter=3)
model1.plot_consumption_function(filepath=figure_path, filename='c_func_10')


model1 = DeatonModel(rho=2, sigma=10)
model1.solve(max_iter=100)
model1.plot_consumption_function(filepath=figure_path, filename='c_func_100')

model1 = DeatonModel(rho=2, sigma=10)
model1.solve(max_iter=1000)
model1.plot_consumption_function(filepath=figure_path, filename='c_func_1000')




import matplotlib.pyplot as plt

# Create a single figure with all three consumption functions
plt.figure(figsize=(12, 8))

# Parameterization 1: rho=2, sigma=10, max_iter=3
model1 = DeatonModel(rho=2, sigma=10)
model1.solve(max_iter=3, verbose=True)
# Plot for middle income state
s = model1.S // 2
plt.plot(model1.wprime[:, s], model1.c_old[:, s], 'b-', linewidth=2, label='ρ=2, σ=10 (3 iter)')

# Parameterization 2: rho=2, sigma=10, max_iter=100  
model2 = DeatonModel(rho=2, sigma=10)
model2.solve(max_iter=100, verbose=True)
plt.plot(model2.wprime[:, s], model2.c_old[:, s], 'r--', linewidth=2, label='ρ=2, σ=10 (100 iter)')

# Parameterization 3: rho=3, sigma=10, max_iter=1000
model3 = DeatonModel(rho=2, sigma=10)
model3.solve(max_iter=1000, verbose=True)
plt.plot(model3.wprime[:, s], model3.c_old[:, s], 'g-.', linewidth=2, label='ρ=3, σ=10 (1000 iter)')

# Add 45-degree line
w_range = np.linspace(0, np.max(model1.wprime), 100)
plt.plot(w_range, w_range, 'k--', alpha=0.5, label='45° line')

plt.xlabel('Cash on Hand (w)')
plt.ylabel('Consumption (c)')
plt.xlim(0, 250)
plt.ylim(0, 250)
plt.title('Deaton Consumption Functions: Different Risk Aversion and Iterations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the combined plot
plt.savefig(f"{figure_path}/combined_consumption_functions.pdf")
plt.show()




