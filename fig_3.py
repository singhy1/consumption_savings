from deaton_model import DeatonModel
import matplotlib.pyplot as plt
import numpy as np

# Set your figure path
figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures"

#model = DeatonModel(phi=.7, rho=2, r=.02, mu=30, delta=.05, S=15)
#model.solve(max_iter=1000)
#model.plot_consumption_function(filepath=figure_path, filename="fig3")


mean_income = 100
phi = 0.7
innovation_std = 10

# Correct mu parameter for tauchen
mu_constant = mean_income * (1 - phi)  # = 100 * 0.3 = 30

print(f"Income process parameters:")
print(f"  Unconditional mean: {mean_income}")
print(f"  mu constant: {mu_constant}")
print(f"  phi: {phi}")
print(f"  innovation std: {innovation_std}")
print(f"  unconditional std: {innovation_std / np.sqrt(1 - phi**2):.2f}")

# Create model with CORRECTED parameters
model1 = DeatonModel(
    rho=2,           # Risk aversion
    r=0.02,          # Interest rate
    delta=0.10,      # Time preference
    mu=mu_constant,  # ← FIXED: This is the constant term, not the mean!
    sigma=innovation_std,
    phi=phi,
    S=15
)

model1.solve(max_iter=1000)
model1.plot_consumption_function(filepath=figure_path, filename="fig3")