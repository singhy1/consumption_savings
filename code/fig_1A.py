from deaton_model import DeatonModel
import matplotlib.pyplot as plt
import numpy as np

# Set your figure path
output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"

# Define model configurations
model_configs = [
    {"rho": 2, "sigma": 10, "label": r"$\rho=2, \sigma=10$", "color": "blue"},
    #{"rho": 2, "sigma": 15, "label": r"$\rho=2, \sigma=15$", "color": "red"},
    {"rho": 3, "sigma": 10, "label": r"$\rho=3, \sigma=10$", "color": "green"},
    #{"rho": 3, "sigma": 15, "label": r"$\rho=3, \sigma=15$", "color": "purple"}
]

# Solve all models and store results
models = []
print("Solving models...\n")

for i, config in enumerate(model_configs, 1):
    print(f"Model {i}: rho={config['rho']}, sigma={config['sigma']}")
    model = DeatonModel(r=0.05, delta=0.10, rho=config['rho'], sigma=config['sigma'])
    model.solve(max_iter=1000, verbose=False)
    
    # Print convergence info
    if model.converged:
        print(f"  ✓ Converged in {model.n_iterations} iterations")
    else:
        print(f"  ✗ Did not converge")
    
    print(f"  beta*R = {model.beta * model.R:.6f}\n")
    
    models.append((model, config))

# Create the comparison plot
print("Creating comparison plot...")
plt.figure(figsize=(12, 8))

# Plot consumption functions for each model
for model, config in models:
    # For each model, plot all income state consumption functions
    # We'll use a semi-transparent style so overlapping lines are visible
    for s in range(model.S):
        plt.plot(model.wprime[:, s], model.c_old[:, s], 
                color=config['color'], alpha=0.3, linewidth=1)
    
    # Plot one representative line (middle income state) with full opacity for legend
    mid_state = model.S // 2
    plt.plot(model.wprime[:, mid_state], model.c_old[:, mid_state],
            color=config['color'], alpha=1.0, linewidth=2, label=config['label'])

# Add 45-degree line
w_range = np.linspace(0, 250, 100)
plt.plot(w_range, w_range, 'k--', alpha=0.5, linewidth=1.5)

# ============================================================
# NEW: Vertical line at w = 100
# ============================================================
plt.axvline(x=100, color='black', linestyle='-', linewidth=2)


# ============================================================
# NEW: Line c = (mu + rw)(1+r)
# ============================================================
r = 0.05
c_line = (100 + r*w_range) / (1 + r)
plt.plot(w_range, c_line, color='black', linestyle='-', linewidth=2,
         label=r"$c = (\mu + rw)/(1+r)$")


# Formatting
plt.xlabel('Cash on Hand (w)', fontsize=18)
plt.ylabel('Consumption (c)', fontsize=18)
plt.xlim(0, 200)
plt.ylim(0, 200)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Consumption Functions', fontsize=22)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig(f"{output_path}/figures/fig1A.pdf", dpi=600)
plt.close()  # Close the figure without showing it
print(f"Figure saved to: {output_path}/fig1A.pdf")

print("\nAll done!")