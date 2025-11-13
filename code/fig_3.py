from deaton_model import DeatonModel

# Set your figure path
output_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/output"

# Create model with CORRECTED parameters    
model1 = DeatonModel(rho=2, r=0.02, delta=0.05, mu=30, sigma=10, phi=.7,S=10)

model1.solve(max_iter=1000)
model1.plot_consumption_function(filepath=f"{output_path}/figures", filename="fig3")

   
#model2 = DeatonModel(rho=2, r=0.02, delta=0.05, mu=30, sigma=10, phi=.6,S=10)

#model2.solve(max_iter=1000)
#model2.plot_consumption_function(filepath=f"{output_path}/figures", filename="fig3B")