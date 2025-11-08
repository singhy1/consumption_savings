
from deaton_model import DeatonModel

figure_path = "C:/Users/yash2/OneDrive/Desktop/phd_classes/macro_1/consumption_savings/figures" 

model1 = DeatonModel(rho=3, sigma=15)
model1.solve(max_iter=10)
model1.plot_consumption_function(filepath=figure_path, filename='c_func_rho=3_sigma=15')
