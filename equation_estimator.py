from scipy.optimize import minimize
import numpy as np

# Objective function to minimize
def objective(params):
    a, b, c = params
    error = (
        (20 - (a + b/(0 + c)))**2 +
        (1 - (a + b/(100 + c)))**2 +
        (5 - (a + b/(50 + c)))**2
    )
    return error

# Initial guess
initial_guess = [1, 1, 1]

# Using minimize function to find the best parameters
result = minimize(objective, initial_guess)

a_opt, b_opt, c_opt = result.x
print(f"Optimized Parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}")