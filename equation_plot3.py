import numpy as np
import matplotlib.pyplot as plt

# Optimized parameters
a = 0.5
b = 75.0
c = 50.0

# Define the function for y
def equation(x):
    return a + b / (x + c)

# Create an array of x values
x_values = np.linspace(0, 100, 200)  # x values from 0 to 100
y_values = equation(x_values)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r'$y = 0.5 + \frac{75}{x + 50}$', color='blue')
plt.scatter([0, 100, 50], [2, 1, 1.25], color='red', label='Data Points', zorder=5)  # Points to match original conditions
plt.title('Plot of the Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(0, 3)  # Set y limits for better visibility
plt.axhline(0, color='gray', lw=0.5, ls='--')  # x-axis
plt.axvline(0, color='gray', lw=0.5, ls='--')  # y-axis
plt.grid()
plt.legend()
plt.ylim(-0.5, 2)  # Set y limits for better visibility
plt.xlim(0, 100)  # Set x limits for better visibility
# Save the plot
plt.savefig('Equation_plot3.png')  # Save the plot
plt.close()  # Close the plot to free memory
