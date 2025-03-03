import math

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define the equation parameters
k = 4.61512052
L = 100

# Create an array of x values
x_values = np.linspace(0, 90, 10)

# Calculate the corresponding y values
y_values = 2 + (1 / k) * np.log(1 - (x_values / L))
print(-np.log(1 - (x_values / L)))
# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label='y = 1 - (1 / k) * ln(1 - (x / L))')
plt.title('Plot of the Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-10, 110)  # Set y limits for better visibility
plt.axhline(0, color='gray', lw=0.5, ls='--')  # x-axis
plt.axvline(0, color='gray', lw=0.5, ls='--')  # y-axis
plt.grid()
plt.legend()
plt.ylim(-0.5, 2)  # Set y limits for better visibility
plt.xlim(0, 100)  # Set x limits for better visibility
# Save the plot
plt.savefig('Equation_plot3.png')  # Save the plot
plt.close()  # Close the plot to free memory
