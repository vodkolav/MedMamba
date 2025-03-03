import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define the equation parameters
k = 8.046
L = 100

# Create an array of x values
x_values = np.linspace(0, 2, 100)  # From 0 to 5

# Calculate the corresponding y values
y_values = L * (1 - np.exp(-k * (x_values - 1)))

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label=r'$y = 100(1 - e^{-8.046(x - 1)})$', color='blue')
plt.title('Plot of the Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-10, 110)  # Set y limits for better visibility
plt.axhline(0, color='gray', lw=0.5, ls='--')  # x-axis
plt.axvline(0, color='gray', lw=0.5, ls='--')  # y-axis
plt.grid()
plt.legend()
# Save the plot
plt.savefig('Equation_plot.png')  # Save the plot
plt.close()  # Close the plot to free memory
