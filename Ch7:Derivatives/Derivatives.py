import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Function definition
# f(x) = 2x^2
# -------------------------------------------------
def f(x):
    return 2 * x**2


# -------------------------------------------------
# Generate smooth curve for plotting
# -------------------------------------------------
x = np.arange(0, 5, 0.001)
y = f(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function: f(x) = 2x²")
plt.show()


# -------------------------------------------------
# Numerical derivative (approximation)
# -------------------------------------------------
x_point = 2.0
delta = 1e-4

numerical_derivative = (f(x_point + delta) - f(x_point)) / delta
print("Numerical derivative at x =", x_point, ":", numerical_derivative)


# -------------------------------------------------
# Analytical derivative
# f'(x) = 4x
# -------------------------------------------------
analytical_derivative = 4 * x_point
print("Analytical derivative at x =", x_point, ":", analytical_derivative)


# -------------------------------------------------
# Tangent line at x = 2
# y = mx + b
# -------------------------------------------------
slope = analytical_derivative
y_point = f(x_point)
bias = y_point - slope * x_point

def tangent_line(x):
    return slope * x + bias


# -------------------------------------------------
# Plot function + tangent line
# -------------------------------------------------
plt.plot(x, y, label="f(x) = 2x²")
plt.plot(x, tangent_line(x), label="Tangent at x = 2")
plt.scatter(x_point, y_point)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Derivative as Tangent Line")
plt.show()
