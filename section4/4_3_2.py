import numpy
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01 * x**2 + 0.01 * x


x = numpy.arange(0, 20, 0.1)
y = function_1(x)
plt.plot(x, y)

x_1 = 10
y = numerical_diff(function_1, x_1) * (x - x_1) + function_1(x_1)
plt.plot(x, y)

plt.show()