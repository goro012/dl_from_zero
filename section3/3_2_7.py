import numpy
import matplotlib.pyplot as plt


def ReLU(x):
    return numpy.maximum(0, x)


x = numpy.arange(-5.0, 5.0, 0.1)
y = ReLU(x)
plt.plot(x, y)
plt.show()
