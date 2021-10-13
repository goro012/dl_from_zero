import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def identity_function(x):
    return x

X = numpy.array([1.0, 0.5])
W1 = numpy.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = numpy.array([0.1, 0.2, 0.3])

A1 = numpy.dot(X, W1) + B1
Z1 = sigmoid(A1)

W2 = numpy.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = numpy.array([0.1, 0.2])

A2 = numpy.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

W3 = numpy.array([[0.1, 0.3], [0.2, 0.4]])
B3 = numpy.array([0.1, 0.2])

A3 = numpy.dot(Z2, W3) + B3
Z3 = identity_function(A3)

assert numpy.all(Z3 - numpy.array([0.31682708, 0.69627909]) < 1e-4)