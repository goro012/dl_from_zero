import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network["W1"] = numpy.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = numpy.array([0.1, 0.2, 0.3])
    network["W2"] = numpy.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = numpy.array([0.1, 0.2])
    network["W3"] = numpy.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = numpy.array([0.1, 0.2])

    return network

def forward(network, x):

    a1 = numpy.dot(x, network["W1"]) + network["b1"]
    z1 = sigmoid(a1)
    a2 = numpy.dot(z1, network["W2"]) + network["b2"]
    z2 = sigmoid(a2)
    a3 = numpy.dot(z2, network["W3"]) + network["b3"]
    y = identity_function(a3)

    return y


network = init_network()
x = numpy.array([1.0, 0.5])
y = forward(network, x)

assert numpy.all(y - numpy.array([0.31682708, 0.69627909]) < 1e-4)