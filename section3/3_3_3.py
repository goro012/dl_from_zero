import numpy


X = numpy.array([1, 2])
W = numpy.array([[1, 3, 5], [2, 4, 6]])

Y = numpy.dot(X, W)

assert numpy.all(Y == numpy.array([5, 11, 17]))