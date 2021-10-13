import numpy


A = numpy.array([[1, 2], [3, 4]])
B = numpy.array([[5, 6], [7, 8]])

assert numpy.all(numpy.dot(A, B) == numpy.array([[19, 22], [43, 50]]))