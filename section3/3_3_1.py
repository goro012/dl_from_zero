import numpy


a = 0
b = numpy.array([1, 2, 3])
c = numpy.array([[1, 2, 3], [4, 5, 6]])

assert numpy.ndim(a) == 0
assert numpy.ndim(b) == 1
assert numpy.ndim(c) == 2
print("全部正解")