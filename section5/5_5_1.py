import numpy


class ReLu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

x = numpy.array([[1., -0.5], [-2., 3.]])
relu = ReLu()
out = relu.forward(x)
assert numpy.all(out == numpy.array([[1., 0], [0, 3.]]))

dx = relu.backward(numpy.array([[1, 1], [1, 1]]))
assert numpy.all(dx == numpy.array([[1., 0.], [0., 1.]]))