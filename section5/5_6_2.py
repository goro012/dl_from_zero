import numpy


class Affine:
    def __init__(self, W, b):
        self.W = w
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = numpy.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = numpy.dot(dout, self.W.T)
        self.dW = numpy.dot(self.x.T, dout)
        self.db = numpy.sum(dout, axis=0)