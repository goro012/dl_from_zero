class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + numpy.exp(-x))
        return self.out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx