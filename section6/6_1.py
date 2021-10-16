import numpy
import matplotlib.pyplot as plt


class SGD:
    def __init__(self, lr=0.9):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = numpy.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.9):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = numpy.zeros_like(val)

        for key in params.keys():
            print(grads[key])
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (numpy.sqrt(self.h[key]) + 1e-7)


def function1(x):
    return (x[0]**2) / 20 + x[1]**2

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = numpy.zeros_like(x)
    
    it = numpy.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad


x_history = []
params = {}
grad = {}
optimizer = Momentum()

params["x"] = numpy.array([-7., 2.])
x_history.append(params["x"].copy())

for i in range(100):
    grad["x"] = numerical_gradient(function1, params["x"])

    optimizer.update(params, grad)
    x_history.append(params["x"].copy())

x_history = numpy.array(x_history)
plt.scatter(x_history[:,0], x_history[:,1])
plt.show()
