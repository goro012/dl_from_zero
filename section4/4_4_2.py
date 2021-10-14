import numpy
import matplotlib.pyplot as plt


def softmax(x):
    max_x = numpy.max(x)
    exp_x = numpy.exp(x - max_x)
    return exp_x / numpy.sum(exp_x)

def cross_entropy_error(y_pred, y_true):
    if type(y_pred) != numpy.ndarray:
        y_pred = numpy.array(y_pred)
    if type(y_true) != numpy.ndarray:
        y_true = numpy.array(y_true)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(1, y_pred.size)
        y_true = y_true.reshape(1, y_true.size)


    delta = 1e-7
    batch_size = y_pred.shape[0]
    return -numpy.sum(y_true * numpy.log(y_pred + delta)) / batch_size

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

def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x


class SimpleNet:
    def __init__(self):
        self.W = numpy.random.randn(2, 3)

    def predict(self, X):
        return numpy.dot(X, self.W)

    def loss(self, X, y_true):
        y_pred = self.predict(X)
        y_pred = softmax(y_pred)
        loss = cross_entropy_error(y_pred, y_true)
        return loss



X = numpy.array([0.6, 0.9])
y_true = numpy.array([0, 0, 1])
net = SimpleNet()

f = lambda w: net.loss(X, y_true)
dW = numerical_gradient(f, net.W)
print(dW)


