import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from dataset.mnist import load_mnist

import numpy
import matplotlib.pyplot as plt
import tqdm


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    x = x - numpy.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=-1, keepdims=True)

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


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = numpy.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = numpy.dot(dout, self.W.T)
        self.dW = numpy.dot(self.x.T, dout)
        self.db = numpy.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class Relu:
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


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params["W1"] = weight_init_std * numpy.random.randn(input_size, hidden_size)
        self.params["b1"] = numpy.zeros(hidden_size)
        self.params["W2"] = weight_init_std * numpy.random.randn(hidden_size, output_size)
        self.params["b2"] = numpy.zeros(output_size)

        self.layers = {}
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)

        return X

    def loss(self, X, y_true):
        y_pred = self.predict(X)
        return self.lastLayer.forward(y_pred, y_true)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        y_pred = numpy.argmax(y_pred, axis=1)
        y_true = numpy.argmax(y_true, axis=1)
        
        accuracy = numpy.sum(y_pred == y_true) / len(y_pred)

        return accuracy

    def numerical_gradient(self, X, y_true):
        loss_W = lambda W: self.loss(X, y_true)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, X, y_true):
        
        # forward
        self.loss(X, y_true)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads


(X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

input_size = 784
hidden_size = 100
output_size = 10

network = TwoLayerNet(input_size, hidden_size, output_size)

X_batch = X_train[:3]
y_batch = y_train[:3]

numerical_grads = network.numerical_gradient(X_batch, y_batch)
bp_grads = network.gradient(X_batch, y_batch)

for key in numerical_grads.keys():
    print(f"{key}: {numpy.mean(numerical_grads[key] - bp_grads[key])}")
