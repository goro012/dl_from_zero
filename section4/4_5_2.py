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

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params["W1"] = weight_init_std * numpy.random.randn(input_size, hidden_size)
        self.params["b1"] = numpy.zeros(hidden_size)
        self.params["W2"] = weight_init_std * numpy.random.randn(hidden_size, output_size)
        self.params["b2"] = numpy.zeros(output_size)

    def predict(self, X):

        a1 = numpy.dot(X, self.params["W1"]) + self.params["b1"]
        z1 = sigmoid(a1)
        a2 = numpy.dot(z1, self.params["W2"]) + self.params["b2"]
        y = softmax(a2)

        return y

    def loss(self, X, y_true):
        y_pred = self.predict(X)
        return cross_entropy_error(y_pred, y_true)

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
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = X.shape[0]
        
        # forward
        a1 = numpy.dot(X, W1) + b1
        z1 = sigmoid(a1)
        a2 = numpy.dot(z1, W2) + b2
        y_pred = softmax(a2)
        
        # backward
        dy = (y_pred - y_true) / batch_num
        grads['W2'] = numpy.dot(z1.T, dy)
        grads['b2'] = numpy.sum(dy, axis=0)
        
        dz1 = numpy.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = numpy.dot(X.T, da1)
        grads['b1'] = numpy.sum(da1, axis=0)

        return grads


(X_train, y_train), (X_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []


iters_num = 10000
batch_size = 100
lr = 0.01
iter_per_epoch = max(X_train.shape[0] / batch_size, 1)

input_size = 784
hidden_size = 100
output_size = 10

network = TwoLayerNet(input_size, hidden_size, output_size)

for i in tqdm.tqdm(range(iters_num)):
    batch_mark = numpy.random.choice(X_train.shape[0], batch_size)
    X_batch = X_train[batch_mark]
    y_batch = y_train[batch_mark]

    grads = network.gradient(X_batch, y_batch)

    for key in ["W1", "b1", "W2", "b2"]:
        network.params[key] -= lr * grads[key]

    loss = network.loss(X_batch, y_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(X_train, y_train)
        test_acc = network.accuracy(X_test, y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

plt.plot(range(iters_num), train_loss_list)
plt.show()

plt.plot(range(len(train_acc_list)), train_acc_list, label="train")
plt.plot(range(len(test_acc_list)), test_acc_list, label="test")
plt.legend()
plt.show()
