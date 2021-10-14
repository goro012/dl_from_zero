import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

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


data_num = 10
input_size = 100
hidden_size = 10
output_size = 3

net = TwoLayerNet(input_size, hidden_size, output_size)
print(net.params["W1"].shape)
print(net.params["b1"].shape)
print(net.params["W2"].shape)
print(net.params["b2"].shape)

X = numpy.random.rand(data_num, input_size)
y_pred = net.predict(X)
y_true = numpy.random.rand(data_num, output_size)

grads = net.numerical_gradient(X, y_true)
print(grads["W1"].shape)
print(grads["b1"].shape)
print(grads["W2"].shape)
print(grads["b2"].shape)





