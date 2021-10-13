import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from dataset.mnist import load_mnist
from PIL import Image
import numpy
import pickle


def identity_function(x):
    return x

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def softmax(x):
    max_x = numpy.max(x)
    exp_x = numpy.exp(x - max_x)
    sum_exp_x = numpy.sum(exp_x)
    return exp_x / sum_exp_x

def get_data():
    (X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return X_test, y_test


def init_network():
    file_name = os.path.join(os.path.dirname((os.path.abspath(__file__))), "sample_weight.pkl")
    with open(file_name, "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):

    a1 = numpy.dot(x, network["W1"]) + network["b1"]
    z1 = sigmoid(a1)
    a2 = numpy.dot(z1, network["W2"]) + network["b2"]
    z2 = sigmoid(a2)
    a3 = numpy.dot(z2, network["W3"]) + network["b3"]
    # y = softmax(a3)
    y = identity_function(a3)

    return y


X_test, y_test = get_data()
network = init_network()

# 1行ごと
accuracy_count = 0
for x, y_t in zip(X_test, y_test):
    y_pred = predict(network, x)
    p = numpy.argmax(y_pred)
    if p == y_t:
        accuracy_count += 1
print(f"accuracy rate: {accuracy_count / len(X_test)}")

# 一括
y_pred = predict(network, X_test)
p = numpy.argmax(y_pred, axis=1)
print(f"accuracy rate: {numpy.sum(p == y_test) / len(X_test)}")

# バッチ
batch_size = 100
accuracy_count = 0

for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]
    y_pred = predict(network, X_batch)
    y_p = numpy.argmax(y_pred, axis=1)
    accuracy_count += numpy.sum(y_p == y_batch)
print(f"accuracy rate: {accuracy_count / len(X_test)}")