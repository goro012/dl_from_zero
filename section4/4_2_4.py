import sys
import os
sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
from dataset.mnist import load_mnist
from PIL import Image
import numpy
import pickle


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

y_true = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_pred = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

assert numpy.sum(y_pred) == 1.
assert numpy.all(cross_entropy_error(y_pred, y_true) - 0.510815457 < 1e-3)