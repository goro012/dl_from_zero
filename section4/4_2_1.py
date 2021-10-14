import numpy

def mean_squared_error(y_pred, y_true):
    if type(y_pred) != numpy.ndarray:
        y_pred = numpy.array(y_pred)
    if type(y_true) != numpy.ndarray:
        y_true = numpy.array(y_true)
    return 0.5 * numpy.sum((y_pred - y_true)**2)


def cross_entropy_error(y_pred, y_true):
    if type(y_pred) != numpy.ndarray:
        y_pred = numpy.array(y_pred)
    if type(y_true) != numpy.ndarray:
        y_true = numpy.array(y_true)

    delta = 1e-7
    return -numpy.sum(y_true * numpy.log(y_pred + delta))

y_true = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_pred = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

assert numpy.sum(y_pred) == 1.
assert numpy.all(mean_squared_error(y_pred, y_true) - 0.0975 < 1e-3)
assert numpy.all(cross_entropy_error(y_pred, y_true) - 0.510815457 < 1e-3)