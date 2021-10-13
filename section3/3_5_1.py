import numpy


def softmax(x):
    exp_x = numpy.exp(x)
    sum_exp_x = numpy.sum(exp_x)
    return exp_x / sum_exp_x

x = numpy.array([0., 1.])
assert numpy.all(softmax(x) - numpy.array([1/3.7182818, 2.7182818/3.7182818]) < 1e-3)