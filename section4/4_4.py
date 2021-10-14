import numpy
import matplotlib.pyplot as plt


def numerical_gradient(f, x):
    h = 1e-4
    grad = numpy.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

x = numpy.array([1., 3.])
# print(f"y={function_2(x)}")
print(f"grad={numerical_gradient(function_2, x)}")
