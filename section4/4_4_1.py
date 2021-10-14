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

def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x

def gradient_descent_with_history(f, init_x, lr=0.05, step_num=100):
    x = init_x

    rtn = [x.copy()]
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        rtn.append(x.copy())
    
    return numpy.array(rtn)


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = numpy.array([3., 4.])
x = gradient_descent_with_history(function_2, init_x)

plt.scatter(x[:,0], x[:,1])
plt.show()
