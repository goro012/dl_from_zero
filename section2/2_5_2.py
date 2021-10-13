import numpy


def AND(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.7

    tmp = numpy.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([-0.5, -0.5])
    b = 0.7

    tmp = numpy.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([0.2, 0.2])
    b = -0.1

    tmp = numpy.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(s1, s2)
    y = AND(s1, s2)
    return y


assert AND(0, 0) == 0
assert AND(0, 1) == 0
assert AND(1, 0) == 0
assert AND(1, 1) == 1

assert NAND(0, 0) == 1
assert NAND(0, 1) == 1
assert NAND(1, 0) == 1
assert NAND(1, 1) == 0

assert OR(0, 0) == 0
assert OR(0, 1) == 1
assert OR(1, 0) == 1
assert OR(1, 1) == 1

assert XOR(0, 0) == 0
assert XOR(0, 1) == 1
assert XOR(1, 0) == 1
assert XOR(1, 1) == 0
print("全部正解")