def AND(x1, x2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.7
    
    tmp = w1 * x1 + w2 * x2
    if tmp <= theta:
        return 0
    else:
        return 1

assert AND(0, 0) == 0
assert AND(0, 1) == 0
assert AND(1, 0) == 0
assert AND(1, 1) == 1
print("全部正解")