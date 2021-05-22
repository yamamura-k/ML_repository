import numpy as np
from mytorch import Variable
import mytorch.functions as F
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1)
Y = None
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))
def predict(x):
    y = F.matmul(x, W) + b
    return y
def MSE(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)
lr = 0.0001
iters = 100000

for _ in range(iters):
    y_pred = predict(x)
    loss = MSE(y, y_pred)
    W.clear_grad()
    b.clear_grad()
    loss.backward()
    W.data -= lr*W.grad.data
    b.data -= lr*b.grad.data
    print(W, b, loss)