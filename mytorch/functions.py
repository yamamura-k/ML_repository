from mytorch.core import as_variable
from mytorch import utils
import numpy as np
from mytorch import Function
class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x = self.inputs
        gx = cos(x)*gy
        return gx
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x = self.inputs
        gx = -sin(x)*gy
        return gx
class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx =  gy*(1-y*y)
        return gx

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y
    def backward(self, gy):
        return reshape(gy, self.x_shape)
class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    def backward(self, gy):
        gx = transpose(gy)
        return gx
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis = self.axis, keepdims=self.keepdims)
        return y
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy*diff*(2./len(diff))
        gx1 = -gx0
        return gx0, gx1
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb
class Sigmoid(Function):
    def forward(self, x):
        # y = 1 / (1 + np.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx
class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y
    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy*mask
        return gx

def exp(x):
    return Exp()(x)
def sin(x):
    x = as_variable(x)
    return Sin()(x)
def cos(x):
    x = as_variable(x)
    return Cos()(x)
def tanh(x):
    x = as_variable(x)
    return Tanh()(x)
def reshape(x, shape):
    if x.shape == shape:
        return x
    return Reshape(shape)(x)
def transpose(x):
    return Transpose()(x)
def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)
def matmul(x, W):
    return MatMul()(x, W)
def MSE(x0, x1):
    return MeanSquaredError()(x0, x1)
def linear(x, W, b):
    return Linear()(x, W, b)
def sigmoid(x):
    return Sigmoid()(x)
def relu(x):
    return ReLU()(x)