from mytorch.core import as_variable
from mytorch import utils
import numpy as np
from mytorch import Function
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