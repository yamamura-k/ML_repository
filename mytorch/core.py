import weakref
import contextlib
import numpy as np
from heapq import heappush, heappop
import mytorch

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
def no_grad():
    return using_config("enable_backprop", False)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)
class Config:
    enable_backprop = True
class Variable:
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
            """
            if isinstance(data, float) or isinstance(data, int):
                data = np.array(data)
            else:
                raise TypeError(f"{type(data)} is not supported")
            """
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1
    def clear_grad(self):
        self.grad = None
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        funcs = []
        seen = set()
        def add_func(f):
            if f in seen:
                return
            seen.add(f)
            heappush(funcs, f)
        add_func(self.creator)
        while funcs:
            f = heappop(funcs)
            gys = [output().grad for output in f.outputs]
            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                       add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
    @property
    def shape(self):
        return self.data.shape
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype
    @property
    def T(self):
        return mytorch.functions.transpose(self)
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return mytorch.functions.reshape(self, shape)
    def transpose(self):
        return mytorch.functions.transpose(self)
    def sum(self, axis=None, keepdims=False):
        return mytorch.functions.sum(self, axis, keepdims)
    def matmul(self, W):
        return mytorch.functions.matmul(self, W)
    def dot(self, other):
        return mytorch.functions.matmul(self, other)
    def __neg__(self):
        return neg(self)
    def __add__(self, other):
        return add(self, other)
    def __radd__(self, other):
        return add(self, other)
    def __sub__(self, other):
        return sub(self, other)
    def __rsub__(self, other):
        return rsub(self, other)
    def __mul__(self, other):
        return mul(self, other)
    def __rmul__(self, other):
        return mul(self, other)
    def __truediv__(self, other):
        return div(self, other)
    def __rtruediv__(self, other):
        return rdiv(self, other)
    def __len__(self):
        return len(self.data)
    def __pow__(self, c):
        return pow(self, c)
    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " "*9)
        return 'variable(' + p + ')'#f"variable({p})"

class Parameter(Variable):
    pass

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
               output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    def forward(self, x):
        raise NotImplementedError()
    def backward(self, gy):
        raise NotImplementedError()
    def __lt__(self, other):
        # heapqで管理するにあたり、降順で並べ替えたいのでこういう実装にしている。
        return self.generation >= other.generation
class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = mytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = mytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y
    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = mytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = mytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy*x1, gy*x0
        if self.x0_shape != self.x1_shape:
            gx0 = mytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = mytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        if self.x0_shape != self.x1_shape:
            gx0 = mytorch.functions.sum_to(gx0, self.x0_shape)
            gx1 = mytorch.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1
class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y = x**self.c
        return y
    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx
def neg(x):
    return Neg()(x)
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)
def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)
def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)
def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)
def pow(x, c):
    return Pow(c)(x)