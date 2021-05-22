import weakref
import contextlib
import numpy as np
from heapq import heappush, heappop

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
def as_array(y):
    if np.isscalar(y):
        return np.array(y)
    return y
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Config:
    enable_backprop = True

class Variable:
    __array__priority = 200
    def __init__(self, data, name=None):
        if data is not None and not isinstance(data,np.ndarray):
            if isinstance(data, float) or isinstance(data, int):
                data = np.array(data)
            else:
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
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
        if self.data:
            p = str(self.data).replace("\n", "\n" + " "*9)
            return f"variable({p})"
        return "variable(None)"
    def set_creator(self, f):
        self.creator = f
        self.generation = f.generation + 1
    def clear_grad(self):
        self.grad = None
    def backward(self, no_grad=True, create_graph=False):
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
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            with using_config("enable_backprop", create_graph):
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad += gx
                    if x.creator is not None:
                       add_func(x.creator)
                if no_grad:
                    for y in f.outputs:
                        y().grad = None
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [inp.data for inp in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = ys,
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
               output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
 
    def forward(self, xs):
        raise NotImplementedError()
    def backward(self, gys):
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
        y = x0 + x1
        return y
    def backward(self, gy):
        return gy, gy
class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    def backward(self, gy):
        return gy, -gy
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        return x1*gy, x0*gy
class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy/x1
        gx1 = gy*(-x0/x1**2)
        return gx0, gx1
class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y = x**self.c
        return y
    def backward(self, gy):
         x = self.inputs
         c = self.c
         gx = c*x**(c-1)*gy
         return gx

class Square(Function):
    def forward(self, x):
        y = x**2
        return y
    def backward(self, gy):
         x = self.inputs
         gx = 2*x*gy
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

def square(x):
    return Square()(x)