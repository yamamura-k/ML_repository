from mytorch.utils import plot_dot_graph
import numpy as np
from mytorch import Variable, add, square
import unittest
def test1():
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(y.data, x.grad)
def test2():
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(y.data, x.grad)
def test3():
    x = Variable(np.array(3.0))
    y = Variable(np.array(2.0))
    z = add(square(x), square(y))
    z.backward()
    print(z.data, x.grad, y.grad)
def test4():
    for _ in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
def sphere(x, y):
    return x**2 + y**2
def matyas(x, y):
    z = 0.26*(x**2 + y**2) - 0.48*x*y
    return z
def goldstein(x, y):
    z = (1 + (x+y+1)**2*(19-14*x+3*x**2-14*y+6*x*y+3*y**2))*\
        (30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return z
def test5(func):
    x = Variable(1.0)
    y = Variable(1.0)
    z = func(x, y)
    z.backward()
    x.name="x"
    y.name="y"
    z.name="z"
    funcname=func.__name__
    print(x.grad, y.grad)
    plot_dot_graph(z, save_file=f"{funcname}.png")

test1()
test2()
test3()
test4()
test5(sphere)
test5(matyas)
test5(goldstein)