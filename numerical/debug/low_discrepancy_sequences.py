import numpy as np
import matplotlib.pyplot as plt
from math import *


""" RANDOM FUNCTIONS """


def randint(n):
    n = np.uint32(n)
    n = np.uint32(((n>>16)^n)*0x45d9f3b)
    n = np.uint32(((n>>16)^n)*0x45d9f3b)
    return np.uint32((n>>16)^n)


def randfloat(n):
    return randint(n) / 4294967296.0


def randfloat_2d(n):
    x = randfloat(n)
    y = randfloat(randint(n))
    return np.array([x, y])


def randfloat_3d(n):
    x = randfloat(n)
    y = randfloat(randint(n))
    z = randfloat(randint(randint(n)))
    return np.array([x, y, z])


def vandercorput_base(n, b):
    x = 0.0
    e = 1.0 / b
    while n != 0:
        d = n % b
        x += d * e
        e /= b
        n //= b
    return x


def vandercorput(n):
    return vandercorput_base(n, 2)


def halton_2d(n):
    x = vandercorput_base(n, 2)
    y = vandercorput_base(n, 3)
    return np.array([x, y])


def halton_3d(n):
    x = vandercorput_base(n, 2)
    y = vandercorput_base(n, 3)
    z = vandercorput_base(n, 5)
    return np.array([x, y, z])


randint = np.vectorize(randint)
randfloat = np.vectorize(randfloat)
randfloat_2d = np.vectorize(randfloat_2d, signature='()->(n)')
randfloat_3d = np.vectorize(randfloat_3d, signature='()->(n)')
vandercorput = np.vectorize(vandercorput)
halton_2d = np.vectorize(halton_2d, signature='()->(n)')
halton_3d = np.vectorize(halton_3d, signature='()->(n)')


""" VISUALIZATION """


def plot_randfloat():
    t = np.arange(100, dtype=np.uint32)
    x = randfloat(t)
    plt.plot(np.cos(2*pi*x), np.sin(2*pi*x), 'o')
    plt.show()


def plot_randfloat_2d():
    t = np.arange(1000, dtype=np.uint32)
    x = randfloat_2d(t)
    plt.plot(x[:, 0], x[:, 1], 'o')
    plt.show()


def plot_randfloat_3d():
    t = np.arange(1000, dtype=np.uint32)
    x = randfloat_3d(t)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    plt.show()


def plot_vandercorput():
    t = np.arange(100)
    x = vandercorput(t)
    plt.plot(np.cos(2*pi*x), np.sin(2*pi*x), 'o')
    plt.show()


def plot_halton_2d():
    t = np.arange(1000)
    x = halton_2d(t)
    plt.plot(x[:, 0], x[:, 1], 'o')
    plt.show()


def plot_halton_3d():
    t = np.arange(1000)
    x = halton_3d(t)
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    plt.show()


""" CONVERGENCE TEST """


def fun_1d_soft(x):
    return 0.5*pi*np.sin(pi*x)-1.0


def fun_1d_hard(x):
    return 1.4 if 0.3<x<0.6 else -0.6


def fun_2d_soft(x, y):
    return 36.0*x*y*(1-x)*(1-y)-1.0


def fun_2d_hard(x, y):
    return 4-pi if x*x+y*y<1.0 else -pi


def fun_3d_soft(x, y, z):
    return 216.0*x*y*z*(1-x)*(1-y)*(1-z)-1.0


def fun_3d_hard(x, y, z):
    return 6-pi if x*x+y*y+z*z<1.0 else -pi


# functions are defined in [0,1], integral equal to 0
fun_1d_soft = np.vectorize(fun_1d_soft)
fun_1d_hard = np.vectorize(fun_1d_hard)
fun_2d_soft = np.vectorize(fun_2d_soft)
fun_2d_hard = np.vectorize(fun_2d_hard)
fun_3d_soft = np.vectorize(fun_3d_soft)
fun_3d_hard = np.vectorize(fun_3d_hard)


BATCH_SIZE = 100000
BATCH_SIZE = 10000


def test_convergence_1d(fun, rand, message=""):
    n = np.arange(1, BATCH_SIZE+1)
    xs = rand(n)
    ys = fun(xs)
    avr = np.cumsum(ys)/n
    err = np.abs(avr)
    convergence = np.sum(np.log(n)*np.log(err))/np.sum(np.log(n)*np.log(n))
    print(message, convergence)


def test_convergence_2d(fun, rand, message=""):
    n = np.arange(1, BATCH_SIZE+1)
    xs = rand(n)
    ys = fun(xs[:, 0], xs[:, 1])
    avr = np.cumsum(ys)/n
    err = np.abs(avr)
    convergence = np.sum(np.log(n)*np.log(err))/np.sum(np.log(n)*np.log(n))
    print(message, convergence)


def test_convergence_3d(fun, rand, message=""):
    n = np.arange(1, BATCH_SIZE+1)
    xs = rand(n)
    ys = fun(xs[:, 0], xs[:, 1], xs[:, 2])
    avr = np.cumsum(ys)/n
    err = np.abs(avr)
    convergence = np.sum(np.log(n)*np.log(err))/np.sum(np.log(n)*np.log(n))
    print(message, convergence)


print("1d")
#test_convergence_1d(fun_1d_soft, randfloat, message="random,soft:")  # -1/2
#test_convergence_1d(fun_1d_hard, randfloat, message="random,hard:")  # -1/2
test_convergence_1d(fun_1d_soft, vandercorput, message="quasi,soft:")  # -1
test_convergence_1d(fun_1d_hard, vandercorput, message="quasi,hard:")  # -1

print("2d")
#test_convergence_2d(fun_2d_soft, randfloat_2d, message="random,soft:")  # -1/2
#test_convergence_2d(fun_2d_hard, randfloat_2d, message="random,hard:")  # -1/2
test_convergence_2d(fun_2d_soft, halton_2d, message="quasi,soft:")  # -1
test_convergence_2d(fun_2d_hard, halton_2d, message="quasi,hard:")  # -2/3

print("3d")
#test_convergence_3d(fun_3d_soft, randfloat_3d, message="random,soft:")  # -1/2
#test_convergence_3d(fun_3d_hard, randfloat_3d, message="random,hard:")  # -1/2
test_convergence_3d(fun_3d_soft, halton_3d, message="quasi,soft:")  # -1
test_convergence_3d(fun_3d_hard, halton_3d, message="quasi,hard:")  # -2/3
