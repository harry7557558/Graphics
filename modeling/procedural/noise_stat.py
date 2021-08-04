# Mathematically analyze the distribution of noise outputs
# Not a successful experiment :(

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate

__import__('os').environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
import tensorflow_probability as tfp  # why

from random import random
from math import *


def mix(a, b, x):
    return a + (b-a) * x


def vandercorput_base(n, b):
    x = 0.0
    e = 1.0 / b
    while n != 0:
        d = n % b
        x += d * e
        e /= b
        n //= b
    return x


def halton_sequence(dim, N):
    # fast but sometimes cause crash
    x = tfp.mcmc.sample_halton_sequence(dim, num_results=N)
    return np.transpose(x.numpy())

    # slow
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]
    result = np.zeros((dim, N))
    for bi in range(dim):
        b = PRIMES[bi]
        for k in range(0, N):
            result[bi][k] = vandercorput_base(k+1, b)
    return result


# smoothstep functions and inverse

ss0 = lambda x: 0.5
ss1 = lambda x: x
ss3 = lambda x: x*x*(3.0-2.0*x)
ss5 = lambda x: x*x*x*(10.0+x*(-15.0+x*6.0))

invss3 = lambda x: 0.5 - np.sin(np.arcsin(1.0-2.0*x)/3.0)
invss5 = np.vectorize(
    lambda x: scipy.optimize.brentq(lambda t: ss5(t)-x, 0.0, 1.0))


# random/noise functions


def rand_cos(N):
    """
        pdf(x) = 1/(pi*sqrt(1-x*x)), -1, 1
        mu = 0
        var = 1/2
        sigma = sqrt(2)/2
        mad = 2/pi
    """
    x = halton_sequence(1, N)[0]
    return np.cos(pi*x)


def rand_ss3(N):
    """
        pdf(x) = 1/(6*invss3(x)*(1-invss3(x))), 0, 1
        mu = 1/2
        var = 0.121428571 (17/140 ?)
        sigma = 0.348466026 (sqrt(17/35)/2 ?)
        mad = 0.3125 (5/16 ?)
    """
    x = halton_sequence(1, N)[0]
    return ss3(x)


def rand_ss5(N):
    """
        pdf(x) = 1/(30*invss5(x)**2*(1+invss5(x)*(invss5(x)-2))), 0, 1
        mu = 1/2
        var = 0.1417749
        sigma = 0.3765301
        mad = 0.34375 (11/32 ?)
    """
    x = halton_sequence(1, N)[0]
    return ss5(x)


def valuenoise1d_0(N):
    """
        pdf(x) = 1-abs(x), -1, 1
        mu = 0
        var = 1/6
        sigma = 1/sqrt(6)
        mad = 1/3
    """
    halton = halton_sequence(3, N)
    y0 = 2.0*halton[0]-1.0
    y1 = 2.0*halton[1]-1.0
    t = halton[2]
    return y0 + (y1-y0) * ss0(t)


def valuenoise1d_1(N):
    """
        pdf(x) = ??
        mu = 0
        var = 0.222222
        sigma = 0.471404
        mad = 0.397715
    """
    halton = halton_sequence(3, N)
    y0 = 2.0*halton[0]-1.0
    y1 = 2.0*halton[1]-1.0
    t = halton[2]
    return y0 + (y1-y0) * ss1(t)


def valuenoise1d_3(N):
    """
        pdf(x) = ??
        mu = 0
        var = 0.247619
        sigma = 0.497613
        mad = 0.422133
    """
    halton = halton_sequence(3, N)
    y0 = 2.0*halton[0]-1.0
    y1 = 2.0*halton[1]-1.0
    t = halton[2]
    return y0 + (y1-y0) * ss3(t)


def valuenoise1d_5(N):
    """
        pdf(x) = ??
        mu = 0
        var = 0.261183
        sigma = 0.511061
        mad = 0.434771
    """
    halton = halton_sequence(3, N)
    y0 = 2.0*halton[0]-1.0
    y1 = 2.0*halton[1]-1.0
    t = halton[2]
    return y0 + (y1-y0) * ss5(t)


def valuenoise2d_1(N):
    """
        pdf(x) = ??
        mu = 0
        var = 0.14815
        sigma = 0.38490
        mad = 0.31802
    """
    halton = halton_sequence(6, N)
    z00 = 2.0*halton[0]-1.0
    z01 = 2.0*halton[1]-1.0
    z10 = 2.0*halton[2]-1.0
    z11 = 2.0*halton[3]-1.0
    x = ss1(halton[4])
    y = ss1(halton[5])
    return mix(mix(z00, z01, x), mix(z10, z11, x), y)


def valuenoise2d_5(N):
    """
        pdf(x) =
        mu =
        var =
        sigma =
        mad =
    """
    z00 = 2.0*np.random.random(N)-1.0
    z01 = 2.0*np.random.random(N)-1.0
    z10 = 2.0*np.random.random(N)-1.0
    z11 = 2.0*np.random.random(N)-1.0
    x = ss5(np.random.random(N))
    y = ss5(np.random.random(N))
    return mix(mix(z00, z01, x), mix(z10, z11, x), y)


def randtest(func):

    N = 2 << 20
    x = func(N)

    mu = np.average(x)
    var = np.sum((x-mu)**2)/(N-1)
    sigma = sqrt(var)
    mad = np.average(abs(x-mu))
    print("Monte Carlo")
    print('mu =', mu)
    print('var =', var)
    print('sigma =', sigma)
    print('mad =', mad)
    print()

    bins = np.linspace(-2, 2, 100)
    hist, bins = np.histogram(x, bins=bins, density=True)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bin_centers, hist)
    plt.show()


def stattest(pdf, x0, x1):
    # After getting these numerical approximations,
    # one can search for closed forms using WolframAlpha
    print("Integration")
    pint = scipy.integrate.quad(pdf, x0, x1)  # should be 1
    mu = scipy.integrate.quad(lambda x: x*pdf(x), x0, x1)
    print(f'int = {round(pint[0], int(-log10(pint[1])))}', pint)
    print(f'mu = {round(mu[0], int(-log10(mu[1])))}', mu)
    var = scipy.integrate.quad(lambda x: (x-mu[0])**2 * pdf(x), x0, x1)
    print(f'var = {round(var[0], int(-log10(var[1])))}', var)
    print(f'sigma = {round(sqrt(var[0]), int(-log10(var[1])))} ({sqrt(var[0])})')
    mad = scipy.integrate.quad(lambda x: abs(x-mu[0]) * pdf(x), x0, x1)
    print(f'mad = {round(mad[0], int(-log10(mad[1])))}', mad)
    print()



if __name__=="__main__":

    #stattest(lambda x: 1-abs(x), -1, 1)
    randtest(valuenoise2d_5)
