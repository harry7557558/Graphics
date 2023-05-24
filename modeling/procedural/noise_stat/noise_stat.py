# Mathematically analyze the distribution of noise outputs

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.integrate

__import__('os').environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
__import__('warnings').filterwarnings("ignore")
import tensorflow_probability as tfp  # why

from random import random
from math import *

# Note that polytrigint for pure polynomial functions
# is 30 times slower than polyint
import polyint
import polytrigint
import sympy

from time import perf_counter


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
    x = tfp.mcmc.sample_halton_sequence(dim, num_results=N, seed=1)
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
ss3 = lambda x: x*x*(3-2*x)
ss5 = lambda x: x*x*x*(10+x*(-15+x*6))

invss3 = lambda x: 0.5 - np.sin(np.arcsin(1.0-2.0*x)/3.0)
invss5 = np.vectorize(
    lambda x: scipy.optimize.brentq(lambda t: ss5(t)-x, 0.0, 1.0))

ss5_grad = lambda x: ((x*30-60)*x+30)*x*x


# reparameterization functions


def uniform_unit_circle(u, v):
    # for u,v in [0,1), return (x,y)
    u = np.sqrt(u)
    v = 2*pi*v
    return (u*np.cos(v), u*np.sin(v))


def uniform_unit_circle_int(u, v):
    # for analytical integration, return (x, y, ∂(x,y)/∂(u,v))
    # v in [0, 2*PI)
    return (u*polytrigint.cos(v), u*polytrigint.sin(v), u)


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
        var = 17/140
        sigma = sqrt(17/140)
        mad = 5/16
    """
    x = halton_sequence(1, N)[0]
    return ss3(x)


def rand_ss5(N):
    """
        pdf(x) = 1/(30*invss5(x)**2*(1+invss5(x)*(invss5(x)-2))), 0, 1
        mu = 1/2
        var = 131/924
        sigma = sqrt(131/924)
        mad = 11/32
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
        var = 2/9
        sigma = sqrt(2)/3
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
        var = 26/105
        sigma = sqrt(26/105)
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
        var = 181/693
        sigma = sqrt(181/693)
        mad = 0.434771
    """
    halton = halton_sequence(3, N)
    y0 = 2.0*halton[0]-1.0
    y1 = 2.0*halton[1]-1.0
    t = halton[2]
    return y0 + (y1-y0) * ss5(t)


def cosinenoise2d(N):
    """
        pdf(x) = ??
        mu = 0
        var = 1/4
        sigma = 1/2
        mad = 0.4052847 (4/pi**2 ?)
        max = 1
    """
    halton = halton_sequence(2, N)
    x, y = halton
    return np.cos(pi*x)*np.cos(pi*y)


def cosinenoise2d_grad(N):
    """
        pdf(x) = ??
        mu = 0
        var = 1/4
        sigma = 1/2
        mad = 0.4052847 (4/pi**2 ?)
        max = 1
    """
    halton = halton_sequence(2, N)
    x, y = halton
    return np.array([
        -pi*np.sin(pi*x)*np.cos(pi*y),
        -pi*np.cos(pi*x)*np.sin(pi*y)
        ])


def valuenoise2d_1(N):
    """
        pdf(x) = ??
        mu = 0
        var = 4/27
        sigma = 2/sqrt(27)
        max = 1
    """
    halton = halton_sequence(6, N)
    z00 = 2*halton[0]-1
    z01 = 2*halton[1]-1
    z10 = 2*halton[2]-1
    z11 = 2*halton[3]-1
    x = ss1(halton[4])
    y = ss1(halton[5])
    return mix(mix(z00, z01, x), mix(z10, z11, x), y)


def valuenoise2d_5(N):
    """
        pdf(x) =
        mu = 0
        var = 32761/160083
        sigma = sqrt(32761/160083)
        max = 1
    """
    halton = halton_sequence(6, N)
    z00 = 2*halton[0]-1
    z01 = 2*halton[1]-1
    z10 = 2*halton[2]-1
    z11 = 2*halton[3]-1
    x = ss5(halton[4])
    y = ss5(halton[5])
    return mix(mix(z00, z01, x), mix(z10, z11, x), y)


def valuenoise2d_5_grad(N):
    """
        mu = [0, 0]
        cov = diag(3620/4851)
        max >= 3.7311058
    """
    halton = halton_sequence(6, N)
    z00 = 2*halton[0]-1
    z01 = 2*halton[1]-1
    z10 = 2*halton[2]-1
    z11 = 2*halton[3]-1
    intpx = ss5(halton[4])
    intpy = ss5(halton[5])
    intpdx = ss5_grad(halton[4])
    intpdy = ss5_grad(halton[5])
    return np.array([
        mix(z10-z00, z11-z01, intpy)*intpdx,
        mix(z01-z00, z11-z10, intpx)*intpdy
        ])


def gradientnoise2d_5(N):
    """
        mu = 0
        var = 193670/6243237
        max >= 0.716239
    """
    halton = halton_sequence(10, N)
    g00x, g00y = 2*halton[0]-1, 2*halton[1]-1
    g01x, g01y = 2*halton[2]-1, 2*halton[3]-1
    g10x, g10y = 2*halton[4]-1, 2*halton[5]-1
    g11x, g11y = 2*halton[6]-1, 2*halton[7]-1
    x, y = halton[8:10]
    v00 = g00x*(x-0) + g00y*(y-0)
    v01 = g01x*(x-0) + g01y*(y-1)
    v10 = g10x*(x-1) + g10y*(y-0)
    v11 = g11x*(x-1) + g11y*(y-1)
    xf, yf = ss5(x), ss5(y)
    return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf)


def gradientnoise2d_5_grad(N):
    """
        mu = [0, 0]
        var = diag(77320/297297)
        max >= 2.2122107
    """
    halton = halton_sequence(10, N)
    g00x, g00y = 2*halton[0]-1, 2*halton[1]-1
    g01x, g01y = 2*halton[2]-1, 2*halton[3]-1
    g10x, g10y = 2*halton[4]-1, 2*halton[5]-1
    g11x, g11y = 2*halton[6]-1, 2*halton[7]-1
    x, y = halton[8:10]
    v00 = g00x*(x-0) + g00y*(y-0)
    v01 = g01x*(x-0) + g01y*(y-1)
    v10 = g10x*(x-1) + g10y*(y-0)
    v11 = g11x*(x-1) + g11y*(y-1)
    intpx, intpy = ss5(x), ss5(y)
    intpdx, intpdy = ss5_grad(x), ss5_grad(y)
    dfdx = g00x+(g10x-g00x)*intpx+(g01x-g00x)*intpy+(g00x+g11x-g01x-g10x)*intpx*intpy \
           + ((v10-v00)+(v00+v11-v01-v10)*intpy)*intpdx
    dfdy = g00y+(g10y-g00y)*intpx+(g01y-g00y)*intpy+(g00y+g11y-g01y-g10y)*intpx*intpy \
           + ((v01-v00)+(v00+v11-v01-v10)*intpx)*intpdy
    return np.array([dfdx, dfdy])


def normalizedgradientnoise2d_5(N):
    """
        mu = 0
        var = 96835/4162158
        max >= 0.5958338
    """
    halton = halton_sequence(10, N)
    g00x, g00y = uniform_unit_circle(halton[0], halton[1])
    g01x, g01y = uniform_unit_circle(halton[2], halton[3])
    g10x, g10y = uniform_unit_circle(halton[4], halton[5])
    g11x, g11y = uniform_unit_circle(halton[6], halton[7])
    x, y = halton[8:10]
    v00 = g00x*(x-0) + g00y*(y-0)
    v01 = g01x*(x-0) + g01y*(y-1)
    v10 = g10x*(x-1) + g10y*(y-0)
    v11 = g11x*(x-1) + g11y*(y-1)
    xf, yf = ss5(x), ss5(y)
    return v00
    return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf)


def normalizedgradientnoise2d_5_grad(N):
    """
        mu = [0, 0]
        var = diag(19330/99099)
        max >= 1.7525575
    """
    halton = halton_sequence(10, N)
    g00x, g00y = uniform_unit_circle(halton[0], halton[1])
    g01x, g01y = uniform_unit_circle(halton[2], halton[3])
    g10x, g10y = uniform_unit_circle(halton[4], halton[5])
    g11x, g11y = uniform_unit_circle(halton[6], halton[7])
    x, y = halton[8:10]
    v00 = g00x*(x-0) + g00y*(y-0)
    v01 = g01x*(x-0) + g01y*(y-1)
    v10 = g10x*(x-1) + g10y*(y-0)
    v11 = g11x*(x-1) + g11y*(y-1)
    intpx, intpy = ss5(x), ss5(y)
    intpdx, intpdy = ss5_grad(x), ss5_grad(y)
    dfdx = g00x+(g10x-g00x)*intpx+(g01x-g00x)*intpy+(g00x+g11x-g01x-g10x)*intpx*intpy \
           + ((v10-v00)+(v00+v11-v01-v10)*intpy)*intpdx
    dfdy = g00y+(g10y-g00y)*intpx+(g01y-g00y)*intpy+(g00y+g11y-g01y-g10y)*intpx*intpy \
           + ((v01-v00)+(v00+v11-v01-v10)*intpx)*intpdy
    return np.array([dfdx, dfdy])


# Testing


def randtest(func):

    print("Monte Carlo")

    N = 2 << 18
    x = func(N)

    mu = np.average(x)
    var = np.sum((x-mu)**2)/(N-1)
    sigma = sqrt(var)
    mad = np.average(abs(x-mu))
    print('mu =', mu)
    print('var =', var)
    print('sigma =', sigma)
    print('mad =', mad)
    print('max =', max(np.amax(x), -np.amin(x)))
    print()

    bins = np.linspace(-2, 2, 100)
    hist, bins = np.histogram(x, bins=bins, density=True)
    bin_centers = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bin_centers, hist)
    plt.show()


def randtest_grad(func):

    print("Monte Carlo")

    N = 2 << 19
    x = func(N)

    mu = np.average(x, axis=1)
    var = np.cov(x)
    print('mu =', mu.tolist())
    print('var =', var.tolist(), pow(np.linalg.det(var), 1.0/var.shape[0]))
    print('max =', max(np.amax(x), -np.amin(x)))
    print()


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


def calc_variance():
    
    def uniform_unit_circle(u, v):
        u = sympy.sqrt(u)
        v = 2*sympy.pi*v
        return (u*sympy.cos(v), u*sympy.sin(v))

    print("Analytical Integration")

    NUMCOMP = 10
    varnames = [f's{i}' for i in range(NUMCOMP)]
    #halton = [polyint.Polynomial(s) for s in varnames]
    halton = [sympy.Symbol(s) for s in varnames]
    diffs = [(varnames[i], 0, 1) for i in range(NUMCOMP)]

    # copy-paste code here
    g00x, g00y = uniform_unit_circle(halton[0], halton[1])
    g01x, g01y = uniform_unit_circle(halton[2], halton[3])
    g10x, g10y = uniform_unit_circle(halton[4], halton[5])
    g11x, g11y = uniform_unit_circle(halton[6], halton[7])
    x, y = halton[8:10]
    v00 = g00x*(x-0) + g00y*(y-0)
    v01 = g01x*(x-0) + g01y*(y-1)
    v10 = g10x*(x-1) + g10y*(y-0)
    v11 = g11x*(x-1) + g11y*(y-1)
    intpx, intpy = ss5(x), ss5(y)
    intpdx, intpdy = ss5_grad(x), ss5_grad(y)
    dfdx = g00x+(g10x-g00x)*intpx+(g01x-g00x)*intpy+(g00x+g11x-g01x-g10x)*intpx*intpy \
           + ((v10-v00)+(v00+v11-v01-v10)*intpy)*intpdx
    dfdy = g00y+(g10y-g00y)*intpx+(g01y-g00y)*intpy+(g00y+g11y-g01y-g10y)*intpx*intpy \
           + ((v01-v00)+(v00+v11-v01-v10)*intpx)*intpdy
    f = np.array([dfdx, dfdy])

    fun = f[0]**2
    print(fun)
    #ans = polyint.integrate(fun, diffs)
    ans = sympy.integrate(fun, *diffs)
    print(ans)
    print(float(ans))


if __name__=="__main__":

    t0 = perf_counter()

    #randtest(normalizedgradientnoise2d_5)
    #randtest_grad(normalizedgradientnoise2d_5_grad)

    calc_variance()

    t1 = perf_counter()
    print("Time elapsed:", t1-t0, "secs")
    
