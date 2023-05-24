# https://www.desmos.com/calculator/jveus4m8ji
# Desmos takes forever. Use SciPy instead.

import numpy as np
import scipy.optimize
import time


# training data
N = 20
X = (np.linspace(-N, N-1, 2*N) + 0.5) / N

Y = np.tanh(20.*(X*X-np.sin(2.*X)+0.2))
# Y = np.sign(X*X-np.sin(2.*X)+0.2)
# Y = np.sign(X*X-np.sin(5.*X))


# model


def B(x):
    """basis functions"""
    one = np.ones((len(x)))
    return np.array([x, np.cos(2.0*x), np.cos(4.0*x), np.cos(6.0*x)])
    # return np.array([one, np.sin(2.0*x), np.cos(2.0*x), np.sin(4.0*x)])
    # return np.array([one, x, x**2, x**3])
    # return np.array([one, np.exp(2.0*x), np.exp(3.0*x), np.exp(-2.0*x)])


B0 = B(X)


# Initial guess for optimization

def initial_guess():
    # return np.zeros(len(B0))
    left = np.matmul(B0, B0.T)
    right = np.matmul(Y, B0.T)
    return np.linalg.solve(left, right)


# Mean squared error


def L_mse(w):
    return np.average((np.tanh(np.matmul(w, B0)) - Y) ** 2)


def G_mse(w):
    f = np.matmul(w, B0)
    t = 2.0 * (np.tanh(f) - Y) / np.cosh(f)**2
    return np.matmul(B0, t) / len(f)


def optimize_L_mse():
    w = initial_guess()
    print(w, L_mse(w))
    r = scipy.optimize.minimize(L_mse, w, jac=G_mse,
                                method='CG', options={'gtol': 0.0})
    print(r)
    w = r.x
    print(w, L_mse(w), np.average(G_mse(w)**2)**0.5)


# Logarithmic error


def L_lnp(w):
    f = np.matmul(w, B0)
    v1 = (1.0-Y) * np.log(1.0 + np.exp(2.0*f))
    v2 = (1.0+Y) * np.log(1.0 + np.exp(-2.0*f))
    # return np.average(v1 + v2)
    return np.average((v1 + v2)**2)  # make it like a quadratic form


def G_lnp(w):
    f = np.matmul(w, B0)
    e2f = np.exp(2.0*f)
    e_2f = 1.0 / e2f
    v1 = (1.0-Y) * np.log(1.0+e2f)
    v2 = (1.0+Y) * np.log(1.0+e_2f)
    g1 = 2.0 * (1.0-Y) * e2f / (1.0+e2f)
    g2 = -2.0 * (1.0+Y) * e_2f / (1.0+e_2f)
    # return np.matmul(B0, g1+g2) / len(f)
    return np.matmul(B0, 2.0*(v1+v2)*(g1+g2)) / len(f)


def optimize_L_lnp():
    w = initial_guess()
    print(w, L_lnp(w))
    r = scipy.optimize.minimize(L_lnp, w, jac=G_lnp,
                                method='CG', options={'gtol': 0.0})
    print(r)
    w = r.x
    print(w, L_lnp(w), np.average(G_lnp(w)**2)**0.5)


# Overall optimization


def optimize_L():
    w = initial_guess()
    print(w, L_mse(w), np.average(G_mse(w)**2)**0.5)
    # optimize lnp - easier to optimize but potentially NAN
    r = scipy.optimize.minimize(L_lnp, w, jac=G_lnp,
                                method='CG', options={'gtol': 1e-4})
    print(r.nit, r.nfev, r.njev)
    w = r.x
    print(w, L_mse(w), np.average(G_mse(w)**2)**0.5)
    # optimize mse
    r = scipy.optimize.minimize(L_mse, w, jac=G_mse,
                                method='CG', options={'gtol': 0.0})
    print(r.nit, r.nfev, r.njev)
    w = r.x
    print(w, L_mse(w), np.average(G_mse(w)**2)**0.5)


t0 = time.perf_counter()
optimize_L()
t1 = time.perf_counter()
print("Time elapsed: {:.2f}secs".format(t1-t0))
