# Fit the bunny using trigonometric basis functions with varying frequencies

import numpy as np
from optimizer import update_plot, optimize_adam, minimize_bfgs

import time


def load_data(path, size):
    """Load volume data, return coordinates and values"""
    # load voxels from raw file
    s = 128
    data = np.fromfile(path, dtype=np.uint8)
    assert len(data) == s ** 3
    data = data.reshape((s, s, s))

    # reduce size
    while s > size:
        s //= 2
        data = data.reshape((s, 2, s, 2, s, 2)).mean(axis=(1, 3, 5))

    # coordinates
    x = -1.0 + 2.0 * (np.arange(s) + 0.5) / s
    x = np.array(np.meshgrid(x, x, x))
    y = -np.sign(data - 0.5*255)
    x = x.reshape((3, s**3))
    y = y.flatten()

    return x, y


# Test model

N_WAVES = 32
N_WEIGHTS = N_WAVES*4


def model(w, x):
    y = np.zeros((x.shape[1]))
    y_grad = np.zeros((N_WEIGHTS, x.shape[1]))
    for wi in range(N_WAVES):
        ti = wi % 8
        trig = [np.cos, np.sin]
        d_trig = [lambda u: -np.sin(u), np.cos]
        amp, fx, fy, fz = w[wi], w[wi +
                                   N_WAVES], w[wi+2*N_WAVES], w[wi+3*N_WAVES]
        bx = trig[(ti >> 0) & 1](fx*x[0])
        by = trig[(ti >> 1) & 1](fy*x[1])
        bz = trig[(ti >> 2) & 1](fz*x[2])
        dbx = x[0]*d_trig[(ti >> 0) & 1](fx*x[0])
        dby = x[1]*d_trig[(ti >> 1) & 1](fy*x[1])
        dbz = x[2]*d_trig[(ti >> 2) & 1](fz*x[2])
        y += amp * bx*by*bz
        y_grad[wi] += bx*by*bz
        y_grad[wi+1*N_WAVES] += amp * dbx*by*bz
        y_grad[wi+2*N_WAVES] += amp * bx*dby*bz
        y_grad[wi+3*N_WAVES] += amp * bx*by*dbz
    return y, y_grad


# Loss functions


def loss_lnp(w, x, y):
    """Loss function is quadratic form-alike when w is away from the minimum
        Less likely to stuck in a local minimum"""
    f, f_grad = model(w, x)
    e2f = np.exp(2.0*f)
    e_2f = 1.0 / e2f
    v1 = (1.0-y) * np.log(1.0+e2f)
    v2 = (1.0+y) * np.log(1.0+e_2f)
    g1 = 2.0 * (1.0-y) * e2f / (1.0+e2f)
    g2 = -2.0 * (1.0+y) * e_2f / (1.0+e_2f)
    v = np.average((v1+v2)**2)
    g = np.matmul(f_grad, 2.0*(v1+v2)*(g1+g2)) / len(w)
    return v, g


def loss_mse(w, x, y):
    """Loss function is quadratic form-alike when w is close to the minimum"""
    f, f_grad = model(w, x)
    d = np.tanh(f) - y
    c2 = np.cosh(f) ** 2
    v = np.average(d**2)
    g = np.matmul(f_grad, 2.0 * d / c2) / len(f)
    return v, g


def evaluate_model(lossfun, w, x, y):
    """Print the loss and the magnitude of gradient of loss"""
    val, grad = lossfun(w, x, y)
    update_plot(w, val)
    print("Evaluate loss:", val, np.linalg.norm(grad))
    print(','.join(["{:.4f}".format(wi).rstrip('0') for wi in w]))


# Fitting

def fit():
    """Fit the dataset using polynomial regression model"""

    path = "../v_sdfbunny_128x128x128_uint8.raw"

    w = -1.0 + 2.0 * np.random.random((N_WEIGHTS))

    # gd = scipy.optimize.check_grad(lambda w_: model(w_, x)[0].sum(),
    #                                lambda w_: model(w_, x)[1].sum(axis=1), w)
    # print(gd)
    # return

    x, y = load_data(path, 16)

    evaluate_model(loss_mse, w, x, y)

    w = optimize_adam(loss_lnp, w, x, y, 1024, 0.01, 0.9, 0.999, 100, 1e-4)
    evaluate_model(loss_mse, w, x, y)

    w = minimize_bfgs(lambda w_: loss_lnp(w_, x, y),
                      w, maxiter=1000, gtol=1e-2)
    evaluate_model(loss_mse, w, x, y)

    x, y = load_data(path, 32)

    w = optimize_adam(loss_mse, w, x, y, 1024, 0.01, 0.9, 0.999, 40, 1e-4)
    evaluate_model(loss_mse, w, x, y)

    w = minimize_bfgs(lambda w_: loss_mse(w_, x, y),
                      w, maxiter=1000, gtol=1e-4)
    evaluate_model(loss_mse, w, x, y)

    x, y = load_data(path, 64)

    w = minimize_bfgs(lambda w_: loss_mse(w_, x, y),
                      w, maxiter=1000, gtol=1e-5)
    evaluate_model(loss_mse, w, x, y)


if __name__ == "__main__":

    np.random.seed(0)

    t0 = time.perf_counter()

    fit()

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
