# Some code shared by multiple source files

import numpy as np
from scipy.optimize.minpack2 import dcsrch
import matplotlib.pyplot as plt
import time


# Plotting


plot_times = []
plot_losses = []
ax = None
plot_data = None


def update_plot(weights, loss):
    """This function is slow
        To speed up, close the matplotlib window"""
    global ax, plot_data
    plot_times.append(time.perf_counter())
    plot_losses.append(loss)

    open(".weights", 'w').write(str(weights.tolist()))

    is_init = ax is None
    if is_init:
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)

    plt.xlabel("Time (secs)")
    plt.ylabel("Loss")
    plt.yscale('log')
    plot_data = ax.plot(plot_times, plot_losses, 'r')

    plt.draw()
    plt.pause(0.01)


# SGD

def optimize_adam(lossfun, w, x, y, batch_size, learning_step, beta_1, beta_2, max_epoch, gtol):
    loss = 0.0
    grad = np.zeros((len(w)))
    grad2 = np.zeros((len(w)))
    train_order = np.arange(len(y))
    for epoch in range(max_epoch):
        np.random.shuffle(train_order)
        for batch in range(0, len(y), batch_size):
            indices = train_order[batch:batch+batch_size]
            loss_t, grad_t = lossfun(w, x[:, indices], y[indices])
            loss = beta_1 * loss + (1.0-beta_1) * loss_t
            grad = beta_1 * grad + (1.0-beta_1) * grad_t
            grad2 = beta_2 * grad2 + (1.0-beta_2) * grad_t*grad_t
            w -= learning_step * grad / (np.sqrt(grad2)+1e-8)
        grad_norm = np.linalg.norm(grad)
        if not np.isfinite(grad_norm):
            print("NAN encountered in optimize_adam")
            break
        print(f"Epoch {epoch}, loss={loss}, grad={grad_norm}")
        update_plot(w, loss)
        if grad_norm < gtol:
            break
    return w


# BFGS from SciPy


def scalar_search_wolfe1(fun, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=1e100, amin=1e-100, xtol=1e-14):
    if phi0 is None or derphi0 is None:
        phi0, derphi0 = fun(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    maxiter = 100
    for i in range(maxiter):
        stp, phi1, derphi1, task = dcsrch(alpha1, phi1, derphi1,
                                          c1, c2, xtol, task,
                                          amin, amax, isave, dsave)
        if task[:2] == b'FG':
            alpha1 = stp
            phi1, derphi1 = fun(stp)
        else:
            break
    else:
        stp = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        stp = None  # failed

    return stp, phi1, phi0


def line_search_wolfe1(fun, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       c1=1e-4, c2=0.9,
                       xtol=1e-14):
    if gfk is None:
        gfk = fun(xk)[1]

    gradient = True

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        val, gval[0] = fun(xk + s*pk)
        if gradient:
            gc[0] += 1
        else:
            fc[0] += len(xk) + 1
        return val, np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
        phi, old_fval, old_old_fval, derphi0,
        c1=c1, c2=c2, xtol=xtol)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]


def minimize_bfgs(fun, x0, gtol=1e-5, maxiter=None):
    x0 = np.array(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    old_fval, gfk = fun(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    warnflag = 0
    gnorm = np.linalg.norm(gfk)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
            line_search_wolfe1(fun, xk, pk, gfk,
                               old_fval, old_old_fval)
        if alpha_k is None:  # line search error
            warnflag = 2
            break
        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = fun(xkp1)[1]

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        gnorm = np.linalg.norm(gfk)
        if (gnorm <= gtol):
            break

        if True:
            update_plot(xk, old_fval)
            print(k, old_fval, gnorm)

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                           sk[np.newaxis, :])

    fval = old_fval

    if warnflag == 2:
        print("Precision loss")
    elif k >= maxiter:
        print("Maximum number of iterations exceeded")
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        print("NAN encountered")
    else:
        print("BFGS optimization success")

    return xk
