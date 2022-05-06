# Some code shared by multiple source files

import numpy as np
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


def dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
    """https://github.com/scipy/scipy/blob/main/scipy/optimize/minpack2/dcstep.f"""

    sgnd = dp * (dx / abs(dx))
    stpf = stp

    if fp > fx:
        theta = 3.0 * (fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp < stx:
            gamma = -gamma
        p = (gamma-dx) + theta
        q = ((gamma-dx)+gamma) + dp
        r = p / q
        stpc = stx + r * (stp-stx)
        stpq = stx + ((dx/((fx-fp)/(stp-stx)+dx))/2.0)*(stp-stx)
        if abs(stpc-stx) < abs(stpq-stx):
            stpf = stpc
        else:
            stpf = stpc + (stpq-stpc)/2.0
        brackt = True

    elif sgnd < 0.0:
        theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))
        gamma = s * np.sqrt((theta/s)**2 - (dx/s)*(dp/s))
        if stp > stx:
            gamma = -gamma
        p = (gamma-dp) + theta
        q = ((gamma-dp)+gamma) + dx
        r = p / q
        stpc = stp + r * (stx-stp)
        stpq = stp + (dp/(dp-dx))*(stx-stp)
        if abs(stpc-stp) > abs(stpq-stp):
            stpf = stpc
        else:
            stpf = stpq
        brackt = True

    elif abs(dp) < abs(dx):
        theta = 3.0*(fx-fp)/(stp-stx) + dx + dp
        s = max(abs(theta), abs(dx), abs(dp))

        gamma = s * np.sqrt(max(0.0, (theta/s)**2-(dx/s)*(dp/s)))
        if stp > stx:
            gamma = -gamma
        p = (gamma-dp) + theta
        q = (gamma+(dx-dp)) + gamma
        r = p / q
        if r < 0.0 and gamma != 0.0:
            stpc = stp + r * (stx-stp)
        elif stp > stx:
            stpc = stpmax
        else:
            stpc = stpmin
        stpq = stp + (dp/(dp-dx))*(stx-stp)

        if brackt:
            if abs(stpc-stp) < abs(stpq-stp):
                stpf = stpc
            else:
                stpf = stpq
            if stp > stx:
                stpf = min(stp+0.66*(sty-stp), stpf)
            else:
                stpf = max(stp+0.66*(sty-stp), stpf)
        else:
            if abs(stpc-stp) > abs(stpq-stp):
                stpf = stpc
            else:
                stpf = stpq
            stpf = max(stpmin, min(stpmax, stpf))

    else:
        if brackt:
            theta = 3.0*(fp-fy)/(sty-stp) + dy + dp
            s = max(abs(theta), abs(dy), abs(dp))
            gamma = s * np.sqrt((theta/s)**2-(dy/s)*(dp/s))
            if stp > sty:
                gamma = -gamma
            p = (gamma-dp)+theta
            q = ((gamma-dp)+gamma)+dy
            r = p / q
            stpc = stp+r*(sty-stp)
            stpf = stpc
        elif stp > stx:
            stpf = stpmax
        else:
            stpf = stpmin

    if fp > fx:
        sty = stp
        fy = fp
        dy = dp
    else:
        if sgnd < 0.0:
            sty = stx
            fy = fx
            dy = dx
        stx = stp
        fx = fp
        dx = dp
    stp = stpf

    return stx, fx, dx, sty, fy, dy, stp, brackt


def scalar_search_wolfe1(fun, f0, old_f0, g0,
                         ftol=1e-4, gtol=0.9,
                         stpmax=1e100, stpmin=1e-100, xtol=1e-14):
    """https://github.com/scipy/scipy/blob/main/scipy/optimize/minpack2/dcsrch.f"""

    if g0 != 0.0:
        stp = min(1.0, 1.01*2*(f0 - old_f0)/g0)
        if stp <= 0.0:
            stp = 1.0
    else:
        stp = 1.0

    f, g = f0, g0

    if stp < stpmin or stp > stpmax or g >= 0.0:
        # bad start
        stp = -1.0
        return stp, f, f0

    brackt = False
    stage = 1
    gtest = ftol * g0
    width = stpmax - stpmin
    width1 = 2.0 * width

    xtrapl, xtrapu = 1.1, 4.0
    stx, sty = 0.0, 0.0
    fx, gx = f0, g0
    fy, gy = f0, g0
    stmin = 0.0
    stmax = stp + xtrapu * stp

    f, g = fun(stp)

    maxiter = 100
    for i in range(maxiter):

        ftest = f0 + stp * gtest
        if stage == 1 and f <= ftest and g >= 0.0:
            stage = 2

        if (brackt and (stp <= stmin or stp >= stmax)) or \
                (brackt and stmax-stmin <= xtol*stmax) or \
                (stp == stpmax and f <= ftest and g <= gtest) or \
                (stp == stpmin and (f > ftest or g >= gtest)):
            # warning
            stp = -1.0
            return stp, f, f0

        if f <= ftest and abs(g) <= gtol * (-g0):
            # success
            return stp, f, f0

        if stage == 1 and f <= fx and f >= ftest:

            fm = f - stp * gtest
            fxm = fx - stx * gtest
            fym = fy - sty * gtest
            gm = g - gtest
            gxm = gx - gtest
            gym = gy - gtest

            stx, fxm, gxm, sty, fym, gym, stp, brackt = dcstep(
                stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax)

            fx = fxm + stx * gtest
            fy = fym + sty * gtest
            gx = gxm + gtest
            gy = gym + gtest

        else:
            stx, fx, gx, sty, fy, gy, stp, brackt = dcstep(
                stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax)
        # print(stx, sty)
        # __import__('sys').exit(0)

        if brackt:
            if abs(sty-stx) >= 0.66*width1:
                stp = stx + 0.5 * (sty-stx)
            width1 = width
            width = abs(sty-stx)

        if brackt:
            stmin = min(stx, sty)
            stmax = max(stx, sty)
        else:
            stmin = stp + xtrapl * (stp-stx)
            stmax = stp + xtrapu * (stp-stx)

        stp = min(max(stp, stpmin), stpmax)

        if brackt and (stp <= stmin or stp >= stmax) or (brackt and stmax-stmin <= xtol*stmax):
            stp = stx

        f, g = fun(stp)

    # maxiter exceeded
    stp = -1.0
    return stp, f, f0


def line_search_wolfe1(fun, xk, pk, gfk,
                       old_fval=None, old_old_fval=None):
    gval = gfk

    def phi(s):
        nonlocal gval
        # print("s/xk/pk {:.16f} {:.16f} {:.16f}".format(s, np.linalg.norm(xk), np.linalg.norm(pk)))
        x_param = xk + s * pk
        val, gval = fun(x_param)
        # print("x_param/val/gval {:.16f} {:.16f} {:.16f}".format(np.linalg.norm(x_param), val, np.linalg.norm(gval)))
        return val, np.dot(gval, pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
        phi, old_fval, old_old_fval, derphi0)
    return stp, fval, old_fval, gval


def minimize_bfgs(fun, x0, gtol=1e-5, maxiter=None):
    x0 = np.array(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    old_fval, gfk = fun(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=np.float64)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    warnflag = 0
    gnorm = np.linalg.norm(gfk)
    while (gnorm > gtol) and (k < maxiter):
        # print("Hk", np.linalg.norm(Hk))

        pk = -np.dot(Hk, gfk)
        alpha_k, old_fval, old_old_fval, gfkp1 = \
            line_search_wolfe1(fun, xk, pk, gfk,
                               old_fval, old_old_fval)
        if alpha_k < 0.0:
            # line search error
            warnflag = 2
            break
        # print("alpha/xk/pk {:.16f} {:.16f} {:.16f}".format(alpha_k, np.linalg.norm(xk), np.linalg.norm(pk)))
        # print("ofval/oofval/gfkp1 {:.16f} {:.16f} {:.16f}".format(old_fval, old_old_fval, np.linalg.norm(gfkp1)))

        xkp1 = xk + alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        yk = gfkp1 - gfk
        gfk = gfkp1
        k += 1
        gnorm = np.linalg.norm(gfk)
        if (gnorm <= gtol):
            break

        if True:
            update_plot(xk, old_fval)
            print("BFGS {:d} {:.16f} {:.16f}".format(k, old_fval, gnorm))
            # __import__('sys').exit(0)

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        if 0:
            # SciPy implementation
            rhok_inv = np.dot(yk, sk)
            if rhok_inv == 0.:
                rhok = 1000.0
            else:
                rhok = 1. / rhok_inv
            A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
            A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
            Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                               sk[np.newaxis, :])
        else:
            # Wikipedia formula
            skyk = np.dot(sk, yk)
            if skyk == 0.0:
                skyk = 1e-3
            w1 = (np.dot(sk, yk) + np.dot(yk, np.dot(Hk, yk))) / skyk**2
            m1 = np.tensordot(sk, sk, axes=0)
            w2 = np.dot(sk, yk)
            w2 = -1e3 if w2 == 0.0 else -1.0 / w2
            hkyk = np.dot(Hk, yk)
            m2 = np.tensordot(hkyk, sk, axes=0) + \
                np.tensordot(sk, hkyk, axes=0)
            Hk += m1 * w1 + m2 * w2

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


if __name__ == "__main__":

    from scipy.optimize import rosen, rosen_der, check_grad

    nfev = 0

    def rosenbrock(x):
        global nfev
        nfev += 1
        # return rosen(x), rosen_der(x)
        # return np.dot(x, x), 2.0 * x
        val = 0.0
        grad = np.zeros(len(x))
        for i in range(0, len(x), 2):
            val += (1.0-x[i])**2 + 100.0*(x[i+1] - x[i]*x[i])**2
            grad[i] = -2.0*(1.0-x[i]) - 400.0*x[i]*(x[i+1] - x[i]*x[i])
            grad[i+1] = 200.0 * (x[i+1] - x[i]*x[i])
        return val, grad

    np.random.seed(0)
    x0 = np.random.random(20)
    x0 = -np.ones(128)

    # print("checkGrad",
    #       check_grad(lambda x_: rosenbrock(x_)[0], lambda x_: rosenbrock(x_)[1], x0))

    print(rosenbrock(x0)[0])

    x1 = minimize_bfgs(rosenbrock, x0, gtol=1e-5, maxiter=1000)
    print(rosenbrock(x1)[0])
    print("nfev:", nfev)
