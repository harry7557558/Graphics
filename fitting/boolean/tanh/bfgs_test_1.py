# Test BFGS optimizer techniques in fitting the bunny

import numpy as np
import scipy.optimize

import matplotlib.pyplot as plt

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

    # plot
    if 0:
        import matplotlib.pyplot as plt
        ax = plt.figure().add_subplot(projection='3d', proj_type='ortho')
        indices = np.where(y < 0.0)[0]
        xs = x[:, indices]
        ax.scatter(xs[0], xs[1], xs[2])
        plt.show()

    return x, y


# Test model

MODEL_DEG = 2  # parameter for the model
N_WEIGHTS = (2*MODEL_DEG+1)**3  # number of weights


def model(w, x):
    """Input w and x, evaluate y and dy/dw
        The model has (2*deg+1)**3 weights
        Hessian is zero"""
    y = np.zeros((x.shape[1]))
    y_grad = np.zeros((N_WEIGHTS, x.shape[1]))
    yi = 0
    for i in range(MODEL_DEG+1):
        i *= 2
        cossinx = [np.cos(i*x[0]), np.sin(i*x[0])]
        for j in range(MODEL_DEG+1):
            j *= 2
            cossiny = [np.cos(j*x[1]), np.sin(j*x[1])]
            for k in range(MODEL_DEG+1):
                k *= 2
                cossinz = [np.cos(k*x[2]), np.sin(k*x[2])]
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if (i == 0 and di == 1) or (j == 0 and dj == 1) or (k == 0 and dk == 1):
                                continue
                            b = cossinx[di] * cossiny[dj] * cossinz[dk]
                            y += w[yi] * b
                            y_grad[yi] = b
                            yi += 1
    return y, y_grad


# Loss functions

def rosenbrock(w, x, y, hess: bool = False):
    """So trivial to optimize"""
    v = scipy.optimize.rosen(w)
    grad = scipy.optimize.rosen_der(w)
    if hess:
        hess = scipy.optimize.rosen_hess(w)
        return v, grad, hess
    return v, grad


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
    update_plot(val)
    print("Evaluate loss:", val, np.linalg.norm(grad))


# Plotting


plot_times = []
plot_losses = []
ax = None
plot_data = None


def update_plot(loss):
    global ax, plot_data
    plot_times.append(time.perf_counter())
    plot_losses.append(loss)
    return  # slow

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
    grad = np.zeros((N_WEIGHTS))
    grad2 = np.zeros((N_WEIGHTS))
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
        update_plot(loss)
        if grad_norm < gtol:
            break
    return w


# BFGS from SciPy

from scipy.optimize.minpack2 import dcsrch

def scalar_search_wolfe1(fun, phi0=None, old_phi0=None, derphi0=None,
                         c1=1e-4, c2=0.9,
                         amax=1e100, amin=1e-100, xtol=1e-14):
    if phi0 is None:
        phi0 = fun(0.)[0]
    if derphi0 is None:
        derphi0 = fun(0.)[1]

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
        # maxiter reached, the line search did not converge
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

    f = lambda x_: fun(x_)[0]
    myfprime = lambda x_: fun(x_)[1]

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
            update_plot(old_fval)
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



def fit(loss_fun):
    """Fit the dataset using polynomial regression model"""

    path = "../v_sdfbunny_128x128x128_uint8.raw"
    x, y = load_data(path, 64)

    w = -1.0 + 2.0 * np.random.random((N_WEIGHTS))
    # w = np.array([1.04,0.04,0.007,0.04,0.078,-0.219,0.061,-0.074,-0.024,0.019,0.004,-0.24,-0.098,0.068,0.026,0.119,0.109,0.054,0.015,0.09,0.009,0.128,0.124,0.113,0.029,-0.306,0.068,0.088,-0.014,-0.035,0.001,-0.359,-0.28,0.082,0.047,-0.483,0.107,-0.2,0.041,0.327,0.016,-0.102,0.,0.133,-0.001,-0.027,-0.,-0.591,-0.046,0.135,0.053,-0.244,-0.093,0.05,0.014,-0.23,0.263,0.125,0.094,0.015,-0.021,-0.24,0.002,-0.073,0.002,-0.047,0.001,-0.264,-0.297,0.329,0.085,0.151,0.074,0.112,0.04,0.05,0.031,0.068,0.012,0.062,0.011,0.049,0.119,0.027,0.067,-0.03,0.03,-0.016,0.045,-0.25,-0.032,0.055,0.005,-0.242,-0.031,0.015,0.003,-0.002,-0.051,0.027,0.012,0.013,-0.087,0.05,0.008,0.093,0.027,0.061,0.099,0.124,0.022,0.184,0.012,0.102,0.019,0.081,0.012,0.09,0.161,0.008,0.002,0.056,0.099,0.102,0.03])
    w = np.array([103.0367,-25.3708,-15.8335,11.6108,7.8243,-17.9513,38.3982,-52.7572,11.1286,23.1443,16.0248,-27.4161,-16.8879,-21.4439,-22.6709,-30.7671,9.9681,39.0475,-8.6764,-30.8616,-7.4073,18.3632,19.4684,8.337,10.7101,-35.2999,7.1184,-63.1839,7.3419,-13.8774,8.2123,0.4939,15.1122,46.1844,-3.341,16.8887,-38.2098,37.6967,5.2995,26.7976,16.8192,-23.2636,25.4099,-18.9935,18.7306,20.0169,-6.7697,60.4412,-15.7763,28.6786,26.8365,-55.5322,-14.6338,19.5,0.9783,8.1402,19.2986,-11.3631,-6.5031,1.7237,-2.745,-15.1787,5.2532,-8.1609,-15.315,3.8639,32.0019,-44.8861,-19.9472,4.0406,-23.6296,17.9484,13.1543,-15.2347,-14.7932,-14.3798,27.7666,40.3707,-4.6044,-15.1914,-17.6412,-8.3691,-1.9153,-20.7766,24.1618,-10.8879,23.6964,-8.6937,21.3745,18.21,0.3244,-7.2169,-30.5414,-31.1802,12.9007,-27.386,6.5547,-27.4039,-0.0333,-20.8781,0.6471,42.5062,-21.8372,-9.1928,11.6025,23.4285,-1.6631,-8.7733,19.1944,-42.9565,-3.5248,-1.8103,2.1937,28.4051,-1.2552,-29.7799,-12.6006,32.447,10.9337,7.7163,7.6425,-15.7202,-1.1797,16.452,-2.0437])
    w = np.array([606.8017,-240.2183,-68.3464,80.061,-2.2299,-79.3557,65.7109,-246.8982,23.2484,155.4221,117.6655,-163.091,-75.0391,-112.9862,-103.4045,-185.2921,49.123,186.9448,-37.2344,-164.4319,-104.5521,109.5031,137.1446,63.0439,123.3055,-98.9545,86.3617,-375.8437,-63.0564,-103.3159,44.4045,36.3136,134.5989,166.1676,42.4676,21.3996,-177.07,222.2477,69.0997,106.0827,63.6233,1.5477,191.335,-170.2144,-40.645,130.3904,-87.1691,296.4545,11.5642,131.015,70.0397,-201.9716,-73.0731,20.57,31.088,77.2114,171.4528,-9.1405,-41.9037,40.029,61.3606,-133.4075,69.3548,-107.3001,-13.693,-35.0387,204.4996,-214.2851,-219.8495,-3.5714,-199.1899,121.8092,41.295,-23.837,-90.1477,-126.9926,102.6492,203.4558,25.0447,-78.7034,1.699,-43.2496,-21.2928,-22.8441,42.3011,-42.5697,76.2054,24.7772,100.8049,167.2447,68.9699,-22.4896,-164.9182,-188.5559,-30.0277,-163.369,19.8574,-126.3183,-87.296,-105.6453,-3.5826,127.1411,-23.0229,-14.3509,56.5845,73.4919,102.2064,-56.149,111.3715,-175.484,-97.5988,-173.1644,-53.2611,151.3999,-0.2323,-137.9323,-79.755,110.2686,131.0864,90.1431,103.0315,-68.1877,-12.9456,66.3717,-1.7919])

    evaluate_model(loss_fun, w, x, y)

    # w = optimize_adam(loss_fun, w, x, y, 1024, 0.01, 0.9, 0.999, 100, 1e-6)
    w = minimize_bfgs(lambda w_: loss_fun(w_, x, y), w, maxiter=1000, gtol=1e-5)
    # w = scipy.optimize.minimize(lambda w_: loss_fun(w_, x, y), w, jac=True, method="CG", options={"maxiter": 100, 'gtol': 1e-6}).x
    evaluate_model(loss_fun, w, x, y)

    print(','.join(["{:.4f}".format(wi).rstrip('0') for wi in w]))


if __name__ == "__main__":

    np.random.seed(0)

    t0 = time.perf_counter()

    # fit(rosenbrock)
    # fit(loss_lnp)
    fit(loss_mse)

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
