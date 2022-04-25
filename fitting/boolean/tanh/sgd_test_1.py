# Test SGD techniques in fitting the bunny

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
    """Loss function is quadratic form-alike when w is away from the minimum"""
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


def loss_mse(w, x, y, hess: bool = False):
    """Loss function is quadratic form-alike when w is close to the minimum"""
    f, f_grad = model(w, x)
    d = np.tanh(f) - y
    c2 = np.cosh(f) ** 2
    v = np.average(d**2)
    g = np.matmul(f_grad, 2.0 * d / c2) / len(f)
    if hess:
        # hf = 2.0 * (1.0 - d * np.sinh(2.0*f)) / c2**2
        hf = (2.0 * (1.0 - d * np.sinh(2.0*f)) / c2) / c2
        # h = np.einsum('ab,cd,e->ac', f_grad, f_grad, hf)
        h = np.zeros((N_WEIGHTS, N_WEIGHTS))
        for i in range(len(hf)):
            h += np.tensordot(f_grad[:, i], f_grad[:, i], axes=0) * hf[i]
        h /= len(f)
        return v, g, h
    return v, g


def evaluate_model(lossfun, w, x, y):
    """Print the loss and the magnitude of gradient of loss"""
    val, grad = lossfun(w, x, y)
    print("Evaluate loss:", val, np.linalg.norm(grad))


# Plotting


losses = []
ax = None
plot_data = None


def update_plot(loss):
    global ax, plot_data
    losses.append(loss)

    x = np.arange(len(losses)) + 1

    is_init = ax is None
    if is_init:
        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)

    plt.yscale('log')
    plot_data = ax.plot(x, losses, 'r')

    plt.draw()
    plt.pause(0.01)


# Stochastic gradient descent

def optimize_sgd(lossfun, w, x, y, batch_size, learning_step, momentum, max_epoch, gtol):

    loss = 0.0
    grad = np.zeros((N_WEIGHTS))

    train_order = np.arange(len(y))

    for epoch in range(max_epoch):
        np.random.shuffle(train_order)
        for batch in range(0, len(y), batch_size):
            indices = train_order[batch:batch+batch_size]
            loss_t, grad_t = lossfun(w, x[:, indices], y[indices])
            loss = momentum * loss + (1.0-momentum) * loss_t
            grad = momentum * grad + (1.0-momentum) * grad_t
            w -= learning_step*grad
        grad_norm = np.linalg.norm(grad)
        print(f"Epoch {epoch}, loss={loss}, grad={grad_norm}")
        if grad_norm < gtol:
            break

    return w


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
        print(f"Epoch {epoch}, loss={loss}, grad={grad_norm}")
        update_plot(loss)
        if grad_norm < gtol:
            break

    return w


def optimize_cg(lossfun, w, x, y, batch_size, learning_step, momentum, max_epoch, gtol):
    # some messy guessed code, overperforms simple SGD but is unstable, worse than Adam
    # not to be used in practice

    loss = 0.0
    grad = np.zeros((len(w)))
    cgrad = np.zeros((len(w)))

    train_order = np.arange(len(y))

    for epoch in range(max_epoch):
        np.random.shuffle(train_order)
        for batch in range(0, len(y), batch_size):
            indices = train_order[batch:batch+batch_size]
            loss_t, grad_t = lossfun(w, x[:, indices], y[indices])
            grad_t *= -1.0
            loss = momentum * loss + (1.0-momentum) * loss_t
            beta = np.dot(grad_t, grad_t-grad) / (np.dot(grad, grad)+1e-8)
            beta = max(beta, 0.)
            cgrad = grad_t + beta * cgrad
            grad = grad_t
            h = 0.01
            v0 = np.dot(grad_t, cgrad)
            loss_t_1, grad_t_1 = lossfun(w+h*cgrad, x[:, indices], y[indices])
            grad_t_1 *= -1.0
            v1 = np.dot(grad_t_1, cgrad)
            dvda = (v1-v0)/h
            alpha = -v0/dvda
            alpha = max(alpha, 0.)
            w += learning_step * alpha * cgrad
        # grad_norm = np.linalg.norm(cgrad)
        grad_norm = np.linalg.norm(grad)
        print(f"Epoch {epoch}, loss={loss}, grad={grad_norm}")
        if grad_norm < gtol:
            break

    return w


def optimize_newton(lossfun, w, x, y, batch_size, learning_step, max_epoch, gtol):
    # Works but not so stable

    loss = 0.0
    dw = np.zeros((N_WEIGHTS))

    train_order = np.arange(len(y))

    for epoch in range(max_epoch):
        np.random.shuffle(train_order)
        for batch in range(0, len(y), batch_size):
            indices = train_order[batch:batch+batch_size]
            val, grad, hess = lossfun(w, x[:, indices], y[indices], True)
            loss = val
            dw = -np.linalg.solve(hess, grad)
            if np.isnan(np.sum(dw)):
                print("NAN")
                break
            if np.dot(dw, 0.5 * np.matmul(hess, dw) + grad) > 0.:
                dw = -dw
            w += learning_step * dw
        grad_norm = np.linalg.norm(grad)
        print(f"Epoch {epoch}, loss={loss}, grad={grad_norm}")
        if grad_norm < gtol:
            break

    return w


def fit():
    """Fit the dataset using polynomial regression model"""

    path = "../v_sdfbunny_128x128x128_uint8.raw"
    x, y = load_data(path, 64)

    w = -1.0+2.0*np.random.random((N_WEIGHTS))
    # w = np.array([9999,-3120,-1738,814,-158,-2273,1753,-4161,799,1446,1487,-2523,-950,-1127,-1120,-4080,792,3957,-243,-2412,-2123,2261,1799,619,1498,-3619,738,-6012,-322,-1431,-928,1154,1623,3167,-314,1624,-3383,2961,1622,3095,1016,-394,2871,-1680,1569,386,-1206,4634,424,2291,-277,-3887,-107,1103,-476,1667,1913,-477,-1951,1207,192,-760,1356,-1006,-789,1555,2554,-5176,-2806,-265,-1724,1675,551,-1053,-715,-1839,2099,4071,518,-1969,341,-894,30,-470,950,-1081,2265,293,1883,2288,304,-946,-3354,-2694,-986,-3470,1012,-2513,-1469,-1772,1313,2359,-646,429,557,1960,86,-1988,1982,-4523,-765,-975,103,3969,304,-2234,-1569,3040,1587,1037,353,-1711,-211,868,204],dtype=np.float64)
    w = np.array([1.040,0.040,0.007,0.040,0.078,-0.219,0.061,-0.074,-0.024,0.019,0.004,-0.240,-0.098,0.068,0.026,0.119,0.109,0.054,0.015,0.090,0.009,0.128,0.124,0.113,0.029,-0.306,0.068,0.088,-0.014,-0.035,0.001,-0.359,-0.280,0.082,0.047,-0.483,0.107,-0.200,0.041,0.327,0.016,-0.102,0.000,0.133,-0.001,-0.027,-0.000,-0.591,-0.046,0.135,0.053,-0.244,-0.093,0.050,0.014,-0.230,0.263,0.125,0.094,0.015,-0.021,-0.240,0.002,-0.073,0.002,-0.047,0.001,-0.264,-0.297,0.329,0.085,0.151,0.074,0.112,0.040,0.050,0.031,0.068,0.012,0.062,0.011,0.049,0.119,0.027,0.067,-0.030,0.030,-0.016,0.045,-0.250,-0.032,0.055,0.005,-0.242,-0.031,0.015,0.003,-0.002,-0.051,0.027,0.012,0.013,-0.087,0.050,0.008,0.093,0.027,0.061,0.099,0.124,0.022,0.184,0.012,0.102,0.019,0.081,0.012,0.090,0.161,0.008,0.002,0.056,0.099,0.102,0.030])

    # v, v1, v2 = loss_mse(w, x, y, True)
    # cg = scipy.optimize.check_grad(lambda w_: loss_mse(w_, x, y, True)[1][13],
    #                                lambda w_: loss_mse(w_, x, y, True)[2].T[13], w)
    # print(cg)
    # return

    evaluate_model(loss_mse, w, x, y)

    # w = optimize_sgd(loss_mse, w, x, y, 1024, 0.1, 0.9, 20, 1e-6)
    w = optimize_adam(loss_mse, w, x, y, 1024, 0.01, 0.9, 0.999, 100, 1e-6)
    # w = optimize_cg(loss_mse, w, x, y, 1024, 0.1, 0.9, 10, 1e-6)
    evaluate_model(loss_mse, w, x, y)

    # w = optimize_adam(loss_mse, w, x, y, 1024, 0.01, 0.9, 0.999, 1, 1e-6)
    # res = scipy.optimize.minimize(lambda w_: loss_mse(w_, x, y), w, jac=True,
    #                               method="BFGS", options={'gtol': 1e-6, 'maxiter': 10},
    #                               callback=None)
    # print(res.nit, res.nfev)
    # w = res.x
    # evaluate_model(loss_mse, w, x, y)

    # w = optimize_adam(loss_mse, w, x, y, 1024, 0.01, 0.9, 0.999, 30, 1e-6)
    # w = optimize_newton(loss_mse, w, x, y, len(y), 0.01, 100, 1e-6)

    w = np.round(9999 * w / np.max(np.abs(w))).astype(int)
    print(','.join([str(wi) for wi in w]))


if __name__ == "__main__":

    np.random.seed(0)

    t0 = time.perf_counter()

    fit()

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
