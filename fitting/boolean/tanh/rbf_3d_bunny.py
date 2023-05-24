# Fit implicit 3D model

# Exports text expression compatible with
# https://harry7557558.github.io/tools/raymarching-implicit/index.html

# Doesn't work well because of local minima

import numpy as np
import scipy.optimize

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
    y = -(data - 0.5*255) / 64.0
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


def van_der_corput(n, b):
    """Quasi-random"""
    x = 0.0
    e = 1.0 / b
    while n != 0:
        d = n % b
        x += d * e
        e /= b
        n = n // b
    return x


class RbfModel:
    """ z = b + sum[ m * sqrt((x-x0)^2 + (y-x1)^2) ] """

    def __init__(self, n: int):
        self.n = n
        self.m = np.ones(n)
        self.x0 = np.zeros(n)
        self.x1 = np.zeros(n)
        self.x2 = np.zeros(n)
        for i in range(n):
            self.x0[i] = -1.0+2.0*van_der_corput(i+1, 2)
            self.x1[i] = -1.0+2.0*van_der_corput(i+1, 3)
            self.x2[i] = -1.0+2.0*van_der_corput(i+1, 5)
        self.b = 1.0

    def initial_guess(self, x, y):
        b0 = []
        for i in range(self.n):
            dx1 = x[0] - self.x0[i]
            dy1 = x[1] - self.x1[i]
            dz1 = x[2] - self.x2[i]
            b0.append(np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1))
        b0.append(np.ones(x.shape[1]))
        b0 = np.array(b0)
        left = np.matmul(b0, b0.T)
        right = np.matmul(b0, y)
        w = np.linalg.solve(left, right)
        self.m = w[0:self.n]
        self.b = w[self.n]

    def get_weights(self):
        return self.pack_weights(self.n, self.m, self.x0, self.x1, self.x2, self.b)

    @staticmethod
    def pack_weights(n, m, x0, x1, x2, b):
        w = np.zeros((4*n+1))
        w[0:n] = m
        w[n:2*n] = x0
        w[2*n:3*n] = x1
        w[3*n:4*n] = x2
        w[4*n] = b
        return w

    @staticmethod
    def unpack_weights(n, w):
        m = w[0:n]
        x0 = w[n:2*n]
        x1 = w[2*n:3*n]
        x2 = w[3*n:4*n]
        b = w[4*n]
        return m, x0, x1, x2, b

    def eval_raw(self, x):
        """Non-static version of eval"""
        s = self.b * np.ones(x.shape[1])
        for i in range(n):
            dx1 = x[0] - self.x0[i]
            dy1 = x[1] - self.x1[i]
            dz1 = x[2] - self.x2[i]
            s += self.m[i] * np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
        return s

    @staticmethod
    def eval(n, w, x):
        """Value"""
        m, x0, x1, x2, b = RbfModel.unpack_weights(n, w)
        s = b * np.ones(x.shape[1])
        for i in range(n):
            dx1 = x[0] - x0[i]
            dy1 = x[1] - x1[i]
            dz1 = x[2] - x2[i]
            s += m[i] * np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
        return s

    @staticmethod
    def eval_grad(n, w, x):
        """Value + gradient"""
        m, x0, x1, x2, b = RbfModel.unpack_weights(n, w)
        s = b * np.ones(x.shape[1])
        g = np.ones((len(w), x.shape[1]))
        for i in range(n):
            dx1 = x[0] - x0[i]
            dy1 = x[1] - x1[i]
            dz1 = x[2] - x2[i]
            r = np.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1)
            s += m[i] * r
            ddx0 = -m[i] * dx1 / r
            ddx1 = -m[i] * dy1 / r
            ddx2 = -m[i] * dz1 / r
            g[i] = r
            g[i+n] = ddx0
            g[i+2*n] = ddx1
            g[i+3*n] = ddx2
        return s, g

    @staticmethod
    def print_model(n, w):
        m, x0, x1, x2, b = RbfModel.unpack_weights(n, w)
        w = np.array(m.tolist() + [b])
        w = np.round(9999 * w / np.max(np.abs(w))).astype(int)
        x = np.round(10000 * np.concatenate((x0, x1, x2))).astype(int)
        print(','.join([str(wi) for wi in w]))
        print(','.join([str(xi) for xi in x]))
        s = "{:.4f}".format(b)
        for i in range(n):
            s += "{:+.4f}sqrt((x{:+.4f})^2+(y{:+.4f})^2+(z{:+.4f})^2)".format(
                m[i], -x0[i], -x1[i], -x2[i])
        print(s + "=0")


def print_loss(lossfun, n, w, x, y):
    val, grad = lossfun(n, w, x, y)
    print(val, np.linalg.norm(grad))


def loss_lnp(n, w, x, y):
    """Loss function is quadratic form-alike when w is away from the minimum"""
    f, dfdw = RbfModel.eval_grad(n, w, x)
    e2f = np.exp(2.0*f)
    e_2f = 1.0 / e2f
    v1 = (1.0-y) * np.log(1.0+e2f)
    v2 = (1.0+y) * np.log(1.0+e_2f)
    g1 = 2.0 * (1.0-y) * e2f / (1.0+e2f)
    g2 = -2.0 * (1.0+y) * e_2f / (1.0+e_2f)
    v = np.average((v1+v2)**2)
    g = np.matmul(dfdw, 2.0*(v1+v2)*(g1+g2)) / x.shape[1]
    return v, g


def optimize_lnp(n, w, x, y):
    res = scipy.optimize.minimize(lambda w_: loss_lnp(n, w_, x, y), w, jac=True,
                                  method="BFGS", options={'gtol': 1e-4})
    print(res.nit, res.nfev, res.njev)
    return res.x


def loss_mse(n, w, x, y):
    """Loss function is quadratic form-alike when w is close to the minimum"""
    f, dfdw = RbfModel.eval_grad(n, w, x)
    d = np.tanh(f) - y
    v = np.average(d**2)
    g = np.matmul(dfdw, 2.0 * d / np.cosh(f)**2) / x.shape[1]
    return v, g


def optimize_mse(n, w, x, y):
    res = scipy.optimize.minimize(lambda w_: loss_mse(n, w_, x, y), w, jac=True,
                                  method="BFGS", options={'gtol': 1e-6})
    print(res.nit, res.nfev, res.njev)
    return res.x


def loss_sdf_mse(n, w, x, y, inc_grad: bool):
    """Loss function for SDF"""
    if inc_grad:
        f, dfdw = RbfModel.eval_grad(n, w, x)
        d = f - y
        v = np.average(d*d)
        g = np.matmul(dfdw, 2.0*d) / x.shape[1]
        return v, g
    else:
        d = RbfModel.eval(n, w, x) - y
        return np.average(d*d)


def optimize_sdf_mse(model: RbfModel, x, y):
    """Try global optimization"""
    np.random.seed(0)

    def get_energy(x0):
        model.x0 = x0[0]
        model.x1 = x0[1]
        model.x2 = x0[2]
        model.initial_guess(x, y)
        d = model.eval_raw(x) - y
        loss = np.sqrt(np.mean(d*d))
        mag = max(np.mean(np.maximum(abs(model.x0)-1, 0.)**2 +
                          np.maximum(abs(model.x1)-1, 0.)**2 +
                          np.maximum(abs(model.x2)-1, 0.)**2), 0.0)
        # return loss
        return loss + 0.5*mag**2

    x0 = np.array([model.x0, model.x1, model.x2])
    e0 = get_energy(x0)
    temp = 3.0  # temperature
    while temp > 0.005:
        dx = temp * np.random.multivariate_normal(
            np.zeros(3), np.identity(3), (model.n)).T
        x1 = x0 + dx
        e1 = get_energy(x0+dx)
        # if np.exp(-10.*(e1-e0)/temp) > np.random.random():
        if e1 < e0:
            x0 = x1
            e0 = e1
        print("T={:.4f} E={:.4f}".format(temp, e0), np.linalg.norm(x0))
        temp *= 0.98
    get_energy(x0)


if __name__ == "__main__":

    path = "../v_sdfbunny_128x128x128_uint8.raw"
    x, y_sd = load_data(path, 16)
    y = np.sign(y_sd)

    t0 = time.perf_counter()

    n = 30
    model = RbfModel(n)
    model.initial_guess(x, y)
    w = model.get_weights()

    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    optimize_sdf_mse(model, x, y_sd)
    w = model.get_weights()
    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    x, y_sd = load_data(path, 32)
    y = np.sign(y_sd)

    w = optimize_lnp(n, w, x, y)
    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    x, y_sd = load_data(path, 64)
    y = np.sign(y_sd)

    w = optimize_mse(n, w, x, y)
    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
