# Fit binary images
# Exports LaTeX to paste into Desmos

# Doesn't work well because of local minima

from PIL import Image, ImageOps

import numpy as np
import scipy.optimize

import time


def load_image(path):
    """Load a binary image, returns pixel values"""
    # get pixels
    n = 128
    img = Image.open(path)
    img = ImageOps.flip(img)
    img = img.resize((n, n), resample=Image.NEAREST)
    pixels = np.array(img).mean(axis=2) / 255.0
    x1 = (np.arange(img.width) + 0.5) / img.width
    x2 = (np.arange(img.height) + 0.5) / img.height
    x = 2.0 * np.stack(np.meshgrid(x1, x2), axis=2) - 1.0
    y = 2.0 * pixels - 1.0

    # to SDF (BFS)
    yd = np.ones((n, n)) * 2.0
    yp = np.zeros((n, n, 2))
    st = set({})
    for i in range(n-1):
        for j in range(n-1):
            s = y[i][j] + y[i][j+1] + y[i+1][j] + y[i+1][j+1]
            if s not in [-4.0, 4.0]:
                g = 0.25 * (x[i][j] + x[i][j+1] + x[i+1][j] + x[i+1][j+1])
                st.add((i, j))
                for u in range(2):
                    for v in range(2):
                        yd[i+u][j+v] = np.linalg.norm(x[i+u][j+v] - g)
                        yp[i+u][j+v] = g
    while len(st) != 0:
        ss = set({})
        for i0, j0 in st:
            p0 = yp[i0][j0]
            for di in range(-1, 3):
                for dj in range(-1, 3):
                    i, j = i0+di, j0+dj
                    if i < 0 or i >= n or j < 0 or j >= n:
                        continue
                    d = np.linalg.norm(x[i][j] - p0)
                    if d < yd[i][j]:
                        yd[i][j] = d
                        yp[i][j] = p0
                        ss.add((i, j))
        st = ss
    y = y * yd

    # visualize
    if 0:
        import matplotlib.pyplot as plt
        plt.imshow(y, interpolation='nearest')
        plt.show()

    x = x.reshape((img.width*img.height, 2)).T
    y = y.flatten()
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
        for i in range(n):
            self.x0[i] = -1.0+2.0*van_der_corput(i+1, 2)
            self.x1[i] = -1.0+2.0*van_der_corput(i+1, 3)
        self.b = 1.0

    def initial_guess(self, x, y):
        b0 = []
        for i in range(self.n):
            dx1 = x[0] - self.x0[i]
            dy1 = x[1] - self.x1[i]
            b0.append(np.sqrt(dx1*dx1 + dy1*dy1))
        b0.append(np.ones(x.shape[1]))
        b0 = np.array(b0)
        left = np.matmul(b0, b0.T)
        right = np.matmul(b0, y)
        w = np.linalg.solve(left, right)
        self.m = w[0:self.n]
        self.b = w[self.n]

    def get_weights(self):
        return self.pack_weights(self.n, self.m, self.x0, self.x1, self.b)

    @staticmethod
    def pack_weights(n, m, x0, x1, b):
        w = np.zeros((3*n+1))
        w[0:n] = m
        w[n:2*n] = x0
        w[2*n:3*n] = x1
        w[3*n] = b
        return w

    @staticmethod
    def unpack_weights(n, w):
        m = w[0:n]
        x0 = w[n:2*n]
        x1 = w[2*n:3*n]
        b = w[3*n]
        return m, x0, x1, b

    def eval_raw(self, x):
        """Non-static version of eval"""
        s = self.b * np.ones(x.shape[1])
        for i in range(n):
            dx1 = x[0] - self.x0[i]
            dy1 = x[1] - self.x1[i]
            s += self.m[i] * np.sqrt(dx1*dx1 + dy1*dy1)
        return s

    @staticmethod
    def eval(n, w, x):
        """Value"""
        m, x0, x1, b = RbfModel.unpack_weights(n, w)
        s = b * np.ones(x.shape[1])
        for i in range(n):
            dx1 = x[0] - x0[i]
            dy1 = x[1] - x1[i]
            s += m[i] * np.sqrt(dx1*dx1 + dy1*dy1)
        return s

    @staticmethod
    def eval_grad(n, w, x):
        """Value + gradient"""
        m, x0, x1, b = RbfModel.unpack_weights(n, w)
        s = b * np.ones(x.shape[1])
        g = np.ones((len(w), x.shape[1]))
        for i in range(n):
            dx1 = x[0] - x0[i]
            dy1 = x[1] - x1[i]
            r = np.sqrt(dx1*dx1 + dy1*dy1)
            s += m[i] * r
            ddx0 = -m[i] * dx1 / r
            ddx1 = -m[i] * dy1 / r
            g[i] = r
            g[i+n] = ddx0
            g[i+2*n] = ddx1
        return s, g

    @staticmethod
    def print_model(n, w):
        m, x0, x1, b = RbfModel.unpack_weights(n, w)
        s = "{:.4f}".format(b)
        p = []
        for i in range(n):
            p.append("({:.4f},{:.4f})".format(x0[i], x1[i]))
            s += "{:+.4f}\\sqrt{{(x{:+.4f})^2+(y{:+.4f})^2}}".format(
                m[i], -x0[i], -x1[i])
        print(','.join(p))
        print(s + "=0")


def print_loss(lossfun, n, w, x, y):
    val, grad = lossfun(n, w, x, y)
    print(w, val, np.linalg.norm(grad))


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
    # res = scipy.optimize.basinhopping(
    #     lambda w_: loss_lnp(n, w_, x, y)[0], x0=w)
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
        model.initial_guess(x, y)
        d = model.eval_raw(x) - y
        loss = np.sqrt(np.mean(d*d))
        mag = max(np.mean(np.maximum(abs(model.x0)-1, 0.)**2 +
                          np.maximum(abs(model.x1)-1, 0.)**2), 0.0)
        # return loss
        return loss + 0.5*mag**2

    x0 = np.array([model.x0, model.x1])
    e0 = get_energy(x0)
    temp = 2.0  # temperature
    while temp > 0.01:
        dx = temp * np.random.multivariate_normal(
            np.zeros(2), np.identity(2), (model.n)).T
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

    x, y_sd = load_image("data/binimg2.png")
    y = np.sign(y_sd)

    t0 = time.perf_counter()

    n = 20
    model = RbfModel(n)
    model.initial_guess(x, y)
    w = model.get_weights()

    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    optimize_sdf_mse(model, x, y_sd)
    w = model.get_weights()
    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    w = optimize_lnp(n, w, x, y)
    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    w = optimize_mse(n, w, x, y)
    print_loss(loss_mse, n, w, x, y)
    RbfModel.print_model(n, w)

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
