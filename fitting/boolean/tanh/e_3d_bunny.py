# Fit implicit 3D model

# Exports text expression compatible with
# https://harry7557558.github.io/tools/raymarching-implicit/index.html

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


def model_poly(x: np.ndarray, deg: int):
    """Polynomial with 3 variables
       Number of weights: nCr(deg+3,3)
       Not so good when deg is large
    """
    def fp(v, p):
        return '' if p == 0 else v if p == 1 else f"{v}^{p}"
    b0 = []
    tags = []
    for d in range(deg+1):
        for i in range(d+1):
            for j in range(d-i+1):
                k = d - (i+j)
                b0.append(x[0]**i * x[1]**j * x[2]**k)
                tags.append(fp('x', i) + fp('y', j) + fp('z', k))
    return np.array(b0), tags


def model_trig(x: np.ndarray, deg: int):
    """Trigonometric
       Number of weights: (2*deg+1)**3
       Sometimes results in too many weights"""
    def format_trig(k, varname):
        if k == 0:
            return ["", ""]
        br = f"({varname})" if k == 1 else f"({k}{varname})"
        return ["cos" + br, "sin" + br]
    b0 = []
    tags = []
    for i in range(deg+1):
        i *= 2
        cossinx = [np.cos(i*x[0]), np.sin(i*x[0])]
        tagx = format_trig(i, 'x')
        for j in range(deg+1):
            j *= 2
            cossiny = [np.cos(j*x[1]), np.sin(j*x[1])]
            tagy = format_trig(j, 'y')
            for k in range(deg+1):
                k *= 2
                cossinz = [np.cos(k*x[2]), np.sin(k*x[2])]
                tagz = format_trig(k, 'z')
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if (i == 0 and di == 1) or (j == 0 and dj == 1) or (k == 0 and dk == 1):
                                continue
                            b0.append(cossinx[di] * cossiny[dj] * cossinz[dk])
                            tags.append(tagx[di] + tagy[dj] + tagz[dk])
    return (np.array(b0), tags)


def print_model(w, tags):
    w = np.round(9999 * w / np.max(np.abs(w))).astype(int)
    print(','.join([str(wi) for wi in w]))
    s = ""
    for (wi, tag) in zip(w, tags):
        wi = str(wi)
        if wi == "0":
            continue
        if wi[0] != '-':
            wi = '+' + wi
        if wi in ['+1', '-1'] and tag != "":
            wi = wi[0]
        s += wi + tag
    print(s.lstrip('+') + "=0")


def print_loss(lossfun, w, b0, y):
    val, grad = lossfun(w, b0, y)
    print(val, np.linalg.norm(grad))


def initial_guess(b0, y):
    """Initial guess for weights
        Solving a linear system works in O(N^3)"""
    left = np.matmul(b0, b0.T)
    right = np.matmul(b0, y)
    return np.linalg.solve(left, right)


def loss_lnp(w, b0, y):
    """Loss function is quadratic form-alike when w is away from the minimum"""
    f = np.matmul(w, b0)
    e2f = np.exp(2.0*f)
    e_2f = 1.0 / e2f
    v1 = (1.0-y) * np.log(1.0+e2f)
    v2 = (1.0+y) * np.log(1.0+e_2f)
    g1 = 2.0 * (1.0-y) * e2f / (1.0+e2f)
    g2 = -2.0 * (1.0+y) * e_2f / (1.0+e_2f)
    v = np.average((v1+v2)**2)
    g = np.matmul(b0, 2.0*(v1+v2)*(g1+g2)) / len(w)
    return (v, g)


def optimize_lnp(w, b0, y, tol):
    res = scipy.optimize.minimize(lambda x: loss_lnp(x, b0, y), w, jac=True,
                                  method="BFGS", options={'gtol': tol})
    print(res.nit, res.nfev, res.njev)
    return res.x


def loss_mse(w, b0, y):
    """Loss function is quadratic form-alike when w is close to the minimum"""
    f = np.matmul(w, b0)
    d = np.tanh(f) - y
    v = np.average(d**2)
    g = np.matmul(b0, 2.0 * d / np.cosh(f)**2) / len(f)
    return (v, g)


def optimize_mse(w, b0, y, tol):
    res = scipy.optimize.minimize(lambda x: loss_mse(x, b0, y), w, jac=True,
                                  method="BFGS", options={'gtol': tol})
    print(res.nit, res.nfev, res.njev)
    return res.x


def fit_poly():
    """Fit the dataset using polynomial regression model"""

    def model(x):
        return model_poly(x, 8)

    path = "../v_sdfbunny_128x128x128_uint8.raw"
    x, y = load_data(path, 16)
    b0, tags = model(x)

    w = initial_guess(b0, y)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    w = optimize_lnp(w, b0, y, 1e-2)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    x, y = load_data(path, 32)
    b0, tags = model(x)

    w = optimize_mse(w, b0, y, 1e-4)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    x, y = load_data(path, 64)
    b0, tags = model(x)

    w = optimize_mse(w, b0, y, 1e-8)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)


def fit_trig():
    """Fit the dataset using trigonometric regression model"""

    def model(x):
        return model_trig(x, 2)

    path = "../v_sdfbunny_128x128x128_uint8.raw"
    x, y = load_data(path, 16)
    b0, tags = model(x)

    w = initial_guess(b0, y)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    w = optimize_lnp(w, b0, y, 1e-2)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    x, y = load_data(path, 32)
    b0, tags = model(x)

    w = optimize_mse(w, b0, y, 1e-3)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    x, y = load_data(path, 64)
    b0, tags = model(x)

    w = optimize_mse(w, b0, y, 1e-6)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)


if __name__ == "__main__":

    t0 = time.perf_counter()

    # fit_poly()
    fit_trig()

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
