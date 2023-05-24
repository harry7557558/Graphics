# Fit binary images
# Exports LaTeX to paste into Desmos

from PIL import Image, ImageOps

import numpy as np
import scipy.optimize

import time


def load_image(path):
    """Load a binary image, returns coordinates and value"""
    img = Image.open(path)
    img = ImageOps.flip(img)
    # img = img.resize((128, 128), resample=Image.NEAREST)
    # img = img.resize((64, 64), resample=Image.NEAREST)
    # img = img.resize((32, 32), resample=Image.NEAREST)
    pixels = np.array(img).mean(axis=2) / 255.0
    x1 = (np.arange(img.width) + 0.5) / img.width
    x2 = (np.arange(img.height) + 0.5) / img.height
    x = 2.0 * np.stack(np.meshgrid(x1, x2), axis=2) - 1.0
    y = 2.0 * pixels - 1.0
    x = x.reshape((img.width*img.height, 2)).T
    y = y.flatten()
    if 0:
        for i in range(0, img.width*img.height):
            if y[i] == -1.0:
                print("({:.2f},{:.2f})".format(x[0][i], x[1][i]), end=',')
        print(end='\n')
    return (x, y)


def model_poly(x: np.ndarray, deg: int):
    """Polynomial with 2 variables
        dimension: deg*(deg+1)//2
       Not so good when deg is too large
    """
    b0 = []
    tags = []
    for d in range(deg+1):
        for i in range(d+1):
            j = d - i
            b0.append(x[0]**i * x[1]**j)
            tags.append(f"x^{{{i}}}y^{{{j}}}")
    return (np.array(b0), tags)


def model_trig(x: np.ndarray, deg: int):
    """Trigonometric
        dimension: (2*deg+1)**2
       Sometimes results in too many weights"""
    def format_trig(fun, k, varname):
        return "" if k == 0 else f"\\{fun}({varname})" if k == 1 else f"\\{fun}({k}{varname})"
    b0 = []
    tags = []
    for i in range(deg+1):
        for j in range(deg+1):
            i, j = i*2, j*2
            b0.append(np.cos(i*x[0])*np.cos(j*x[1]))
            tags.append(format_trig('cos', i, 'x') +
                        format_trig('cos', j, 'y'))
            if i != 0:
                b0.append(np.sin(i*x[0])*np.cos(j*x[1]))
                tags.append(format_trig('sin', i, 'x') +
                            format_trig('cos', j, 'y'))
                if j != 0:
                    b0.append(np.sin(i*x[0])*np.sin(j*x[1]))
                    tags.append(format_trig('sin', i, 'x') +
                                format_trig('sin', j, 'y'))
            if j != 0:
                b0.append(np.cos(i*x[0])*np.sin(j*x[1]))
                tags.append(format_trig('cos', i, 'x') +
                            format_trig('sin', j, 'y'))
    return (np.array(b0), tags)


def print_model(w, tags):
    w = np.round(9999 * w / np.max(np.abs(w))).astype(int)
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
    print(w, val, np.linalg.norm(grad))


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


def optimize_lnp(w, b0, y):
    # res = scipy.optimize.minimize(lambda x: loss_lnp(x, b0, y)[0], w,
    #                               method="CG", options={'gtol': 0.0})
    res = scipy.optimize.minimize(lambda x: loss_lnp(x, b0, y), w, jac=True,
                                  method="BFGS", options={'gtol': 1e-4})
    print(res.nit, res.nfev, res.njev)
    return res.x


def loss_mse(w, b0, y):
    """Loss function is quadratic form-alike when w is close to the minimum"""
    f = np.matmul(w, b0)
    d = np.tanh(f) - y
    v = np.average(d**2)
    g = np.matmul(b0, 2.0 * d / np.cosh(f)**2) / len(f)
    return (v, g)


def optimize_mse(w, b0, y):
    res = scipy.optimize.minimize(lambda x: loss_mse(x, b0, y), w, jac=True,
                                  method="BFGS", options={'gtol': 0.0})
    print(res.nit, res.nfev, res.njev)
    return res.x


if __name__ == "__main__":

    x, y = load_image("data/binimg1.png")

    b0, tags = model_trig(x, 2)

    t0 = time.perf_counter()

    w = initial_guess(b0, y)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    w = optimize_lnp(w, b0, y)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    w = optimize_mse(w, b0, y)
    print_loss(loss_mse, w, b0, y)
    print_model(w, tags)

    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}secs".format(t1-t0))
