# fit color images using color(t)=a+b*t+c*cos(d*t+e)

import numpy as np
from math import sin, cos, sqrt, hypot, atan2, pi, log10
from img.load_data import load_image
import scipy.sparse.linalg

import time
import matplotlib.pyplot as plt
import sys


def func(t, a, b, c, d, e):
    return a + b*t + c*np.cos(d*t+e)


def fit_freq(x, y, f):
    """ least square fitting when the frequency is given """
    cov = np.zeros((4, 4))
    vec = np.zeros((4))
    for i in range(len(x)):
        di = np.array([1, x[i], cos(f*x[i]), sin(f*x[i])])
        cov += np.tensordot(di, di, axes=0)
        vec += y[i] * di
    #a, b, u, v = np.linalg.solve(cov, vec)
    a, b, u, v = scipy.sparse.linalg.cg(cov, vec)[0]  # more stable
    loss = sqrt(np.average((y-(a+b*x+u*np.cos(f*x)+v*np.sin(f*x)))**2))
    return ([a, b, hypot(u, v), f, -atan2(v, u)], loss)


def minimize_1d(fun, x0, x1):
    """ minimize a one-dimensional function using golden section search """
    g1, g0 = 2.0*sin(0.1*pi), 1.0-2.0*sin(0.1*pi)
    t0 = g1*x0+g0*x1
    t1 = g0*x0+g1*x1
    y0, y1 = fun(t0), fun(t1)
    for i in range(64):
        if y0 < y1:
            x1 = t1
            y1 = y0
            t1 = t0
            t0 = g1*x0+g0*x1
            y0 = fun(t0)
        else:
            x0 = t0
            y0 = y1
            t0 = t1
            t1 = g0*x0+g1*x1
            y1 = fun(t1)
        if x1-x0 < 1e-4:
            break
    return (t0, y0) if y0 < y1 else (t1, y1)


def fit_array(x, y):

    def loss_fun(f):
        return fit_freq(x, y, f)[1]

    freqs = (6.28*np.arange(sqrt(0.01), sqrt(3.), 0.1)**2).tolist()
    losses = [loss_fun(f) for f in freqs]

    best_i, min_dif = -1, 1e100
    for i in range(1, len(freqs)-1):
        if losses[i] < min_dif:
            best_i, min_dif = i, losses[i]

    f = minimize_1d(loss_fun, freqs[best_i-1], freqs[best_i+1])[0]
    coes = fit_freq(x, y, f)[0]
    loss = sqrt(np.average((func(x, *coes) - y)**2))
    print("loss = {:.1f} / 255".format(255*loss))
    return (coes, loss)


def fit_color(cols):
    w = len(cols)
    x = np.array([(i+.5)/w for i in range(w)])
    cr = fit_array(x, np.array([c[0] for c in cols]))
    cg = fit_array(x, np.array([c[1] for c in cols]))
    cb = fit_array(x, np.array([c[2] for c in cols]))
    return (cr[0], cg[0], cb[0])


def debug_fit(arr):
    """ debug color fitting """
    w = len(arr)
    x = np.array([(i+.5)/w for i in range(w)])

    t0 = time.perf_counter()
    coes, loss = fit_array(x, np.array(arr))
    t1 = time.perf_counter()
    print("Time elapsed: {:.2f}s".format(t1-t0))

    freqs = (6.28*np.arange(sqrt(0.01), sqrt(3.), 0.1)**2).tolist()
    losses = [fit_freq(x, arr, f)[1] for f in freqs]

    plt.clf()
    plt.plot(freqs, losses)
    freqs.append(coes[3])
    losses.append(loss)
    plt.plot(freqs, losses, "o")
    plt.show()

    sys.exit()


def debug_color(data):
    """ debug color fitting """
    w = len(data)
    x = np.array([(i+.5)/w for i in range(w)])
    cr, cg, cb = fit_color(data)

    plt.clf()
    plt.plot(x, data[:, 0], "r")
    plt.plot(x, data[:, 1], "g")
    plt.plot(x, data[:, 2], "b")
    plt.plot(x, func(x, *cr), "r")
    plt.plot(x, func(x, *cg), "g")
    plt.plot(x, func(x, *cb), "b")
    plt.show()

    sys.exit()


def float2str_min(x, prec=3):
    """ float to string for color output """
    sign = '+' if x >= 0.0 else '-'
    s = "{:.{prec}f}".format(abs(x), prec=prec)  # 3-4 decimal places
    while s[-1] == '0':
        s = s[:len(s)-1]
    while s[0] == '0':
        s = s[1:]
    if s == '.':
        s = '0.'
    return sign + s


def generate_code(pics):

    js_code = """const ColorFunctions = {
  clp: function(x) {
    return Math.round(255.*(x<0.?0.:x>1.?1.:x));
  },
  tocol: function(r, g, b) {
    return 'rgb('+this.clp(r)+','+this.clp(g)+','+this.clp(b)+')';
  },
"""

    cpp_code = """template<typename vec3, typename Float>
class ColorFunctions {
  static Float clp(Float x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
"""

    for (colorname, data) in pics:
        print(colorname)
        coes = fit_color(data)
        js_code += "  " + colorname + ": function(t) {\n"
        cpp_code += "  static vec3 " + colorname + "(Float t) {\n"
        for i in range(3):
            prec = max(int(log10(abs(coes[i][2])))+2, 3)
            a = float2str_min(coes[i][0], prec=3)
            b = float2str_min(coes[i][1], prec=3)
            c = float2str_min(coes[i][2], prec=3)
            d = float2str_min(coes[i][3], prec=prec)
            e = float2str_min(coes[i][4], prec=prec)
            st = f"{a}{b}*t{c}*cos({d}*t{e})".lstrip('+').replace('(+', '(')
            js_code += "    var " + 'rgb'[i] + " = " + \
                st.replace("cos", "Math.cos") + ';\n'
            cpp_code += "    Float " + 'rgb'[i] + " = " + \
                st + ';\n'
        js_code += "    return this.tocol(r, g, b);\n  },\n"
        cpp_code += "    return vec3(clp(r),clp(g),clp(b));\n  }\n"

    js_code += "};"
    cpp_code += "};\n"

    return {"js_code": js_code, "cpp_code": cpp_code}


if __name__ == "__main__":

    names = open("img/index", "r").read().split('\n')
    color_pics = [(names[i], load_image("img/"+str(i+1)+".png"))
                  for i in range(50)]

    #debug_fit(np.array([c[2] for c in color_pics[41][1]]))
    #debug_color(np.array(color_pics[2][1], dtype=np.float64))

    code = generate_code(color_pics)

    open("cosine.js", "wb").write(bytearray(code['js_code'], 'utf-8'))
    open("cosine.h", "wb").write(bytearray(code['cpp_code'], 'utf-8'))
