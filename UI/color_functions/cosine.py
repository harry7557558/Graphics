# fit color images using color(t)=a+b*t+c*cos(d*t+e)

import numpy as np
from math import sin, cos, sqrt, hypot, atan2
import scipy.stats
from scipy.optimize import curve_fit
from img.load_data import load_image


def func(t, a, b, c, d, e):
    return a + b*t + c*np.cos(d*t+e)


def initial_guess(x, y, search_step):
    # brute force search for the best frequency: u*cos(f*x)+v*sin(f*x)
    a_best, b_best, f_best, u_best, v_best = 0, 0, 0, -1, -1
    min_loss = 1e100
    for f in [6.28*t*t for t in np.arange(sqrt(0.1), sqrt(8.), search_step)]:
        cov = np.zeros((4, 4))
        vec = np.zeros((4))
        for i in range(len(x)):
            di = np.array([1, x[i], cos(f*x[i]), sin(f*x[i])])
            cov += np.tensordot(di, di, axes=0)
            vec += y[i] * di
        a, b, u, v = np.linalg.solve(cov, vec)
        loss = sqrt(np.average((y-(u*np.cos(x)+v*np.sin(x)))**2))
        if loss < min_loss:
            a_best, b_best, f_best, u_best, v_best = a, b, f, u, v
            min_loss = loss
    return [a_best, b_best, hypot(u_best, v_best), f_best, -atan2(v_best, u_best)]


def fit_array(x, y):
    guess = initial_guess(x, y, 0.1)
    try:
        popt, pcov = curve_fit(
            func, x, y, maxfev=2000, p0=guess)
    except:
        print("Fitting failed")
        popt = initial_guess(x, y, 0.01)

    loss = sqrt(np.average((func(x, *popt) - y)**2))
    print("loss = {:.1f} / 255".format(255*loss))
    return (popt, loss)


def fit_color(cols):
    w = len(cols)
    x = np.array([(i+.5)/w for i in range(w)])
    cr = fit_array(x, np.array([c[0] for c in cols]))
    cg = fit_array(x, np.array([c[1] for c in cols]))
    cb = fit_array(x, np.array([c[2] for c in cols]))
    return (cr[0], cg[0], cb[0])


def float2str_min(x):
    """ float to string for color output """
    sign = '+' if x >= 0.0 else '-'
    s = "{:.3f}".format(abs(x))  # 3-4 decimal places
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
            a, b, c, d, e = [float2str_min(t) for t in coes[i]]
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

    code = generate_code(color_pics)

    open("cosine.js", "wb").write(bytearray(code['js_code'], 'utf-8'))
    open("cosine.h", "wb").write(bytearray(code['cpp_code'], 'utf-8'))
