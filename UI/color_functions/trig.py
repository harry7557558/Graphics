# fit color images using this series:
# c₀ + c₁⋅x + a₀⋅cos(π⋅x-u₀) + ∑ₖ[aₖ⋅cos(2kπ⋅x-uₖ)]

import numpy as np
from math import sin, cos, pi, sqrt, hypot, atan2
from img.load_data import load_image


def fit_array_deg(x, y, deg):
    """ ∑ᵢ(uᵢ⋅uᵢᵀ) wₖ = ∑ᵢ(fᵢ⋅uᵢ) """
    dim = 2*deg+4

    def generate_u(x):
        u = [1, x, cos(0.5*x), sin(0.5*x)]
        for k in range(1, deg+1):
            u.append(cos(k*x))
            u.append(sin(k*x))
        return np.array(u)

    cov = np.zeros((dim, dim))
    vec = np.zeros((dim))
    for i in range(len(x)):
        ui = generate_u(x[i])
        cov += np.tensordot(ui, ui, axes=0)
        vec += y[i] * ui

    sol = np.linalg.solve(cov, vec)

    err = sol[0]*x**0 + sol[1]*x + sol[2]*np.cos(0.5*x) + sol[3]*np.sin(0.5*x)
    for k in range(1, deg+1):
        err += sol[2*k+2]*np.cos(k*x) + sol[2*k+3]*np.sin(k*x)
    loss = sqrt(sum((err-y)**2) / len(x))

    ans = [sol[0], sol[1]]
    for k in range(0, deg+1):
        # convert a*cos(k*x)+b*sin(k*x) to a*cos(kx+b)
        a, b = sol[2*k+2], sol[2*k+3]
        a, b = hypot(a, b), -atan2(b, a)
        ans += [a, b]

    return (ans, loss)


def fit_array(x, y, req_loss=4.0/255.0, max_deg=2):
    for deg in range(0, max_deg+1):
        coes, loss = fit_array_deg(x, y, deg)
        if loss < req_loss:
            break
    print("deg={};".format(deg), "loss = {:.1f} / 255".format(255*loss))
    return coes


def fit_color(cols):
    w = len(cols)
    x = np.array([2.0*pi*(i+.5)/w for i in range(w)])
    cr = fit_array(x, np.array([c[0] for c in cols]))
    cg = fit_array(x, np.array([c[1] for c in cols]))
    cb = fit_array(x, np.array([c[2] for c in cols]))
    return [cr, cg, cb]


def float2str_min(x, mul=1.0):
    """ float to string for color output """
    if type(x) is list:
        x = [float2str_min(xi, mul).lstrip('+').rstrip('.')
             for xi in x]
        return '+vec3(' + ','.join(x) + ')'
    x *= mul
    sign = '+' if x >= 0.0 else '-'
    s = "{:.2f}".format(abs(x))
    while s[-1] == '0':
        s = s[:len(s)-1]
    while s[0] == '0':
        s = s[1:]
    if s == '.':
        s = '0.'
    return sign + s


def encode_series(c):
    deg = (len(c)-2)//2-1
    s = f"{float2str_min(c[0])}{float2str_min(c[1],2.0*pi)}*x"
    for k in range(0, deg+1):
        s += f"{float2str_min(c[2*k+2])}*" + \
            f"cos({float2str_min(max(k, 0.5),2.0*pi).lstrip('+')}*x" + \
            f"{float2str_min(c[2*k+3])})"
    return s.lstrip('+')


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

    glsl_code = """"""

    for (colorname, data) in pics:
        print(colorname)
        coes = fit_color(data)
        # generate JS and C++ code
        js_code += "  " + colorname + ": function(x) {\n"
        cpp_code += "  static vec3 " + colorname + "(Float x) {\n"
        for c in range(3):
            st = encode_series(coes[c])
            js_code += "    var " + 'rgb'[c] + " = " + \
                st.replace("cos", "Math.cos") + ';\n'
            cpp_code += "    Float " + 'rgb'[c] + " = " + \
                st + ';\n'
        js_code += "    return this.tocol(r, g, b);\n  },\n"
        cpp_code += "    return vec3(clp(r),clp(g),clp(b));\n  }\n"
        # generate GLSL code
        glsl_code += "vec3 " + colorname + "T(float x) {\n"
        l = max([len(c) for c in coes])
        for c in range(3):
            coes[c] += [0.0] * (l-len(coes[c]))
        coes = np.array(coes).T.tolist()
        st = encode_series(coes)
        glsl_code += "    return clamp(" + st + ",0.,1.);\n}\n"

    js_code += "};"
    cpp_code += "};\n"

    return {
        "js_code": js_code,
        "cpp_code": cpp_code,
        "glsl_code": glsl_code
    }


if __name__ == "__main__":

    names = open("img/index", "r").read().split('\n')
    color_pics = [(names[i], load_image("img/"+str(i+1)+".png"))
                  for i in range(50)]

    code = generate_code(color_pics)

    open("trig.js", "wb").write(bytearray(code['js_code'], 'utf-8'))
    open("trig.h", "wb").write(bytearray(code['cpp_code'], 'utf-8'))
    open("trig.glsl", "wb").write(bytearray(code['glsl_code'], 'utf-8'))

    import js_generator
