# get color function from color images (polynomial regression)

import numpy as np
from math import sqrt
from img.load_data import load_image


def fit_color(vals, maxdeg):
    # polynomial fitting
    w = len(vals)
    x = np.array([(i+.5)/w for i in range(w)])
    vals = np.array(vals).transpose()
    coes = []
    for v in vals:
        for deg in range(2, maxdeg+1):
            coes_t = np.polyfit(x, v, deg, full=True)
            loss = sqrt(coes_t[1][0] / w)
            if loss < 4.0 / 255:
                break
        print("deg={};".format(deg),
              "loss = {:.1f} / 255".format(255*coes_t[1][0]))
        coes.append(list(coes_t[0]))
    return coes


def float2str_min(x):
    """ float to string for color output """
    if type(x) is list:
        x = [float2str_min(xi).lstrip('+').rstrip('.')
             for xi in x]
        return '+vec3(' + ','.join(x) + ')'
    sign = '+' if x >= 0.0 else '-'
    s = "{:.2f}".format(abs(x))
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
  // calculations are done in double precision
  static Float clp(double x) {
    return (Float)(x<0.?0.:x>1.?1.:x);
  }
public:
"""

    glsl_code = """"""

    for (colorname, data) in pics:
        print(colorname)
        coes = fit_color(data, maxdeg=4)
        # generate JS and C++ code
        js_code += "  " + colorname + ": function(t) {\n"
        cpp_code += "  static vec3 " + colorname + "(double t) {\n"
        for c in range(3):
            deg = len(coes[c])-1
            st = '('*(deg-1) + ')*t'.join([float2str_min(coes[c][i])
                                           for i in range(deg+1)]).lstrip('+').replace(')', '', 1)
            js_code += "    var " + 'rgb'[c] + " = " + st + ';\n'
            cpp_code += "    double " + 'rgb'[c] + " = " + \
                st + ';\n'   # double precision required
        js_code += "    return this.tocol(r, g, b);\n  },\n"
        cpp_code += "    return vec3(clp(r),clp(g),clp(b));\n  }\n"
        # generate GLSL code
        glsl_code += "vec3 " + colorname + "P(float t) {\n"
        deg = max([len(c) for c in coes])-1
        for c in range(3):
            coes[c] = [0.0] * (deg+1-len(coes[c])) + coes[c]
        coes = np.array(coes).T.tolist()
        st = '('*(deg-1) + ')*t'.join([float2str_min(coes[i])
                                       for i in range(deg+1)]).lstrip('+').replace(')', '', 1)
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

    open("poly.js", "wb").write(bytearray(code['js_code'], 'utf-8'))
    open("poly.h", "wb").write(bytearray(code['cpp_code'], 'utf-8'))
    open("poly.glsl", "wb").write(bytearray(code['glsl_code'], 'utf-8'))

    import js_generator
