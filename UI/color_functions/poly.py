# get color function from color images (polynomial regression)

import numpy as np
from math import sqrt
from img.load_data import load_image


def fit_color(vals):
    # polynomial fitting
    w = len(vals)
    x = np.array([(i+.5)/w for i in range(w)])
    vals = np.array(vals).transpose()
    coes = []
    for v in vals:
        for deg in range(3, 10+1):
            coes_t = np.polyfit(x, v, deg, full=True)
            loss = sqrt(coes_t[1][0] / w)
            if loss < 2.0 / 255:
                break
        print("deg={};".format(deg),
              "loss = {:.1f} / 255".format(255*coes_t[1][0]))
        coes.append(list(coes_t[0]))
    return coes


def float2str_min(x):
    """ float to string for color output """
    sign = '+' if x >= 0.0 else '-'
    s = "{:.4f}".format(abs(x))  # 3-4 decimal places
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

    for (colorname, data) in pics:
        print(colorname)
        coes = fit_color(data)
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

    js_code += "};"
    cpp_code += "};\n"

    return {"js_code": js_code, "cpp_code": cpp_code}


if __name__ == "__main__":

    names = open("img/index", "r").read().split('\n')
    color_pics = [(names[i], load_image("img/"+str(i+1)+".png"))
                  for i in range(50)]

    code = generate_code(color_pics)

    open("poly.js", "wb").write(bytearray(code['js_code'], 'utf-8'))
    open("poly.h", "wb").write(bytearray(code['cpp_code'], 'utf-8'))
