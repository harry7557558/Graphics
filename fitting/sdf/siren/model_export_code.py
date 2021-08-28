import numpy as np


def num2str(x, signed=False, d=3):
    s = "{:.{prec}f}".format(x, prec=d)
    while s[0] == '0':
        s = s[1:]
    while s[0] == '-' and s[1] == '0':
        s = '-' + s[2:]
    while len(s) > 0 and s[-1] in ['0', '.']:
        s = s[0:len(s)-1]
    if s in ['', '-']:
        s = '0'
    if signed and s[0] != '-':
        s = '+' + s
    return s


def arr_to_str_1d(arr, lbra='[', rbra=']', d=3):
    s = []
    for i in range(len(arr)):
        s.append(num2str(arr[i], d=d))
    return lbra + ",".join(s) + rbra


def arr_to_str_2d(arr, lbra='[', rbra=']', d=3):
    s = []
    for i in range(len(arr)):
        s.append(arr_to_str_1d(arr[i], lbra, rbra, d))
    return lbra + ",".join(s) + rbra


def vec2str(v, d=3):
    return f'vec{len(v)}(' + ','.join([num2str(x, d=d) for x in v]) + ')'


def mat4str(m, d=3):
    v = [m[0][0], m[0][1], m[0][2], m[0][3],
         m[1][0], m[1][1], m[1][2], m[1][3],
         m[2][0], m[2][1], m[2][2], m[2][3],
         m[3][0], m[3][1], m[3][2], m[3][3]]
    return 'mat4(' + ','.join([num2str(x, d=d) for x in v]) + ')'


def export_glsl(weights):
    code = ""

    for l in range(len(weights)):
        w, b = weights[l]
        if l == 0:
            for ii in range(4):
                sx = 'p.x*' + vec2str(w[0][4*ii:4*ii+4])
                sy = 'p.y*' + vec2str(w[1][4*ii:4*ii+4])
                sz = 'p.z*' + vec2str(w[2][4*ii:4*ii+4])
                sb = vec2str(b[4*ii:4*ii+4])
                s = 'sin(' + '+'.join([sx, sy, sz, sb]) + ')'
                code += f'vec4 f{l}{ii}={s};\n'
        elif l+1 < len(weights):
            w = np.array(w)
            for ii in range(4):
                s0 = mat4str(w[0:4, 4*ii:4*ii+4]) + f'*f{l-1}0'
                s1 = mat4str(w[4:8, 4*ii:4*ii+4]) + f'*f{l-1}1'
                s2 = mat4str(w[8:12, 4*ii:4*ii+4]) + f'*f{l-1}2'
                s3 = mat4str(w[12:16, 4*ii:4*ii+4]) + f'*f{l-1}3'
                sb = vec2str(b[4*ii:4*ii+4])
                s = 'sin(' + '\n    +'.join([s0, s1, s2, s3, sb]) + ')'
                code += f'vec4 f{l}{ii}={s};\n'
        else:
            w = np.array(w).reshape((16,))
            s0 = 'dot(' + vec2str(w[0:4]) + f',f{l-1}0)'
            s1 = 'dot(' + vec2str(w[4:8]) + f',f{l-1}1)'
            s2 = 'dot(' + vec2str(w[8:12]) + f',f{l-1}2)'
            s3 = 'dot(' + vec2str(w[12:16]) + f',f{l-1}3)'
            s = '+'.join([s0, s1, s2, s3]) + num2str(b[0], signed=True)
            code += f'return {s};\n'

    #print(code)
    return code
