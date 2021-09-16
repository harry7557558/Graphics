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
