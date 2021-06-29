"""
    Part of the Random Egg Generator
    Analyze traced real-world eggs to find the probability distribution of egg shapes
    Implemented two fitting models, the later one works better
    One may do better than this...
"""


from xml.dom import minidom
from svgpathtools import parse_path, CubicBezier

import numpy as np
import math
import random
from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt


def spline_to_samples(curve, n):
    """ convert a cubic Bezier curve to a list of samples """
    t = (np.arange(0, n) + 0.5) / n
    p = np.dot(curve, [(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3])
    return p


def plot_spline(curves):
    """ plot a cubic spline represented by a complex numpy array """
    plt.clf()
    ax = plt.figure().add_subplot(111)
    ax.set_aspect('equal')
    for curve in curves:
        t = np.linspace(0.0, 1.0, 100)
        p = np.dot(curve, [(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3])
        ax.plot(p.real, p.imag)
    plt.show()


def intersect_polygon(poly, rd):
    """ shoot a ray rd from the origin,
        calculate the minimum parameter t for rd*t to hit the polygon """
    min_t = float('inf')
    for i in range(len(poly)):
        p = poly[i-1]
        q = poly[i]
        t, u = np.linalg.solve(
            [[rd.real, (p-q).real], [rd.imag, (p-q).imag]],
            [p.real, p.imag])
        if 0 <= u < 1 and t > 0:
            min_t = min(min_t, t)
    return min_t


def path_to_polar_samples(path_d, sample=64):
    """
        receive an SVG path string, parse and normalize it, and
        return a list of distances to the center of the curve from top to bottom
    """
    # all CCW cubic Bezier curves
    parsed = parse_path(path_d)
    curves = []
    for curve in parsed:
        if type(curve) != CubicBezier:
            raise BaseError("Not a cubic Bezier curve")
        curves.append(np.array([curve.start, curve.control1, curve.control2, curve.end]))
    curves = np.array(curves).conj()

    # normalize curve
    # this can be done analytically but I'm lazy
    points = np.concatenate([spline_to_samples(curve, 20) for curve in curves])
    y_max = np.max(points.imag)
    y_min = np.min(points.imag)
    sc = 2.0 / (y_max - y_min)
    tr = np.average(points.real) + 0.5j*(y_max+y_min)
    points = sc * (points - tr)
    curves = sc * (curves - tr)

    # get samples
    angles = math.pi * ((np.arange(0, sample) + 0.5) / sample)
    dists = []
    for a in angles:
        rd = math.sin(a) + math.cos(a)*1j
        t1 = intersect_polygon(points, rd)
        rd = -math.sin(a) + math.cos(a)*1j
        t2 = intersect_polygon(points, rd)
        dists.append(0.5*(t1+t2))

    return dists


def plot_angles(path_strings):
    SAMPLE = 32
    angles = (np.arange(0, SAMPLE) + 0.5) / SAMPLE
    
    EGG_COUNT = len(path_strings)
    for i in range(EGG_COUNT):
        print(f'{i}/{EGG_COUNT}')
        dists = path_to_polar_samples(path_strings[i], SAMPLE)
        plt.plot(angles, dists)
    plt.show()


def plot_eggs(path_strings):
    SAMPLE = 32
    angles = math.pi * (np.arange(0, SAMPLE) + 0.5) / SAMPLE
    EGG_COUNT = len(path_strings)

    figsize = (4, 6)
    fig = plt.figure(figsize=(9, 6), dpi=80)

    for i in range(EGG_COUNT):
        print(f'{i}/{EGG_COUNT}')
        dists = path_to_polar_samples(path_strings[i], SAMPLE)
        ax = fig.add_subplot(*figsize, i+1)
        plt.plot(dists*np.sin(angles), dists*np.cos(angles), 'b')
        plt.plot(dists*-np.sin(angles), dists*np.cos(angles), 'b')
        ax.set_aspect('equal')

    plt.show()


""" Egg Equation Models """


def fit_egg_1(x, y, f0, f1, f2):
    """ find a0, a1, a2; curve of best fit
        1-sin(x)²*(a0*cos(f0*x)+a1*cos(f1*x)+a2*cos(f2*x)) """
    def func(x, a0, a1, a2):
        return 1.0 - np.sin(x)**2 * (a0*np.cos(f0*x)+a1*np.cos(f1*x)+a2*np.cos(f2*x))
    popt, pcov = curve_fit(func, x, y)  # quadratic form, should success
    a0, a1, a2 = popt
    err = math.sqrt(np.average((y - func(x, a0, a1, a2))**2))
    return ([a0, a1, a2], err)


def test_fit_egg_1(path_strings):
    SAMPLE = 32
    angles = math.pi * (np.arange(0, SAMPLE) + 0.5) / SAMPLE
    coes = []
    errs = []
    for path_string in path_strings:
        dists = path_to_polar_samples(path_string, SAMPLE)
        coe, err = fit_egg_1(angles, dists, 0.0, 1.0, 2.0)
        coes.append(coe)
        errs.append(err)
    coes = np.array(coes)
    average_err = np.exp(np.average(np.log(err)))
    print("average error:", average_err)  # 0.0072

    ax = plt.figure().add_subplot(projection='3d', proj_type='ortho')
    ax.scatter(coes[:, 0], coes[:, 1], coes[:, 2], c=errs)

    p1 = np.array([0.45, 0.03, 0.11])  # thin, even
    p2 = np.array([0.23, 0.05, 0.03])  # fat
    p3 = np.array([0.42, 0.11, 0.09])  # thin, uneven
    trig = Poly3DCollection([[p1, p2, p3]], alpha=0.25, facecolor='#800000')
    plt.gca().add_collection3d(trig)
    
    plt.show()


def fit_egg_2(x, y):
    """ find a0, a1, a2; curve of best fit
        1-sin(x)²*(a0*cos(f0*x)+a1*cos(f1*x)+a2*cos(f2*x)) """
    def func(x, a0, a1, a2):
        return 1.0 - np.sin(x)**2 * (a0 + a1*np.exp(-x) + a2*np.cos(x))
    popt, pcov = curve_fit(func, x, y)  # quadratic form, should success
    a0, a1, a2 = popt
    err = math.sqrt(np.average((y - func(x, a0, a1, a2))**2))
    return ([a0, a1, a2], err)


def test_fit_egg_2(path_strings):
    SAMPLE = 32
    angles = math.pi * (np.arange(0, SAMPLE) + 0.5) / SAMPLE
    coes = []
    errs = []
    for path_string in path_strings:
        dists = path_to_polar_samples(path_string, SAMPLE)
        coe, err = fit_egg_2(angles, dists)
        coes.append(coe)
        errs.append(err)
    coes = np.array(coes)
    average_err = np.exp(np.average(np.log(err)))
    print("average error:", average_err)  # 0.0011

    mu = np.average(coes, axis=0)
    cov = np.cov((coes-mu).T)
    covvar, covdir = np.linalg.eig(cov)
    sigma = np.sqrt(covvar)
    covdir = covdir.T

    ax = plt.figure().add_subplot(projection='3d', proj_type='ortho')
    ax.scatter(coes[:, 0], coes[:, 1], coes[:, 2], c=errs)

    print('mu =', mu.tolist())
    for i in range(3):
        d = 2.0 * sigma[i]*covdir[i]
        v1, v2 = mu+d, mu-d
        ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]])
        print(d.tolist())
    
    plt.show()


""" Final """


def plot_random_eggs_1():
    def random_coefficients():
        p1 = np.array([0.45, 0.03, 0.11])  # thin, even
        p2 = np.array([0.23, 0.05, 0.03])  # fat
        p3 = np.array([0.42, 0.11, 0.09])  # thin, uneven
        u = random.random()
        v = random.random()
        if u+v > 1.0:
            u, v = 1.0-u, 1.0-v
        p = p1 + (p2-p1)*u + (p3-p1)*v
        return p

    def egg_equ(t, a0, a1, a2):
        t0 = t
        t = 2.0*np.arcsin(np.sin(0.5*t))  # piecewise reflection
        r = 1.0 - np.sin(t)**2 * (a0 + a1*np.cos(t) + a2*np.cos(2*t))
        return np.transpose(np.vstack((r*np.sin(t0), r*np.cos(t0))))

    def random_egg():
        a0, a1, a2 = random_coefficients()
        t = np.linspace(0.0, 2.0*math.pi, 100)
        return egg_equ(t, a0, a1, a2)
    
    figsize = (4, 6)
    fig = plt.figure(figsize=(9, 6), dpi=80)
    EGG_COUNT = np.prod(figsize)

    for i in range(EGG_COUNT):
        points = random_egg()
        ax = fig.add_subplot(*figsize, i+1)
        plt.plot(points[:,0], points[:,1], 'b')
        ax.set_aspect('equal')

    plt.show()


def plot_random_eggs_2():
    def random_coefficients():
        u = 2.0*math.pi * random.random()
        v = 2.0*random.random()-1.0
        w = random.random() ** (1.0/3.0)
        x = w * math.sqrt(1.0-v*v) * math.cos(u)
        y = w * math.sqrt(1.0-v*v) * math.sin(u)
        z = w * v
        mu = np.array([0.05048232509261819, 1.1917245785944626, -0.2113612433670313])
        p = x * np.array([-0.11211907456861894, 0.8662805303219393, -0.16592978179513765]) \
            + y * np.array([0.05143550009089445, 0.0009434329853603692, -0.02982962425271538]) \
            + z * np.array([0.021209126749381683, 0.00980936323090463, 0.03688134011888825])
        return mu + p

    def egg_equ(t, a0, a1, a2):
        t0 = t
        t = 2.0*np.arcsin(np.sin(0.5*t))  # piecewise reflection
        r = 1.0 - np.sin(t)**2 * (a0 + a1*np.exp(-t) + a2*np.cos(t))
        return np.transpose(np.vstack((r*np.sin(t0), r*np.cos(t0))))

    def random_egg():
        a0, a1, a2 = random_coefficients()
        t = np.linspace(0.0, 2.0*math.pi, 100)
        return egg_equ(t, a0, a1, a2)
    
    figsize = (4, 6)
    fig = plt.figure(figsize=(9, 6), dpi=80)
    EGG_COUNT = np.prod(figsize)

    for i in range(EGG_COUNT):
        points = random_egg()
        ax = fig.add_subplot(*figsize, i+1)
        plt.plot(points[:,0], points[:,1], 'b')
        ax.set_aspect('equal')

    plt.show()


if __name__ == "__main__":

    svgdata = minidom.parse("eggs_traced.svg")
    path_strings = [path.getAttribute('d') for path
                    in svgdata.getElementsByTagName('path')]

    #plot_angles(path_strings)
    #plot_eggs(path_strings)

    #test_fit_egg_2(path_strings[:])

    plot_random_eggs_2()
    
