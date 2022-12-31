# Figure out what's wrong with quadratic tetrahedral elements
# (Doesn't converge to the hand calculation solution)

import numpy as np
from sympy import *

t1, t2, t3 = symbols('t1 t2 t3')
t0 = 1 - (t1+t2+t3)
T = [t0*(2*t0-1), t1*(2*t1-1), t2*(2*t2-1), t3*(2*t3-1),
     4*t0*t1, 4*t0*t2, 4*t0*t3, 4*t1*t2, 4*t1*t3, 4*t2*t3]
W = []
for t in [t1, t2, t3]:
    W.append([diff(ti, t) for ti in T])

GL = [
    (0, 0, 0, 1/60),
    (1, 0, 0, 1/60),
    (0, 1, 0, 1/60),
    (0, 0, 1, 1/60),
    (0.5, 0, 0, 1/15),
    (0.5, 0.5, 0, 1/15),
    (0, 0.5, 0, 1/15),
    (0, 0, 0.5, 1/15),
    (0.5, 0, 0.5, 1/15),
    (0, 0.5, 0.5, 1/15),
    (0.25, 0.25, 0.25, 8/15)
]
weights = np.array([p[3] for p in GL])
assert abs(sum(weights)-1) < 1e-12
print('{'+','.join([str(p[3]) for p in GL])+'}')

# generate weight table
W1 = []
for t1, t2, t3, w in GL:
    s = []
    W1.append([])
    for wi in W:
        wi = [float(_.subs({'t1': t1, 't2': t2, 't3': t3})) for _ in wi]
        assert abs(sum(wi)) < 1e-12
        s.append('{'+','.join(["{:.12g}".format(_) for _ in wi])+'}')
        W1[-1].append(wi)
    print('{'+', '.join(s)+'},')
W = np.array(W1)

# compare with linear
p0 = list(symbols('p0 p1 p2 p3'))
x0 = [p0[1]-p0[0], p0[2]-p0[0], p0[3]-p0[0]]  # rows
def l2q(p0):
    p0 = [np.array(pi) for pi in p0]
    return np.array([
        p0[0], p0[1], p0[2], p0[3],
        0.5*(p0[0]+p0[1]), 0.5*(p0[0]+p0[2]), 0.5*(p0[0]+p0[3]),
        0.5*(p0[1]+p0[2]), 0.5*(p0[1]+p0[3]), 0.5*(p0[2]+p0[3])
    ])
p1 = l2q(p0)
x1 = np.dot(W, p1)
for x in x1:
    for xi, x0i in zip(x, x0):
        assert xi.equals(x0i)

# compare energy
invx0 = np.array(symbols('i00 i01 i02 i10 i11 i12 i20 i21 i22')).reshape((3, 3))
S = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0]
])
nu = .375  # a number that doesn't break floating point
C = np.array([
    [1-nu, nu, nu, 0, 0, 0],
    [nu, 1-nu, nu, 0, 0, 0],
    [nu, nu, 1-nu, 0, 0, 0],
    [0, 0, 0, .5-nu, 0, 0],
    [0, 0, 0, 0, .5-nu, 0],
    [0, 0, 0, 0, 0, .5-nu]
])
W0 = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1]])
dedx0 = np.dot(S, np.kron(np.dot(invx0, W0), np.eye(3)))
K0 = dedx0.T@C@dedx0
u0 = np.array([(1, 2, 3), (4, 5, 6), (-2, -1, 0), (-5, 4, -3)])
x0 = u0.reshape(12)
E0 = expand(x0.T@K0@x0)
print(E0)
wxw = [np.dot(invx0, wi) for wi in W]
wxw = [np.kron(wi, np.eye(3)) for wi in wxw]
dedx = [np.dot(S, wi) for wi in wxw]
K = [dedxi.T@C@dedxi for dedxi in dedx]
x = l2q(u0).reshape(30)
E = [expand(x.T@Ki@x) for Ki in K]
E = np.dot(weights, E)
print(E)
assert(E.equals(E0))
