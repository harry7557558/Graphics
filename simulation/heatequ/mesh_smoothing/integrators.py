import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from state import State
from pygame import Vector2, Vector3


def euler(state: State, h: float):
    dpdt = state.calc_dpdt()
    for i in range(state.n):
        state.vertices[i] += dpdt[i] * h
    state.recompute_draw()


def ieuler(state: State, h: float, pv: bool):
    if h == 0.0:
        return
    dpdu = state.calc_dpdu(pv)
    mat = scipy.sparse.identity(state.n) / h - dpdu
    dpdt = state.calc_dpdt(pv)

    # state.visualize_matrix(mat, "D:\\sparse-matrix.png")
    # print(sorted(abs(np.linalg.eig(dpdu.toarray())[0])))

    # explore the properties
    if 0:
        m = np.identity(state.n)+np.linalg.inv(mat.toarray())@dpdu
        eigvals, eigvecs = np.linalg.eig(m)
        eigvals_sorted = sorted(eigvals.real)
        print(eigvals_sorted)
        assert eigvals_sorted[0] >= 0.0
        assert abs(eigvals_sorted[-1]-1.0) < 1e-6
        for i in range(state.n):
            if abs(eigvals[i].real-1.0) < 1e-6:
                eigval = eigvals[i]
                eigvec = 10000*state.n**0.5*eigvecs[:, i]
                print(eigval)
                print(','.join(map(str, map(int, eigvec))))

    dxdt = np.array([p.x for p in dpdt])
    dydt = np.array([p.y for p in dpdt])
    dzdt = np.array([p.z for p in dpdt])
    p0 = np.zeros((state.n))  # matrix may be singular
    tol = 1e-6  # matters when h is large
    dx, dx_info = scipy.sparse.linalg.bicg(mat, dxdt, x0=p0, tol=tol, atol=tol)
    dy, dy_info = scipy.sparse.linalg.bicg(mat, dydt, x0=p0, tol=tol, atol=tol)
    dz, dz_info = scipy.sparse.linalg.bicg(mat, dzdt, x0=p0, tol=tol, atol=tol)

    for i in range(state.n):
        state.vertices[i] += Vector3(dx[i], dy[i], dz[i])
    state.recompute_draw()
