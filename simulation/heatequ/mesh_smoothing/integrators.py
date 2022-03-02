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

    dxdt = np.array([p.x for p in dpdt])
    dydt = np.array([p.y for p in dpdt])
    dzdt = np.array([p.z for p in dpdt])
    dx, dx_info = scipy.sparse.linalg.bicg(mat, dxdt)
    dy, dy_info = scipy.sparse.linalg.bicg(mat, dydt)
    dz, dz_info = scipy.sparse.linalg.bicg(mat, dzdt)

    for i in range(state.n):
        state.vertices[i] += Vector3(dx[i], dy[i], dz[i])
    state.recompute_draw()
