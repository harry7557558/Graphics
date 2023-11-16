import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from state import State


def update_euler(state: State, h: float):
    dpdt = state.calc_dpdt_pv()
    for i in range(state.n):
        state.points[i] += dpdt[i] * h


def update_implicit_euler(state: State, h: float):
    if h == 0.0:
        return
    ddpdtdp = state.calc_ddpdtdp()
    mat = scipy.sparse.identity(2*state.n) / h - ddpdtdp
    dpdt = state.calc_dpdt_arr()
    dp, dp_info = scipy.sparse.linalg.bicg(mat, dpdt,
                                           # maxiter=2*state.n, atol=0.0,
                                           x0=np.zeros(2*state.n))
    state.set_points_arr(state.get_points_arr() + dp)


def update_implicit_euler_pv(state: State, h: float):
    if h == 0.0:
        return
    ddpdtdp = state.calc_ddpdtdp_pv()
    mat = scipy.sparse.identity(2*state.n) / h - ddpdtdp
    dpdt = state.calc_dpdt_pv_arr()
    dp, dp_info = scipy.sparse.linalg.bicg(mat, dpdt,
                                           # maxiter=2*state.n, atol=0.0,
                                           x0=np.zeros(2*state.n))
    state.set_points_arr(state.get_points_arr() + dp)
