import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from state import State


def euler(state: State, h: float):
    dudt = state.calc_dudt()
    state.u += dudt * h


def ieuler(state: State, h: float):
    if h == 0.0:
        return
    dpdu = state.calc_dpdu()
    mat = scipy.sparse.identity(state.n) / h - dpdu
    dudt = state.calc_dudt().reshape((state.n, 1))
    du, du_info = scipy.sparse.linalg.cg(mat, dudt)
    du = du.reshape((state.xn, state.yn))
    state.u += du
