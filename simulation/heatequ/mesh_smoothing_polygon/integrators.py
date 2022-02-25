import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from state import State


def update_euler(state: State, h: float):
    dpdt = state.calc_dpdt_pv()
    for i in range(state.n):
        state.points[i] += dpdt[i] * h


def update_implicit_euler(state: State, h: float):
    ddpdtdp = state.calc_ddpdtdp_pv()
    #state.visualize_matrix(ddpdtdp, "D:\\sparse.png")
    #state.visualize_matrix(state.calc_ddpdtdp_pv_n()-ddpdtdp, "D:\\sparse.png")
    mat = scipy.sparse.identity(2*state.n) / h - ddpdtdp
    dpdt = state.calc_dpdt_pv_arr()
    dp, dp_info = scipy.sparse.linalg.bicg(mat, dpdt,
                                           x0=np.zeros(2*state.n))
    state.set_points_arr(state.get_points_arr() + dp)
