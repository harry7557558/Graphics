import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from state import State
from copy import deepcopy


class Base:

    def __init__(self, state: State, EVAL_COUNT: int, CACHE_COUNT: int):
        self.current_state = state
        self.prev_states = []
        self.EVAL_COUNT = EVAL_COUNT  # derivative evaluations per step
        self.CACHE_COUNT = CACHE_COUNT  # min number of prev_states

    def push_state(self) -> None:
        """add the current state to the queue"""
        self.prev_states.append(deepcopy(self.current_state))
        more = len(self.prev_states)-self.CACHE_COUNT
        self.prev_states = self.prev_states[max(more, 0):]

    def update(self, dt: float):
        pass


class Euler(Base):

    def __init__(self, state: State):
        super().__init__(state, 1, 0)

    def update(self, dt: float):

        dhdt, dudt, dvdt = self.current_state.calc_dsdt()
        self.current_state.h += dhdt * dt
        self.current_state.u += dudt * dt
        self.current_state.v += dvdt * dt
        self.current_state.time += dt


class Midpoint(Base):

    def __init__(self, state: State):
        super().__init__(state, 2, 0)

    def update(self, dt: float):

        t0 = self.current_state.time
        h0 = np.array(self.current_state.h)
        u0 = np.array(self.current_state.u)
        v0 = np.array(self.current_state.v)
        ht0, ut0, vt0 = self.current_state.calc_dsdt()

        self.current_state.h = h0 + 0.5*ht0*dt
        self.current_state.u = u0 + 0.5*ut0*dt
        self.current_state.v = v0 + 0.5*vt0*dt
        self.current_state.time = t0 + 0.5*dt
        ht1, ut1, vt1 = self.current_state.calc_dsdt()

        self.current_state.h = h0 + ht1*dt
        self.current_state.u = u0 + ut1*dt
        self.current_state.v = v0 + vt1*dt
        self.current_state.time = t0 + dt


class RungeKutta(Base):

    def __init__(self, state: State):
        super().__init__(state, 4, 0)

    def update(self, dt: float):

        t0 = self.current_state.time
        h0 = np.array(self.current_state.h)
        u0 = np.array(self.current_state.u)
        v0 = np.array(self.current_state.v)
        ht1, ut1, vt1 = self.current_state.calc_dsdt()

        self.current_state.h = h0 + 0.5*ht1*dt
        self.current_state.u = u0 + 0.5*ut1*dt
        self.current_state.v = v0 + 0.5*vt1*dt
        self.current_state.time = t0 + 0.5*dt
        ht2, ut2, vt2 = self.current_state.calc_dsdt()

        self.current_state.h = h0 + 0.5*ht2*dt
        self.current_state.u = u0 + 0.5*ut2*dt
        self.current_state.v = v0 + 0.5*vt2*dt
        self.current_state.time = t0 + 0.5*dt
        ht3, ut3, vt3 = self.current_state.calc_dsdt()

        self.current_state.h = h0 + ht3*dt
        self.current_state.u = u0 + ut3*dt
        self.current_state.v = v0 + vt3*dt
        self.current_state.time = t0 + dt
        ht4, ut4, vt4 = self.current_state.calc_dsdt()

        self.current_state.h = h0 + dt/6. * (ht1+2.0*ht2+2.0*ht3+ht4)
        self.current_state.u = u0 + dt/6. * (ut1+2.0*ut2+2.0*ut3+ut4)
        self.current_state.v = v0 + dt/6. * (vt1+2.0*vt2+2.0*vt3+vt4)
        self.current_state.time = t0 + dt


class ImplicitEuler(Base):
    """Without considering time variable
       (Seems like this method loses energy for this type of equations)"""

    def __init__(self, state: State):
        super().__init__(state, 4, 0)

    def update(self, dt: float):
        dstds = self.current_state.calc_dstds()
        #State.visualize_matrix(dstds, "D:\\sparse-matrix.png")
        #mat = np.identity(self.current_state.n) / dt - dpdu
        mat = scipy.sparse.identity(3*self.current_state.n) / dt - dstds
        dsdt = self.current_state.calc_dsdt()
        dsdt = np.array(dsdt).reshape((3*self.current_state.n))
        ds, ds_info = scipy.sparse.linalg.bicg(mat, dsdt)
        ds = ds.reshape((3, self.current_state.xn, self.current_state.yn))
        self.current_state.h += ds[0]
        self.current_state.u += ds[1]
        self.current_state.v += ds[2]
        self.current_state.time += dt
