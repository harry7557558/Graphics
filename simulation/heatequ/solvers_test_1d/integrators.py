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

        dudt = self.current_state.calc_dudt()
        self.current_state.u += dudt * dt
        self.current_state.time += dt


class Midpoint(Base):

    def __init__(self, state: State):
        super().__init__(state, 2, 0)

    def update(self, dt: float):

        t0 = self.current_state.time
        u0 = np.array(self.current_state.u)
        p0 = self.current_state.calc_dudt()

        self.current_state.u = u0 + 0.5*p0*dt
        self.current_state.time = t0 + 0.5*dt
        p1 = self.current_state.calc_dudt()

        self.current_state.u = u0 + p1*dt
        self.current_state.time = t0 + dt


class RungeKutta(Base):

    def __init__(self, state: State):
        super().__init__(state, 4, 0)

    def update(self, dt: float):

        t0 = self.current_state.time
        u0 = np.array(self.current_state.u)

        k1 = self.current_state.calc_dudt()

        self.current_state.u = u0+0.5*dt*k1
        self.current_state.time = t0 + 0.5*dt
        k2 = self.current_state.calc_dudt()

        self.current_state.u = u0+0.5*dt*k2
        self.current_state.time = t0 + 0.5*dt
        k3 = self.current_state.calc_dudt()

        self.current_state.u = u0+dt*k3
        self.current_state.time = t0 + dt
        k4 = self.current_state.calc_dudt()

        self.current_state.u = u0 + dt/6. * (k1+2.0*k2+2.0*k3+k4)
        self.current_state.time = t0 + dt


class ImplicitEuler(Base):
    """Without considering time variable"""

    def __init__(self, state: State):
        # isn't inaccurate when put 1 as EVAL_COUNT
        super().__init__(state, 4, 0)

    def update(self, dt: float):
        #dpdu = self.current_state.calc_dpdu_n()
        dpdu = self.current_state.calc_dpdu()
        #mat = np.identity(self.current_state.n) / dt - dpdu
        mat = scipy.sparse.identity(self.current_state.n) / dt - dpdu
        dudt = self.current_state.calc_dudt()
        #du = np.linalg.solve(mat, dudt)
        du, du_info = scipy.sparse.linalg.cg(mat, dudt)
        self.current_state.u += du
        self.current_state.time += dt
