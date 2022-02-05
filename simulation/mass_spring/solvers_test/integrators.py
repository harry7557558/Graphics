import numpy as np
from state import Vector2, Mass, Spring, State
from copy import deepcopy


class Base:

    def __init__(self, state: State, EVAL_COUNT: int, CACHE_COUNT: int):
        self.current_state = state
        self.prev_states = []
        self.EVAL_COUNT = EVAL_COUNT  # derivative evaluations per step
        self.CACHE_COUNT = CACHE_COUNT  # min number of prev_states

    @staticmethod
    def state_to_vector(state: State) -> np.array:
        arr = []
        for mass in state.masses:
            arr += [mass.x.x, mass.x.y, mass.v.x, mass.v.y]
        return np.array(arr)

    @staticmethod
    def state_to_vector_derivative(state: State) -> np.array:
        arr = []
        for mass in state.masses:
            arr += [mass.v.x, mass.v.y, mass.a.x, mass.a.y]
        return np.array(arr)

    @staticmethod
    def state_to_vector_x(state: State) -> np.array:
        arr = []
        for mass in state.masses:
            arr += [mass.x.x, mass.x.y]
        return np.array(arr)

    @staticmethod
    def state_to_vector_v(state: State) -> np.array:
        arr = []
        for mass in state.masses:
            arr += [mass.v.x, mass.v.y]
        return np.array(arr)

    @staticmethod
    def state_to_vector_a(state: State) -> np.array:
        arr = []
        for mass in state.masses:
            arr += [mass.a.x, mass.a.y]
        return np.array(arr)

    @staticmethod
    def vector_to_state(arr: np.array, state: State) -> None:
        if len(arr) != 4 * len(state.masses):
            raise ValueError("Mismatching vector and state size")
        for i in range(len(state.masses)):
            state.masses[i].x = Vector2(arr[4*i], arr[4*i+1])
            state.masses[i].v = Vector2(arr[4*i+2], arr[4*i+3])

    @staticmethod
    def vector_to_state_x(arr: np.array, state: State) -> None:
        if len(arr) != 2 * len(state.masses):
            raise ValueError("Mismatching vector and state size")
        for i in range(len(state.masses)):
            state.masses[i].x = Vector2(arr[2*i], arr[2*i+1])

    @staticmethod
    def vector_to_state_v(arr: np.array, state: State) -> None:
        if len(arr) != 2 * len(state.masses):
            raise ValueError("Mismatching vector and state size")
        for i in range(len(state.masses)):
            state.masses[i].v = Vector2(arr[2*i], arr[2*i+1])

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

        self.current_state.calc_acceleration()

        y0 = self.state_to_vector(self.current_state)
        yt = self.state_to_vector_derivative(self.current_state)

        y1 = y0 + yt * dt

        self.vector_to_state(y1, self.current_state)
        self.current_state.time += dt


class EulerCromer(Base):
    """https://en.wikipedia.org/wiki/Semi-implicit_Euler_method"""

    def __init__(self, state: State):
        super().__init__(state, 1, 0)

    def update(self, dt: float):

        self.current_state.calc_acceleration()
        x = self.state_to_vector_x(self.current_state)
        v = self.state_to_vector_v(self.current_state)
        a = self.state_to_vector_a(self.current_state)

        v += a * dt
        x += v * dt

        self.vector_to_state_x(x, self.current_state)
        self.vector_to_state_v(v, self.current_state)
        self.current_state.time += dt


class Midpoint(Base):

    def __init__(self, state: State):
        super().__init__(state, 2, 0)

    def update(self, dt: float):

        t0 = self.current_state.time
        y0 = self.state_to_vector(self.current_state)

        self.current_state.calc_acceleration()
        yt0 = self.state_to_vector_derivative(self.current_state)
        y1 = y0 + yt0 * (0.5*dt)
        self.vector_to_state(y1, self.current_state)
        self.current_state.time = t0 + 0.5*dt

        self.current_state.calc_acceleration()
        yt1 = self.state_to_vector_derivative(self.current_state)
        y2 = y0 + yt1 * dt
        self.vector_to_state(y2, self.current_state)
        self.current_state.time = t0 + dt


class RungeKutta(Base):

    def __init__(self, state: State):
        super().__init__(state, 4, 0)

    def update(self, dt: float):

        t0 = self.current_state.time
        y0 = self.state_to_vector(self.current_state)

        self.current_state.calc_acceleration()
        k1 = self.state_to_vector_derivative(self.current_state)

        self.vector_to_state(y0+0.5*dt*k1, self.current_state)
        self.current_state.time = t0 + 0.5*dt
        self.current_state.calc_acceleration()
        k2 = self.state_to_vector_derivative(self.current_state)

        self.vector_to_state(y0+0.5*dt*k2, self.current_state)
        self.current_state.time = t0 + 0.5*dt
        self.current_state.calc_acceleration()
        k3 = self.state_to_vector_derivative(self.current_state)

        self.vector_to_state(y0+dt*k3, self.current_state)
        self.current_state.time = t0 + dt
        self.current_state.calc_acceleration()
        k4 = self.state_to_vector_derivative(self.current_state)

        y1 = y0 + dt/6. * (k1+2.0*k2+2.0*k3+k4)
        self.vector_to_state(y1, self.current_state)
        self.current_state.time = t0 + dt


class Verlet(Base):
    """https://en.wikipedia.org/wiki/Verlet_integration#Basic_St%C3%B6rmer%E2%80%93Verlet"""

    def __init__(self, state: State):
        super().__init__(state, 1, 2)

    def update(self, dt: float):

        self.push_state()

        self.current_state.calc_acceleration()
        x0 = self.state_to_vector_x(self.current_state)
        v0 = self.state_to_vector_v(self.current_state)
        a0 = self.state_to_vector_a(self.current_state)

        if len(self.prev_states) < 2:
            x1 = x0 + v0*dt + 0.5*a0*dt*dt

        else:
            x_1 = self.state_to_vector_x(self.prev_states[-2])
            x1 = 2.0*x0 - x_1 + a0*dt*dt

        v1 = (x1-x0) / dt

        self.vector_to_state_x(x1, self.current_state)
        self.vector_to_state_v(v1, self.current_state)
        self.current_state.time += dt
