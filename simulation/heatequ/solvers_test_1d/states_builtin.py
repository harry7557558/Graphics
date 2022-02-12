from state import State
import math


def dam(n: int, k: float, x0: float, x1: float,
        boundary_temp_l: float = None, boundary_temp_r: float = None) -> State:

    return State(-1.0, 1.0, n, k,
                 lambda x: 1.0 if x0 <= x <= x1 else 0.0,
                 lambda x, t: 0.0,
                 boundary_temp_l, boundary_temp_r)


def dancing_heater(n: int, k: float, x0: float, x1: float,
        boundary_temp_l: float = None, boundary_temp_r: float = None) -> State:

    return State(-1.0, 1.0, n, k,
                 lambda x: 1.0 if x0 <= x <= x1 else 0.0,
                 lambda x, t: math.sin(t) if -0.1 < x < 0.1 else 0.0,
                 boundary_temp_l, boundary_temp_r)


def two_heaters(n: int, k: float, x1: float, x2: float) -> State:

    return State(-1.0, 1.0, n, k,
                 lambda x: 0.0,
                 lambda x, t: math.exp(-100.*(x-x1)**2)+math.exp(-100.*(x-x2)**2),
                 0.0, 0.0)
