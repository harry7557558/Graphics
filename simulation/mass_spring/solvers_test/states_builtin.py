from typing import Callable
from state import Vector2, Mass, Spring, State
import math


def ngrad(fun: Callable[[float], Vector2], p: Vector2) -> Vector2:
    """ Numerical gradient of fun(Vector2) """
    eps = 1e-4
    dfdx = (fun(p + Vector2(eps, 0)) - fun(p - Vector2(eps, 0))) / (2.0*eps)
    dfdy = (fun(p + Vector2(0, eps)) - fun(p - Vector2(0, eps))) / (2.0*eps)
    return Vector2(dfdx, dfdy)


def square_barred(has_ground: bool = False) -> State:
    """A square with a mass in the middle and cross springs, tied at one vertex"""
    drag = 0.1
    masses = [
        Mass(inv_m=0.0,  drag=drag, x=Vector2(-0.5, 1.0)),   # bottom-left, static
        Mass(inv_m=10.0, drag=drag, x=Vector2(0.5, 1.0)),   # bottom-right, 100g
        Mass(inv_m=10.0, drag=drag, x=Vector2(0.5, 2.0)),   # top-right, 100g
        Mass(inv_m=10.0, drag=drag, x=Vector2(-0.5, 2.0)),  # top-left, 100g
        Mass(inv_m=10.0, drag=drag, x=Vector2(0.0, 1.5)),   # center, 100g
    ]

    ks = 10.0  # spring constant in N/m for a 1m spring
    kd = 5.0  # damping constant
    cl = masses[0].x.distance_to(masses[4].x)  # length of a cross spring
    springs = [
        # border springs
        Spring((0, 1), 1.0, ks, kd),
        Spring((1, 2), 1.0, ks, kd),
        Spring((2, 3), 1.0, ks, kd),
        Spring((3, 0), 1.0, ks, kd),
        # cross springs
        Spring((0, 4), cl, ks/cl, kd/cl),
        Spring((1, 4), cl, ks/cl, kd/cl),
        Spring((2, 4), cl, ks/cl, kd/cl),
        Spring((3, 4), cl, ks/cl, kd/cl)
    ]

    def acc_field(p, t):
        return Vector2(0, -1000*p.y) if p.y < 0.0 else Vector2(0)

    return State(masses, springs, acc_field if has_ground else None)


def sheet_hang(mass: float, ks: float, kd: float,
               xd: int, yd: int, lx: float, ly: float,
               has_cross: bool, all_hang: bool) -> State:
    """a hanging square sheet of cloth
    Args:
        mass: total mass of the sheet
        ks, kd: spring constant and damping constant of a 1m spring
        xd: divide x into this number of intervals
        yd: divide y into this number of intervals
        lx: length in x-direction
        ly: length in y-direction
        has_cross: has cross spring or not
        all_hang: True if all top is hanging, otherwise only two corners hanging
    """
    masses = []
    springs = []

    # masses
    inv_m = (xd+1)*(yd+1) / mass
    for y in range(yd+1):
        for x in range(xd+1):
            p = Vector2(x*lx/xd, y*ly/yd)-Vector2(0.5*lx, ly)+Vector2(0, 2.5)
            masses.append(Mass(inv_m, x=p))
    if all_hang:
        for i in range(0, xd+1):
            masses[yd*(xd+1)+i].inv_m = 0.0
    else:
        masses[yd*(xd+1)].inv_m = 0.0
        masses[yd*(xd+1)+xd].inv_m = 0.0

    # springs
    dx, dy = lx/xd, ly/yd
    dl = math.hypot(dx, dy)
    for y in range(yd+1):  # horizontal springs
        for x in range(xd):
            springs.append(Spring(
                (y*(xd+1)+x, y*(xd+1)+(x+1)),
                dx, ks/dx, kd/dx))
    for x in range(xd+1):  # vertical springs
        for y in range(yd):
            springs.append(Spring(
                (y*(xd+1)+x, (y+1)*(xd+1)+x),
                dy, ks/dy, kd/dy))
    if has_cross:  # cross springs
        for x in range(xd):
            for y in range(yd):
                springs.append(Spring(
                    (y*(xd+1)+x, (y+1)*(xd+1)+(x+1)),
                    dl, ks/dl, kd/dl))
                springs.append(Spring(
                    (y*(xd+1)+(x+1), (y+1)*(xd+1)+x),
                    dl, ks/dl, kd/dl))

    # floor
    def acc_field(p, t):
        return Vector2(0, -1000*p.y) if p.y < 0.0 else Vector2(0)

    return State(masses, springs, acc_field)
