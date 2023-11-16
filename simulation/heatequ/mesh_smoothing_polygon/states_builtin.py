from state import State
from pygame import Vector2
import math
import random


random.seed(0)


def noisy_circle(n: int, r0: float, noi: float, power: float = 1.0) -> State:
    points = []
    for i in range(n):
        u = 2.0*(i+0.5)/n-1.0
        a = math.pi * (1, -1)[u < 0] * abs(u)**power
        r = r0 + noi*(2.0*random.random()-1.0)
        points.append(Vector2(r*math.cos(a), r*math.sin(a)))
    weights = [1.0+0.0*i/n for i in range(n)]
    return State(points, weights, True)


def noisy_flower(n: int, fm: int, fw: float, r0: float, noi: float) -> State:
    points = []
    for i in range(n):
        a = 2.0*math.pi*i/n
        r = r0 * (1.0+fw*math.exp(math.sin(fm*a)))
        p = Vector2(r*math.cos(a), r*math.sin(a))
        p += noi*(2.0*Vector2(random.random(), random.random())-Vector2(1))
        points.append(p)
    weights = [1.0+0.0*i/n for i in range(n)]
    return State(points, weights, True)


def noisy_line(n: int, r0: float, noi: float,
               power: float = 1.0, scale_power: bool = False) -> State:
    points = []
    for i in range(n):
        u = (i+0.5)/n
        x = 2.0 * pow(u, power) - 1.0
        y = noi*(2.0*random.random()-1.0)
        if scale_power:
            power*u**(power-1)
        points.append(Vector2(x, y))
    weights = [min(i*(n-i-1), 1) for i in range(n)]
    return State(points, weights, False)
