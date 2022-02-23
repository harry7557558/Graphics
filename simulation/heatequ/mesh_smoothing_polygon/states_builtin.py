from state import State
from pygame import Vector2
import math
import random


random.seed(0)


def noisy_circle(n: int, r0: float, noi: float) -> State:

    points = []
    for i in range(n):
        a = 2.0*math.pi*i/n
        r = r0 + noi*(2.0*random.random()-1.0)
        points.append(Vector2(r*math.cos(a), r*math.sin(a)))
    weights = [1.0+0.0*i/n for i in range(n)]
    return State(points, weights, True)


def noisy_line(n: int, r0: float, noi: float) -> State:

    points = []
    for i in range(n):
        x = 2.0*i/(n-1)-1.0
        y = noi*(2.0*random.random()-1.0)
        points.append(Vector2(x, y))
    weights = [min(i*(n-i-1), 1) for i in range(n)]
    return State(points, weights, False)
