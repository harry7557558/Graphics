from state import State
from pygame import Vector2, Vector3
import math


def pool(rx: float, ry: float, nx: int, ny: int, h0: float) -> State:
    return State(Vector2(-rx, -ry), Vector2(rx, ry), nx, ny,
                 h0, 9.8, .0001, .01, .001,
                 lambda xy: 0.5*math.exp(-(4.*xy.length())**2))
