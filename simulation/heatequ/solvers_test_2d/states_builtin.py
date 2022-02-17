from state import State
from pygame import Vector2, Vector3
import math


def circle_dam(nx: int, ny: int, k: float, circ_o: Vector2, circ_r: float) -> State:

    return State(Vector2(-1), Vector2(1), nx, ny, k,
                 lambda xy: 1.0 if (xy-circ_o).length() < circ_r else 0.0,
                 lambda xy, t: 0.0)


def heater_cooler(nx: int, ny: int, k: float, hr: float, pw: float) -> State:

    return State(Vector2(-1), Vector2(1), nx, ny, k,
                 lambda xy: 0.0,
                 lambda xy, t: pw*math.tanh(1e6*(xy.x+xy.y)) if xy.length() < hr else 0.0)
