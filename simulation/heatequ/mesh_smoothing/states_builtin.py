from grpc import Call
from state import State
from pygame import Vector2, Vector3
import math
from random import random
from typing import Callable


def _gen_parametric_patch(fun: Callable[[float, float], Vector3],
                          un: int, vn: int) -> State:
    """Generate parametric surface
       @fun: parametric equation, 0<=u<=1, 0<=v<=1
       @un: divide u into un equal intervals
       @vn: divide v into vn equal intervals"""
    # vertices
    vertices = [Vector3(0)] * ((un+1)*(vn+1))
    for ui in range(un+1):
        for vi in range(vn+1):
            u = ui / un
            v = vi / vn
            vertices[ui*(vn+1)+vi] = fun(u, v)
    # edges
    edges = []
    for ui in range(un+1):
        for vi in range(vn):
            i = ui*(vn+1)+vi
            j = ui*(vn+1)+(vi+1)
            edges.append((i, j))
    for ui in range(un):
        for vi in range(vn+1):
            i = ui*(vn+1)+vi
            j = (ui+1)*(vn+1)+vi
            edges.append((i, j))
    # faces
    faces = []
    for ui in range(un):
        for vi in range(vn):
            i00 = ui*(vn+1)+vi
            i01 = ui*(vn+1)+(vi+1)
            i10 = (ui+1)*(vn+1)+vi
            i11 = (ui+1)*(vn+1)+(vi+1)
            faces.append((i00, i01, i11))
            faces.append((i00, i11, i10))
            # make consistent
            edges.append((i00, i11))
    # state
    return State(vertices, edges, faces)


def unit_cube() -> State:
    vertices = [
        Vector3(-1, -1, -1),
        Vector3(-1, -1, 1),
        Vector3(-1, 1, -1),
        Vector3(-1, 1, 1),
        Vector3(1, -1, -1),
        Vector3(1, -1, 1),
        Vector3(1, 1, -1),
        Vector3(1, 1, 1),
    ]
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (3, 7), (2, 6)
    ]
    faces = [  # not necessary CCW
        (0, 1, 3), (0, 3, 2),
        (4, 5, 7), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 7), (1, 7, 3),
        (2, 6, 7), (2, 7, 3),
        (0, 4, 6), (0, 6, 2)
    ]
    return State(vertices, edges, faces)


def noisy_plane(xyr: Vector2, xn: int, yn: int, noi: float) -> State:
    def fun(u, v):
        x = xyr.x * (2.0*u-1.0)
        y = xyr.y * (2.0*v-1.0)
        z = noi*(2.0*random()-1.0)
        return Vector3(x, y, z)
    return _gen_parametric_patch(fun, xn, yn)
