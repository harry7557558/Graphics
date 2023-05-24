from grpc import Call
from state import State
from pygame import Vector2, Vector3
import math
from random import random
from typing import Callable


def _gen_parametric_patch(fun: Callable[[float, float], Vector3],
                          un: int, vn: int) -> State:
    """Generate parametric surface, both u and v are open
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
    # state
    return State(vertices, faces)


def _gen_parametric_cylinder(fun: Callable[[float, float], Vector3],
                             un: int, vn: int) -> State:
    """Generate parametric surface, u is closed and v is open
       @fun: parametric equation, 0<=u<=1, 0<=v<=1
       @un: divide u into un equal intervals
       @vn: divide v into vn equal intervals"""
    # vertices
    vertices = [Vector3(0)] * (un*(vn+1))
    for ui in range(un):
        for vi in range(vn+1):
            u = ui / un
            v = vi / vn
            vertices[ui*(vn+1)+vi] = fun(u, v)
    # faces
    faces = []
    for ui in range(un):
        for vi in range(vn):
            i00 = ui*(vn+1)+vi
            i01 = ui*(vn+1)+(vi+1)
            i10 = ((ui+1) % un)*(vn+1)+vi
            i11 = ((ui+1) % un)*(vn+1)+(vi+1)
            faces.append((i00, i01, i11))
            faces.append((i00, i11, i10))
    # state
    return State(vertices, faces)


def _gen_parametric_torus(fun: Callable[[float, float], Vector3],
                          un: int, vn: int) -> State:
    """Generate parametric surface, both u and v are closed
       @fun: parametric equation, 0<=u<=1, 0<=v<=1
       @un: divide u into un equal intervals
       @vn: divide v into vn equal intervals"""
    # vertices
    vertices = [Vector3(0)] * (un*vn)
    for ui in range(un):
        for vi in range(vn):
            u = ui / un
            v = vi / vn
            vertices[ui*vn+vi] = fun(u, v)
    # faces
    faces = []
    for ui in range(un):
        for vi in range(vn):
            i00 = ui * vn + vi
            i01 = ui * vn + (vi+1) % vn
            i10 = ((ui+1) % un) * vn + vi
            i11 = ((ui+1) % un) * vn + (vi+1) % vn
            faces.append((i00, i01, i11))
            faces.append((i00, i11, i10))
    # state
    return State(vertices, faces)


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
    faces = [  # not necessary CCW
        (0, 1, 3), (0, 3, 2),
        (4, 5, 7), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 7), (1, 7, 3),
        (2, 6, 7), (2, 7, 3),
        (0, 4, 6), (0, 6, 2)
    ]
    return State(vertices, faces)


def plane(xyr: Vector2, xn: int, yn: int, noise: float) -> State:
    def fun(u, v):
        x = xyr.x * (2.0*u-1.0)
        y = xyr.y * (2.0*v-1.0)
        z = noise*(2.0*random()-1.0)
        return Vector3(x, y, z)
    return _gen_parametric_patch(fun, xn, yn)


def cylinder(a: float, b: float, h: float, un: int, vn: int, seamed: bool, noise: float) -> State:
    def fun(u, v):
        noi = noise * (2.0*random()-1.0) / (a**2+b**2)**0.5
        x = (a+noi*b) * math.cos(2.0*math.pi*u)
        y = (b+noi*a) * math.sin(2.0*math.pi*u)
        z = h * (2.0*v-1.0)
        return Vector3(x, y, z)
    if seamed:
        return _gen_parametric_patch(fun, un, vn)
    else:
        return _gen_parametric_cylinder(fun, un, vn)


def sphere_uv(a: float, b: float, c: float, un: int, vn: int, seamed: bool, noise: float) -> State:
    def fun(u, v):
        noi = noise * (2.0*random()-1.0)
        noi *= math.sin(math.pi*v)**2 / ((b*c)**2+(a*c)**2+(a*b)**2)**0.5
        x = (a+noi*b*c) * math.cos(2.0*math.pi*u) * math.sin(math.pi*v)
        y = (b+noi*a*c) * math.sin(2.0*math.pi*u) * math.sin(math.pi*v)
        z = (c+noi*a*b) * math.cos(math.pi*v)
        return Vector3(x, y, z)
    if seamed:
        return _gen_parametric_patch(fun, un, vn)
    else:
        return _gen_parametric_cylinder(fun, un, vn)


def torus(r0: float, r1: float, un: float, vn: float, u_seamed: bool, v_seamed: bool, noise: float) -> State:
    def fun(u, v):
        cos, sin, pi = math.cos, math.sin, math.pi
        noi = noise * (2.0*random()-1.0)
        x = (r0+r1*cos(2.0*pi*u))*cos(2.0*pi*v)
        y = (r0+r1*cos(2.0*pi*u))*sin(2.0*pi*v)
        z = r1*sin(2.0*pi*u)
        xn = cos(2.0*pi*u)*cos(2.0*pi*v)
        yn = cos(2.0*pi*u)*sin(2.0*pi*v)
        zn = sin(2.0*pi*u)
        return Vector3(x, y, z) + noi * Vector3(xn, yn, zn)
    if u_seamed and v_seamed:
        return _gen_parametric_patch(fun, un, vn)
    if u_seamed and not v_seamed:
        return _gen_parametric_cylinder(lambda u, v: fun(v, u), vn, un)
    if (not u_seamed) and v_seamed:
        return _gen_parametric_cylinder(fun, un, vn)
    if (not u_seamed) and not v_seamed:
        return _gen_parametric_torus(fun, un, vn)


def cersis(r0: float, r1: float, n: int, un: int, vn: int, noise: float) -> State:
    def fun(u, v):
        cos, sin, pi = math.cos, math.sin, math.pi
        asin, atan2 = math.asin, math.atan2
        u, v = 2.0*pi*u, 2.0*pi*v
        p = Vector3(cos(u)*(r0+r1*cos(v)), sin(u)*(r0+r1*cos(v)), r1*sin(v))
        p += 0.5*asin(sin(n*atan2(p.y, p.x))) * Vector3(cos(u), sin(u), 0)
        p.z *= 0.04 * (p.x**2+p.y**2) + 0.8
        p += noise * (2.0*Vector3(random(), random(), random())-Vector3(1.0))
        return p
    return _gen_parametric_torus(fun, un, vn)
