import pygame
from pygame import Vector2, Vector3
from OpenGL.GL import *
from OpenGL.GLU import *
import math


class Viewport3D():
    """Orthographic projection"""

    def __init__(self, resolution: Vector2, scale: float, center: Vector3, rx: float, rz: float):
        """Angles are in degrees"""
        self._resolution = resolution
        self._scale = scale
        self._center = center
        self._rx = rx
        self._rz = rz

    def mouse_move(self, mouse_delta: Vector2) -> None:
        if type(mouse_delta) != Vector2:
            mouse_delta = Vector2(*mouse_delta)
        self._rx += 1.0*mouse_delta.y
        self._rz += 1.0*mouse_delta.x

    def mouse_scroll(self, mouse_pos: Vector2, zoom: float) -> None:
        if type(mouse_pos) != Vector2:
            mouse_pos = Vector2(*mouse_pos)
        if not zoom > 0.0:
            raise ValueError("Mouse zooming must be positive.")
        self._scale *= zoom

    def draw_line(self, p1: Vector3, p2: Vector3, color: Vector3) -> None:
        glBegin(GL_LINES)
        glColor3f(*color)
        glVertex3fv(tuple(p1))
        glVertex3fv(tuple(p2))
        glEnd()

    def draw_quad(self, verts, color) -> None:
        glBegin(GL_QUADS)
        for i in range(4):
            glColor3fv(color)
            glVertex3fv(verts[i])
        glEnd()

    def draw(self, surface: pygame.Surface) -> None:
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # setup matrix
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45.0, self._resolution.y/self._resolution.x, 1e-1, 1e+2)
        glTranslated(0.0, 0.0, -3.0/self._scale)
        glRotated(self._rx, 1, 0, 0)
        glRotated(self._rz, 0, 0, 1)
        glTranslated(*(-self._center))

        # axes
        sz = 10 ** math.floor(math.log10(4.0/self._scale))
        a = 2.0*sz
        self.draw_line(Vector3(0, 0, 0), Vector3(a, 0, 0), (0.5, 0.2, 0.2))
        self.draw_line(Vector3(0, 0, 0), Vector3(0, a, 0), (0.2, 0.5, 0.2))
        self.draw_line(Vector3(0, 0, 0), Vector3(0, 0, a), (0.2, 0.2, 0.5))

        # grid
        GRID_COLOR = (0.2, 0.2, 0.2)
        for i in range(-10, 11, 1):
            a = 0.1*sz * i
            self.draw_line(Vector3(a, -sz, 0),
                           Vector3(a, sz, 0), GRID_COLOR)
            self.draw_line(Vector3(-sz, a, 0),
                           Vector3(sz, a, 0), GRID_COLOR)
