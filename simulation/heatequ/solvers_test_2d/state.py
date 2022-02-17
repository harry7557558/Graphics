import numpy as np
import scipy.sparse
import pygame
from pygame import Vector2, Vector3
from viewport_3d import Viewport3D
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from typing import Callable


class State:

    def __init__(self,
                 xy0: Vector2, xy1: Vector2, xn: int, yn: int,
                 k: float,
                 initial_temp: Callable[[float], Vector2],
                 heater: Callable[[float], list[Vector2, float]]):
        self.x0 = xy0.x  # minimum x
        self.x1 = xy1.x  # maximum x
        self.y0 = xy0.y  # minimum y
        self.y1 = xy1.y  # maximum y
        self.xn = xn  # number of sample points along x
        self.yn = yn  # number of sample points along y
        self.n = xn*yn
        self.k = k  # ∂u/∂t = k * ∂²u/∂x²
        self.initial_temp = lambda x, y: initial_temp(Vector2(x, y))
        self.heater = lambda x, y, t: heater(Vector2(x, y), t)
        self.dx = (self.x1 - self.x0) / (self.xn - 1)
        self.dy = (self.y1 - self.y0) / (self.yn - 1)
        along_x = np.linspace(self.x0, self.x1, self.xn)
        along_y = np.linspace(self.y0, self.y1, self.yn)
        self.xy = np.dstack(np.meshgrid(along_y, along_x))  # positions
        self.u = np.zeros((self.xn, self.yn))  # temperatures
        for xi in range(self.xn):
            for yi in range(self.yn):
                self.u[xi][yi] = self.initial_temp(*self.xy[xi][yi])
        self.time = 0.0

    def calc_dudt(self) -> np.array:
        """Calculate the power at each point"""
        dudt = np.zeros((self.xn, self.yn))
        # heater
        for xi in range(self.xn):
            for yi in range(self.yn):
                dudt[xi][yi] += self.heater(*self.xy[xi][yi], self.time)
        # laplacian
        for xi in range(self.xn):
            for yi in range(self.yn):
                xi0 = max(xi-1, 0)
                xi1 = min(xi+1, self.xn-1)
                yi0 = max(yi-1, 0)
                yi1 = min(yi+1, self.yn-1)
                u = self.u[xi][yi]
                ux0 = self.u[xi0][yi]
                ux1 = self.u[xi1][yi]
                uy0 = self.u[xi][yi0]
                uy1 = self.u[xi][yi1]
                d2udx2 = (ux0+ux1-2.0*u) / self.dx**2
                d2udy2 = (uy0+uy1-2.0*u) / self.dy**2
                dudt[xi][yi] += self.k * (d2udx2+d2udy2)
        return dudt

    def calc_dpdu_n(self) -> np.array:
        """Numerically calculate the derivative of dudt to u, returns a matrix"""
        eps = 1e-4
        u0 = np.array(self.u).reshape((self.n))
        mat = []
        for i in range(self.n):
            du = np.zeros(self.n)
            du[i] = eps
            self.u = (u0 + du).reshape((self.xn, self.yn))
            p1 = self.calc_dudt().reshape((self.n))
            self.u = (u0 - du).reshape((self.xn, self.yn))
            p0 = self.calc_dudt().reshape((self.n))
            mat.append((p1-p0)/(2.0*eps))
        self.u = u0.reshape((self.xn, self.yn))
        return np.array(mat).T

    def calc_dpdu(self) -> scipy.sparse.base:
        """Analytically calculate ∂p/∂u, returns a matrix"""
        rows = []
        cols = []
        nums = []
        # heater - none
        # laplacian - symmetric
        for xi in range(self.xn):
            for yi in range(self.yn):
                pi = xi*self.yn + yi
                xi0 = max(xi-1, 0)
                xi1 = min(xi+1, self.xn-1)
                yi0 = max(yi-1, 0)
                yi1 = min(yi+1, self.yn-1)
                # d2udx2 = (ux0+ux1-2.0*u) / dx**2
                k = self.k / self.dx**2
                rows += [pi]*3
                cols += [xi0*self.yn + yi,
                         xi1*self.yn + yi,
                         xi*self.yn + yi]
                nums += [k, k, -2.0*k]
                # d2udy2 = (uy0+uy1-2.0*u) / dy**2
                k = self.k / self.dy**2
                rows += [pi]*3
                cols += [xi*self.yn + yi0,
                         xi*self.yn + yi1,
                         xi*self.yn + yi]
                nums += [k, k, -2.0*k]
        return scipy.sparse.coo_matrix((nums, (cols, rows)))

    def draw(self, surface: pygame.Surface, viewport: Viewport3D, color):
        points = np.concatenate((self.xy,
                                 self.u.reshape((self.xn, self.yn, 1))),
                                axis=2)
        for xi in range(self.xn-1):
            for yi in range(self.yn-1):
                verts = [
                    points[xi][yi],
                    points[xi+1][yi],
                    points[xi+1][yi+1],
                    points[xi][yi+1]
                ]
                n = np.cross(verts[2]-verts[0], verts[3]-verts[1])
                n /= np.linalg.norm(n)
                s = abs(n[2])
                s = (1.0-(1.0-s)**0.1)**0.5
                rgb = np.array(color)*s
                viewport.draw_quad(verts, rgb.tolist())

    def calc_mean_temperature(self) -> float:
        return np.average(self.u)

    def draw_info(self, surface: pygame.Surface, font: pygame.font.Font,
                  topleft: tuple[int, int], color=(255, 255, 255)) -> None:
        time = self.time
        temperature = self.calc_mean_temperature()
        text = font.render(
            "{:.2f}s, {:.4f}K".format(
                time, temperature),
            True, color)
        text_rect = text.get_rect()
        text_rect.topleft = topleft
        surface.blit(text, text_rect)

    def visualize_matrix(self, filepath):
        from PIL import Image
        mat = self.calc_dpdu().toarray()
        mat = abs(mat) / np.amax(abs(mat))
        mat = (255*mat).astype(np.uint8)
        img = Image.fromarray(mat)
        img.save(filepath)
