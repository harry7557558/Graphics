import numpy as np
import scipy.sparse
import pygame
from viewport import Viewport
from typing import Callable


class State:

    def __init__(self,
                 x0: float, x1: float, n: float,
                 k: float,
                 initial_temp: Callable[[float], float],
                 heater: Callable[[float], list[float, float]],
                 boundary_temp_l: float = None, boundary_temp_r: float = None
                 ):
        self.x0 = x0  # start of interval
        self.x1 = x1  # end of interval
        self.n = n  # number of sample points
        self.k = k  # ∂u/∂t = k * ∂²u/∂x²
        self.initial_temp = initial_temp  # temperature(x)
        self.heater = heater  # power(x, t)
        self.boundary_temp_l = boundary_temp_l  # None if closed
        self.boundary_temp_r = boundary_temp_r  # None if closed
        self.dx = (self.x1 - self.x0) / (self.n - 1)
        self.x = np.linspace(self.x0, self.x1, self.n)  # positions
        self.u = np.array([initial_temp(x) for x in self.x])  # temperatures
        self.time = 0.0

    def calc_dudt(self) -> np.array:
        """Calculate the power at each point"""
        # heater
        dudt = np.array([self.heater(x, self.time) for x in self.x])
        # laplacian
        for i in range(self.n):
            u0 = self.boundary_temp_l if i == 0 else self.u[i-1]
            if u0 == None:
                u0 = self.u[i]
            u2 = self.boundary_temp_r if i == self.n-1 else self.u[i+1]
            if u2 == None:
                u2 = self.u[i]
            u1 = self.u[i]
            lap_u = (u0+u2-2.0*u1) / self.dx**2
            dudt[i] += self.k * lap_u
        return dudt

    def calc_dpdu_n(self) -> np.array:
        """Numerically calculate the derivative of dudt to u, returns a matrix"""
        eps = 1e-4
        u0 = np.array(self.u)
        mat = []
        for i in range(self.n):
            du = np.zeros(self.n)
            du[i] = eps
            self.u = u0 + du
            p1 = self.calc_dudt()
            self.u = u0 - du
            p0 = self.calc_dudt()
            mat.append((p1-p0)/(2.0*eps))
        self.u = u0
        return np.array(mat).T

    def calc_dpdu(self) -> scipy.sparse.base:
        """Analytically calculate ∂p/∂u, returns a matrix"""
        rows = []
        cols = []
        nums = []
        # heater - zero
        # laplacian - symmetric tridiagonal
        for i in range(self.n):  # for each p
            if i == 0:
                i0 = i if self.boundary_temp_l is None else -1
            else:
                i0 = i-1
            if i == self.n - 1:
                i2 = i if self.boundary_temp_r is None else -1
            else:
                i2 = i+1
            cols.append(i)
            rows.append(i)
            nums.append(-2.0 * self.k/self.dx**2)
            if i0 != -1:
                cols.append(i)
                rows.append(i0)
                nums.append(1.0 * self.k/self.dx**2)
            if i2 != -1:
                cols.append(i)
                rows.append(i2)
                nums.append(1.0 * self.k/self.dx**2)
        return scipy.sparse.coo_matrix((nums, (cols, rows)))

    def draw(self, surface: pygame.Surface, viewport: Viewport, color):
        for i in range(self.n):
            j = min(i+1, self.n-1)
            p1 = (self.x[i], self.u[i])
            p2 = (self.x[j], self.u[j])
            q1 = viewport.world_to_screen(p1)
            q2 = viewport.world_to_screen(p2)
            if max(q1.length(), q2.length()) < 15360:
                pygame.draw.aaline(surface, color, q1, q2)

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
