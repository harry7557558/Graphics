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
                 h0: float, g: float, f: float, k: float, nu: float,
                 initial_h: Callable[[Vector2], float]):
        # parameters - https://en.wikipedia.org/wiki/Shallow_water_equations#Non-conservative_form
        self.h0 = h0  # mean height (m)
        self.g = g  # positive acceleration due to gravity (m/s²)
        self.f = f  # Coriolis coefficient (s⁻¹)
        self.k = k  # viscous drag for velocity
        self.nu = nu  # kinematic viscosity (m²/s)
        self.initial_h = lambda x, y: initial_h(Vector2(x, y))
        # discretization
        self.x0 = xy0.x  # minimum x (m)
        self.x1 = xy1.x  # maximum x (m)
        self.y0 = xy0.y  # minimum y (m)
        self.y1 = xy1.y  # maximum y (m)
        self.xn = xn  # number of sample points along x
        self.yn = yn  # number of sample points along y
        self.n = xn*yn
        self.dx = (self.x1 - self.x0) / (self.xn - 1)
        self.dy = (self.y1 - self.y0) / (self.yn - 1)
        along_x = np.linspace(self.x0, self.x1, self.xn)
        along_y = np.linspace(self.y0, self.y1, self.yn)
        self.xy = np.dstack(np.meshgrid(along_x, along_y))  # positions
        self.xy = np.transpose(self.xy, (1, 0, 2))
        # initial
        self.u = np.zeros((self.xn, self.yn))  # velocity along x (m/s)
        self.v = np.zeros((self.xn, self.yn))  # velocity along y (m/s)
        self.h = np.zeros((self.xn, self.yn))  # height (m)
        for xi in range(self.xn):
            for yi in range(self.yn):
                self.h[xi][yi] = self.h0 + self.initial_h(*self.xy[xi][yi])
        self.time = 0.0

    def calc_dsdt(self) -> np.array:
        """Returns (∂h/∂t, ∂u/∂t, ∂v/∂t)"""
        g, nu, k, f = self.g, self.nu, self.k, self.f
        dhdt = np.zeros((self.xn, self.yn))
        dudt = np.zeros((self.xn, self.yn))
        dvdt = np.zeros((self.xn, self.yn))
        for xi in range(self.xn):
            for yi in range(self.yn):
                xi0 = xi-1 if xi != 0 else xi+1
                xi1 = xi+1 if xi != self.xn-1 else xi-1
                xis0 = -1. if xi == 0 else 1.
                xis1 = -1. if xi == self.xn-1 else 1.
                yi0 = yi-1 if yi != 0 else yi+1
                yi1 = yi+1 if yi != self.yn-1 else yi-1
                yis0 = -1. if yi == 0 else 1.
                yis1 = -1. if yi == self.yn-1 else 1.
                # h
                h = self.h[xi][yi]
                hx0 = self.h[xi0][yi]
                hx1 = self.h[xi1][yi]
                hy0 = self.h[xi][yi0]
                hy1 = self.h[xi][yi1]
                hx = (hx1-hx0)/(2.*self.dx)
                hy = (hy1-hy0)/(2.*self.dy)
                # u
                u = self.u[xi][yi]
                ux0 = self.u[xi0][yi] * xis0
                ux1 = self.u[xi1][yi] * xis1
                uy0 = self.u[xi][yi0]
                uy1 = self.u[xi][yi1]
                ux = (ux1-ux0)/(2.*self.dx)
                uy = (uy1-uy0)/(2.*self.dy)
                ulap = (ux1+ux0-2.*u)/self.dx**2+(uy1+uy0-2.*u)/self.dy**2
                # v
                v = self.v[xi][yi]
                vx0 = self.v[xi0][yi]
                vx1 = self.v[xi1][yi]
                vy0 = self.v[xi][yi0] * yis0
                vy1 = self.v[xi][yi1] * yis1
                vx = (vx1-vx0)/(2.*self.dx)
                vy = (vy1-vy0)/(2.*self.dy)
                vlap = (vx1+vx0-2.*v)/self.dx**2+(vy1+vy0-2.*v)/self.dy**2
                # equation
                dhdt[xi][yi] = -h*(ux+vy) - u*hx - v*hy
                dudt[xi][yi] = -g*hx - u*ux - v*uy + nu*ulap - k*u + f*v
                dvdt[xi][yi] = -g*hy - u*vx - v*vy + nu*vlap - k*v - f*u
        return (dhdt, dudt, dvdt)

    def calc_dstds_n(self) -> np.array:
        """Numerically calculate the derivative of (∂h/∂t,∂u/∂t,∂v/∂t) to (h,u,v), returns a matrix"""
        eps = 1e-4
        s0 = np.array([self.h, self.u, self.v]).reshape((3*self.n))
        mat = []
        for i in range(3*self.n):
            ds = np.zeros(3*self.n)
            ds[i] = eps
            self.h, self.u, self.v = (s0 + ds).reshape((3, self.xn, self.yn))
            st1 = np.array(self.calc_dsdt()).reshape((3*self.n))
            self.h, self.u, self.v = (s0 - ds).reshape((3, self.xn, self.yn))
            st0 = np.array(self.calc_dsdt()).reshape((3*self.n))
            mat.append((st1-st0)/(2.0*eps))
        self.h, self.u, self.v = s0.reshape((3, self.xn, self.yn))
        return np.array(mat).T

    def calc_dstds(self) -> scipy.sparse.base:
        """Analytically calculate ∂p/∂u, returns a matrix"""
        rows = []
        cols = []
        nums = []
        h, u, v = self.h, self.u, self.v
        dx, dy = self.dx, self.dy
        xn, yn = self.xn, self.yn
        for xi in range(xn):
            for yi in range(yn):
                # Equations:
                # ht' = -h*(ux'+vy') -u*hx'-v*hy'
                # ut' = -g*hx' -u*ux'-v*uy' + nu*∇²u -k*u +f*v
                # vt' = -g*hy' -u*vx'-v*vy' + nu*∇²v -k*v -f*u
                xi0 = xi-1 if xi != 0 else xi+1
                xi1 = xi+1 if xi != xn-1 else xi-1
                xis0 = -1. if xi == 0 else 1.
                xis1 = -1. if xi == xn-1 else 1.
                yi0 = yi-1 if yi != 0 else yi+1
                yi1 = yi+1 if yi != yn-1 else yi-1
                yis0 = -1. if yi == 0 else 1.
                yis1 = -1. if yi == yn-1 else 1.
                # ∂ht'/∂h = -(ux'+vy') -u*∂hx'/∂h -v*∂hy'/∂h
                rows += [xi*yn+yi]*5
                cols += [xi*yn+yi,
                         xi1*yn+yi, xi0*yn+yi,
                         xi*yn+yi1, xi*yn+yi0]
                nums += [-(u[xi1][yi]*xis1-u[xi0][yi]*xis0)/(2.*dx)
                         - (v[xi][yi1]*yis1-v[xi][yi0]*yis0)/(2.*dy),
                         -u[xi][yi]/(2.*dx), -u[xi][yi]/(-2.*dx),
                         -v[xi][yi]/(2.*dy), -v[xi][yi]/(-2.*dy)]
                # ∂ht'/∂u = -h*∂ux'/∂u -hx'
                rows += [xi*yn+yi]*3
                cols += [self.n + xi1*yn+yi, self.n + xi0*yn+yi,
                         self.n + xi*yn+yi]
                nums += [-h[xi][yi]*xis1/(2.*dx), -h[xi][yi]*xis0/(-2.*dx),
                         -(h[xi1][yi]-h[xi0][yi])/(2.*dx)]
                # ∂ht'/∂v = -h*∂vy'/∂v -hy'
                rows += [xi*yn+yi]*3
                cols += [2*self.n + xi*yn+yi1, 2*self.n + xi*yn+yi0,
                         2*self.n + xi*yn+yi]
                nums += [-h[xi][yi]*yis1/(2.*dy), -h[xi][yi]*yis0/(-2.*dy),
                         -(h[xi][yi1]-h[xi][yi0])/(2.*dy)]
                # ∂ut'/∂h = -g*∂hx'/∂h
                rows += [self.n + xi*yn+yi]*2
                cols += [xi1*yn+yi, xi0*yn+yi]
                nums += [-self.g/(2.*dx), -self.g/(-2.*dx)]
                # ∂ut'/∂u = -ux'-u*∂ux'/∂u -v*∂uy'/∂u + nu*∂[∇²u]/∂u -k
                rows += [self.n + xi*yn+yi]*5
                cols += [self.n + xi*yn+yi,
                         self.n + xi1*yn+yi, self.n + xi0*yn+yi,
                         self.n + xi*yn+yi1, self.n + xi*yn+yi0]
                d_app = [-(u[xi1][yi]*xis1-u[xi0][yi]*xis0)/(2.*dx),
                         -u[xi][yi]*xis1/(2.*dx),
                         -u[xi][yi]*xis0/(-2.*dx),
                         -v[xi][yi]/(2.*dy),
                         -v[xi][yi]/(-2.*dy)]
                d_vis = [-2.*self.nu*(dx**-2+dy**-2)-self.k,
                         self.nu*xis1/dx**2,
                         self.nu*xis0/dx**2,
                         self.nu/dy**2,
                         self.nu/dy**2]
                nums += [d_app[i]+d_vis[i] for i in range(5)]
                # ∂ut'/∂v = -uy' +f
                rows += [self.n + xi*yn+yi]
                cols += [2*self.n + xi*yn+yi]
                nums += [-(u[xi][yi1]-u[xi][yi0])/(2.*dy) + self.f]
                # ∂vt'/∂h = -g*∂hy'/∂h
                rows += [2*self.n + xi*yn+yi]*2
                cols += [xi*yn+yi1, xi*yn+yi0]
                nums += [-self.g/(2.*dy), -self.g/(-2.*dy)]
                # ∂vt'/∂u = -vx' -f
                rows += [2*self.n + xi*yn+yi]
                cols += [self.n + xi*yn+yi]
                nums += [-(v[xi1][yi]-v[xi0][yi])/(2.*dx) - self.f]
                # ∂vt'/∂v = -u*∂vx'/∂v -vy'-v*∂vy'/∂v + nu*∂[∇²v]/∂v-k
                rows += [2*self.n + xi*yn+yi]*5
                cols += [2*self.n + xi*yn+yi,
                         2*self.n + xi*yn+yi1, 2*self.n + xi*yn+yi0,
                         2*self.n + xi1*yn+yi, 2*self.n + xi0*yn+yi]
                d_app = [-(v[xi][yi1]*yis1-v[xi][yi0]*yis0)/(2.*dy),
                         -v[xi][yi]*yis1/(2.*dy),
                         -v[xi][yi]*yis0/(-2.*dy),
                         -u[xi][yi]/(2.*dx),
                         -u[xi][yi]/(-2.*dx)]
                d_vis = [-2.*self.nu*(dx**-2+dy**-2)-self.k,
                         self.nu*yis1/dy**2,
                         self.nu*yis0/dy**2,
                         self.nu/dx**2,
                         self.nu/dx**2]
                nums += [d_app[i]+d_vis[i] for i in range(5)]

        return scipy.sparse.coo_matrix((nums, (rows, cols)), shape=(3*self.n, 3*self.n))

    def draw(self, surface: pygame.Surface, viewport: Viewport3D, color):
        points = np.concatenate((self.xy,
                                 self.h.reshape((self.xn, self.yn, 1))),
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

    @staticmethod
    def visualize_matrix(mat, filepath):
        from PIL import Image
        if type(mat) != np.ndarray:  # sparse
            mat = mat.toarray()
        mat = abs(mat)
        print(np.amax(mat))
        mat = mat**0.1  # make small vals stand out
        mat /= np.amax(mat)
        mat = (255*mat).astype(np.uint8)
        img = Image.fromarray(mat)
        img.save(filepath)
