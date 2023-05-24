import numpy as np
import scipy.sparse
import pygame
from pygame import Vector2
from viewport import Viewport
from typing import Callable


class State:

    def __init__(self, points: list[Vector2], weights: list[float], closed: bool):
        if len(points) != len(weights):
            raise ValueError("Length of points not equal length of weights")
        self.n = len(points)
        self.points = [Vector2(p) for p in points]
        self.weights = [float(p) for p in weights]
        self.closed = closed

    def get_points_arr(self) -> np.ndarray:
        return np.array([[p.x, p.y] for p in self.points]).flatten()

    def set_points_arr(self, arr: np.ndarray) -> None:
        for i in range(self.n):
            self.points[i] = Vector2(arr[2*i], arr[2*i+1])

    def _calc_indices_weights(self):
        """Used by calc_dpdt() and calc_ddpdtdp()"""
        i0s, i1s = [], []
        w0s, w1s, ws = [], [], []
        mode = "average"
        points = self.points
        for i in range(self.n):
            # indices
            i0, i1 = i-1, i+1
            if self.closed:
                i0, i1 = i0 % self.n, i1 % self.n
            else:
                i0, i1 = max(i0, 0), min(i1, self.n-1)
            i0s.append(i0)
            i1s.append(i1)
            # weights
            w_ = self.weights[i]
            if mode == "average":
                # move to average points
                w0 = w_ * 0.5
                w1 = w_ * 0.5
                w = w_ * -1.0
            if mode == "laplacian":
                # independent to mesh resolution, dependent to mesh size
                # possibly divide by zero
                s2 = 0.25*(points[i1]-points[i0]).length_squared()
                w0 = w_ / s2
                w1 = w_ / s2
                w = w_ * -2.0 / s2
            if mode == "normal":
                # move along normal direction
                # produces "spikes" for out-of-interval
                d0 = points[i0]-points[i]
                d1 = points[i1]-points[i]
                t = -d0.dot(d1-d0) / (d1-d0).length_squared()
                dc = d0 + (d1-d0) * t
                m = 1.0 / (d0.x*d1.y-d1.x*d0.y)
                w0 = w_ * m * (dc.x*d1.y-dc.y*d1.x)
                w1 = w_ * m * (d0.x*dc.y-d0.y*dc.x)
                w = w_ * -1.0
            w0s.append(w0)
            w1s.append(w1)
            ws.append(w)
        return (i0s, i1s, w0s, w1s, ws)

    def calc_dpdt(self) -> list[Vector2]:
        """Calculate the derivative at each point"""
        i0s, i1s, w0s, w1s, ws = self._calc_indices_weights()
        dpdt = []
        for i in range(self.n):
            p0 = self.points[i0s[i]]
            p1 = self.points[i1s[i]]
            p = self.points[i]
            lap = w0s[i]*p0 + w1s[i]*p1 + ws[i]*p
            dpdt.append(lap)
        return dpdt

    def calc_dpdt_arr(self) -> np.ndarray:
        """Same as calc_dpdt() except it returns a flattened NumPy array"""
        return np.array([[p.x, p.y] for p in self.calc_dpdt()]).flatten()

    def calc_ddpdtdp_n(self) -> np.array:
        """Numerically calculates ∂(dpdt)/∂p, returns a matrix"""
        eps = 1e-4
        p0 = self.get_points_arr()
        mat = []
        for i in range(2*self.n):
            dp = np.zeros(2*self.n)
            dp[i] = eps
            self.set_points_arr(p0+dp)
            dpdt1 = self.calc_dpdt_arr()
            self.set_points_arr(p0-dp)
            dpdt0 = self.calc_dpdt_arr()
            mat.append((dpdt1-dpdt0)/(2.0*eps))
        self.set_points_arr(p0)
        return np.array(mat).T

    def calc_ddpdtdp(self) -> scipy.sparse.coo_matrix:
        """Analytically calculate ∂(dpdt)/∂p, returns a matrix
           Note that the matrix may be singular."""
        # assumes weights are independent to p
        i0s, i1s, w0s, w1s, ws = self._calc_indices_weights()
        rows = []
        cols = []
        vals = []
        for i in range(self.n):
            # p0
            rows += [2*i, 2*i+1]
            cols += [2*i0s[i], 2*i0s[i]+1]
            vals += [w0s[i], w0s[i]]
            # p1
            rows += [2*i, 2*i+1]
            cols += [2*i1s[i], 2*i1s[i]+1]
            vals += [w1s[i], w1s[i]]
            # p
            rows += [2*i, 2*i+1]
            cols += [2*i, 2*i+1]
            vals += [ws[i], ws[i]]
        return scipy.sparse.coo_matrix((vals, (rows, cols)))

    def calc_dpdt_pv(self) -> list[Vector2]:
        """calc_dpdt() but preserves volume/area"""
        dpdt = []
        p = self.points
        for i in range(self.n):
            i02, i01, i11, i12 = i-2, i-1, i+1, i+2
            if self.closed:
                i02, i01 = i02 % self.n, i01 % self.n
                i11, i12 = i11 % self.n, i12 % self.n
            else:
                i02, i01 = max(i02, 0), max(i01, 0)
                i11, i12 = min(i11, self.n-1), min(i12, self.n-1)
            w = self.weights[i]
            lap2 = -6.0*p[i]+4.0*(p[i01]+p[i11])-(p[i02]+p[i12])
            dpdt.append(w*lap2)
        return dpdt

    def calc_dpdt_pv_arr(self) -> np.ndarray:
        """Same as calc_dpdt_pv() except it returns a flattened NumPy array"""
        return np.array([[p.x, p.y] for p in self.calc_dpdt_pv()]).flatten()

    def calc_ddpdtdp_pv_n(self) -> np.array:
        """Numerically calculate ∂(dpdt_pv)/∂p, returns a matrix"""
        eps = 1e-4
        p0 = self.get_points_arr()
        mat = []
        for i in range(2*self.n):
            dp = np.zeros(2*self.n)
            dp[i] = eps
            self.set_points_arr(p0+dp)
            dpdt1 = self.calc_dpdt_pv_arr()
            self.set_points_arr(p0-dp)
            dpdt0 = self.calc_dpdt_pv_arr()
            mat.append((dpdt1-dpdt0)/(2.0*eps))
        self.set_points_arr(p0)
        return np.array(mat).T

    def calc_ddpdtdp_pv(self) -> scipy.sparse.coo_matrix:
        """Analytically calculate ∂(dpdt_pv)/∂p, returns a matrix
           Note that the matrix may be singular."""
        rows = []
        cols = []
        vals = []
        for i in range(self.n):
            i02, i01, i11, i12 = i-2, i-1, i+1, i+2
            if self.closed:
                i02, i01 = i02 % self.n, i01 % self.n
                i11, i12 = i11 % self.n, i12 % self.n
            else:
                i02, i01 = max(i02, 0), max(i01, 0)
                i11, i12 = min(i11, self.n-1), min(i12, self.n-1)
            w = self.weights[i]
            # p
            rows += [2*i, 2*i+1]
            cols += [2*i, 2*i+1]
            vals += [-6.0*w, -6.0*w]
            # p01, p11
            rows += [2*i, 2*i+1, 2*i, 2*i+1]
            cols += [2*i01, 2*i01+1, 2*i11, 2*i11+1]
            vals += [4.0*w, 4.0*w, 4.0*w, 4.0*w]
            # p02, p12
            rows += [2*i, 2*i+1, 2*i, 2*i+1]
            cols += [2*i02, 2*i02+1, 2*i12, 2*i12+1]
            vals += [-w, -w, -w, -w]
        return scipy.sparse.coo_matrix((vals, (rows, cols)))

    def draw(self, surface: pygame.Surface, viewport: Viewport, color):
        for i in range(self.n):
            j = (i+1) % self.n
            q1 = viewport.world_to_screen(self.points[i])
            q2 = viewport.world_to_screen(self.points[j])
            if not (j == 0 and not self.closed):
                pygame.draw.aaline(surface, color, q1, q2)

    @staticmethod
    def visualize_matrix(mat, filepath):
        from PIL import Image
        if type(mat) == np.matrix:
            mat = np.array(mat)
        if type(mat) != np.ndarray:  # sparse
            mat = mat.toarray()
        mat = abs(mat)
        print(np.amax(mat))
        # mat = mat**0.1  # make small vals stand out
        mat /= np.amax(mat)
        mat = (255*mat).astype(np.uint8)
        img = Image.fromarray(mat)
        img.save(filepath)
