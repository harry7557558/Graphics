from mmap import mmap
import numpy as np
import scipy.sparse
import pygame
from pygame import Vector2, Vector3
from viewport import Viewport
from OpenGL.GL import *
from OpenGL.GLU import *
import math
from copy import deepcopy
from typing import Callable


class State:

    def __init__(self,
                 vertices: list[Vector3],
                 triangles: list[tuple[int, int, int]]
                 ):
        # vertices and triangles
        self.n = len(vertices)
        self.vertices = [Vector3(p) for p in vertices]
        self.triangles = deepcopy(triangles)
        self.neighbors = [[] for _ in range(self.n)]
        # edges
        edges = {}
        for i in range(len(triangles)):
            v0, v1, v2 = sorted(triangles[i])
            assert v0 != v1 and v1 != v2
            if (v0, v1) not in edges:
                edges[(v0, v1)] = []
            if (v0, v2) not in edges:
                edges[(v0, v2)] = []
            if (v1, v2) not in edges:
                edges[(v1, v2)] = []
            edges[(v0, v1)].append(i)
            edges[(v0, v2)].append(i)
            edges[(v1, v2)].append(i)
        # boundaries
        is_edge = [False] * self.n
        for (ij, trigs) in edges.items():
            i, j = ij
            if len(trigs) == 1:
                is_edge[i] = True
                is_edge[j] = True
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)
        # neighbors
        for (ij, trigs) in edges.items():
            i, j = ij
            if len(trigs) <= 1:
                continue
            if not is_edge[i]:
                self.neighbors[i].append(j)
            if not is_edge[j]:
                self.neighbors[j].append(i)
        # misc
        self.compute_matrix()
        self.compute_matrix_pv()
        self.recompute_draw()

    def compute_matrix(self):
        """Represent dudt as the weighted sum of neighborhood vertices"""
        self.neighbor_weights = [[] for _ in range(self.n)]
        for i in range(self.n):
            nn = len(self.neighbors[i])
            if nn == 0:
                continue
            neighbors = self.neighbor_weights[i]
            neighbors.append((i, -1.0))
            for j in self.neighbors[i]:
                neighbors.append((j, 1.0/nn))
            # print(len(neighbors), end=' ')

    def compute_matrix_pv(self):
        """A version based on compute_weights() with less volume defect
           The result is the negative of the square of the matrix"""
        mat = self.neighbor_weights
        mat_r = [[] for _ in range(self.n)]
        for i in range(self.n):
            mat_ri = {}
            for (k, w1) in mat[i]:
                for (j, w2) in mat[k]:
                    if j in mat_ri:
                        mat_ri[j] -= w1*w2
                    else:
                        mat_ri[j] = -w1*w2
            for (j, w) in mat_ri.items():
                mat_r[i].append((j, w))
        self.neighbor_weights_pv = mat_r

    def calc_dpdt(self, pv: bool) -> list[Vector3]:
        """Calculate the derivative at each point"""
        mat = self.neighbor_weights_pv if pv else self.neighbor_weights
        dpdt = []
        for i in range(self.n):
            p1 = Vector3(0.0)
            for (j, w) in mat[i]:
                p1 += self.vertices[j] * w
            dpdt.append(p1)
        return dpdt

    def calc_dpdu(self, pv: bool) -> scipy.sparse.coo_matrix:
        """Calculate ∂[dpdt]/∂p, returns a matrix"""
        mat = self.neighbor_weights_pv if pv else self.neighbor_weights
        rows = []
        cols = []
        nums = []
        for i in range(self.n):
            for (j, w) in mat[i]:
                rows.append(i)
                cols.append(j)
                nums.append(w)
        return scipy.sparse.coo_matrix((nums, (rows, cols)))

    def recompute_draw(self):
        self.draw_vertices = []
        self.draw_normals = []
        self.draw_indices = []
        for i in range(len(self.triangles)):
            v0 = self.vertices[self.triangles[i][0]]
            v1 = self.vertices[self.triangles[i][1]]
            v2 = self.vertices[self.triangles[i][2]]
            self.draw_vertices += [*v0, *v1, *v2]
            n = (v1-v0).cross(v2-v0)
            self.draw_normals += [*n] * 3
            self.draw_indices += [3*i, 3*i+1, 3*i+2]

    def draw(self, viewport: Viewport, color: Vector3):
        viewport.draw_vbo(
            self.draw_vertices,
            self.draw_normals,
            self.draw_indices,
            color)

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
