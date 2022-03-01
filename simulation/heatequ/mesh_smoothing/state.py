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
        self.recompute_draw()

    def calc_dpdt(self) -> list[Vector3]:
        """Calculate the derivative at each point"""
        dpdt = []
        for i in range(self.n):
            sdp = Vector3(0.0)
            for j in self.neighbors[i]:
                dp = self.vertices[j] - self.vertices[i]
                sdp += dp / len(self.neighbors[i])
            dpdt.append(sdp)
        return dpdt

    def calc_dpdu(self) -> scipy.sparse.base:
        """Analytically calculate ∂[dpdt]/∂p, returns a matrix"""
        pass
        rows = []
        cols = []
        nums = []
        for i in range(self.n):
            nn = len(self.neighbors[i])
            if nn == 0:
                continue
            for j in self.neighbors[i]:
                rows += [i]
                cols += [j]
                nums += [1.0 / nn]
            rows += [i]
            cols += [i]
            nums += [-1.0]
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

    def visualize_matrix(self, filepath):
        from PIL import Image
        mat = self.calc_dpdu().toarray()
        mat = abs(mat) / np.amax(abs(mat))
        mat = (255*mat).astype(np.uint8)
        img = Image.fromarray(mat)
        img.save(filepath)
