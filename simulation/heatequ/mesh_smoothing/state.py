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
                 edges: list[tuple[int, int]],
                 triangles: list[tuple[int, int, int]],
                 vertice_weights: list[float] = None
                 ):
        # vertices and triangles
        self.n = len(vertices)
        self.vertices = [Vector3(p) for p in vertices]
        self.triangles = deepcopy(triangles)
        # vertice weights
        self.weights = [1.0] * self.n
        if vertice_weights != None:
            for i in range(self.n):
                self.weights[i] = float(vertice_weights[i])
        # neighbors
        self.neighbors = [[] for _ in range(self.n)]
        for (i, j) in edges:
            self.neighbors[i].append(j)
            self.neighbors[j].append(i)

    def calc_dpdt(self) -> list[Vector3]:
        """Calculate the derivative at each point"""
        pass
        dudt = np.zeros(self.n)
        return dudt

    def calc_dpdu_n(self) -> np.ndarray:
        """Numerically calculate the derivative of dpdt to p, returns a matrix"""
        pass

    def calc_dpdu(self) -> scipy.sparse.base:
        """Analytically calculate ∂[dpdt]/∂p, returns a matrix"""
        pass
        rows = []
        cols = []
        nums = []
        return scipy.sparse.coo_matrix((nums, (rows, cols)))

    def draw(self, viewport: Viewport, color: Vector3):
        vertices = []
        normals = []
        indices = []
        for i in range(len(self.triangles)):
            v0 = self.vertices[self.triangles[i][0]]
            v1 = self.vertices[self.triangles[i][1]]
            v2 = self.vertices[self.triangles[i][2]]
            vertices += [*v0, *v1, *v2]
            n = (v1-v0).cross(v2-v0).normalize()
            normals += [*n] * 3
            indices += [3*i, 3*i+1, 3*i+2]
        viewport.draw_vbo(vertices, normals, indices, color)

    def visualize_matrix(self, filepath):
        from PIL import Image
        mat = self.calc_dpdu().toarray()
        mat = abs(mat) / np.amax(abs(mat))
        mat = (255*mat).astype(np.uint8)
        img = Image.fromarray(mat)
        img.save(filepath)
