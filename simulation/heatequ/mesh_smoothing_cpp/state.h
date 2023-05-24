#pragma once

#include <functional>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <GL/glew.h>
#include <glm/glm.hpp>
using namespace glm;

#include "sparse.h"


class State {

	// geometry
	int n;  // number of vertices
	std::vector<vec3> vertices;
	std::vector<ivec3> triangles;
	LilMatrix neighbor_weights;
	LilMatrix neighbor_weights_pv;

public:

	State(std::vector<vec3> vertices, std::vector<ivec3> triangles);
	int getN() const {
		return this->n;
	}

	void State::smooth(float h, bool pv);

	void draw(Viewport *viewport, vec3 color) const;
};

State::State(
	std::vector<vec3> vertices, std::vector<ivec3> triangles
) : n((int)vertices.size()), neighbor_weights(n), neighbor_weights_pv(n) {
	// vertices and triangles
	this->vertices = vertices;
	this->triangles = triangles;
	std::vector<std::vector<int>> neighbors(n, std::vector<int>());
	// edges
	auto ivec2Hash = [](const ivec2& v) { return v.x*1664525+v.y; };
	std::unordered_map<ivec2, std::vector<int>, decltype(ivec2Hash)> edges(0, ivec2Hash);
	for (int i = 0; i < (int)triangles.size(); i++) {
		ivec3 v = triangles[i];
		std::sort((int*)&v, (int*)&v + 3);
		assert(v.x < v.y && v.y < v.z);
		edges[ivec2(v.x, v.y)].push_back(i);
		edges[ivec2(v.x, v.z)].push_back(i);
		edges[ivec2(v.y, v.z)].push_back(i);
	}
	// boundaries
	std::vector<bool> is_edge(n, false);
	for (std::pair<ivec2, std::vector<int>> ij_trigs : edges) {
		ivec2 ij = ij_trigs.first;
		std::vector<int> trigs = ij_trigs.second;
		if (trigs.size() == 1) {
			is_edge[ij.x] = true;
			is_edge[ij.y] = true;
			neighbors[ij.x].push_back(ij.y);
			neighbors[ij.y].push_back(ij.x);
		}
	}
	// neighbors
	for (std::pair<ivec2, std::vector<int>> ij_trigs : edges) {
		ivec2 ij = ij_trigs.first;
		std::vector<int> trigs = ij_trigs.second;
		if (trigs.size() > 1) {
			if (!is_edge[ij.x])
				neighbors[ij.x].push_back(ij.y);
			if (!is_edge[ij.y])
				neighbors[ij.y].push_back(ij.x);
		}
	}
	// weight matrix
	for (int i = 0; i < n; i++) {
		int nn = (int)neighbors[i].size();
		if (nn == 0) continue;
		neighbor_weights.addValue(i, i, -1.0f);
		for (int j : neighbors[i]) {
			assert(i != j);
			neighbor_weights.addValue(i, j, 1.0f/float(nn));
		}
	}
	// weight matrix with less volume defect
	for (int i = 0; i < n; i++) {
		for (std::pair<int, float> kw1 : neighbor_weights.mat[i]) {
			int k = kw1.first; float w1 = kw1.second;
			for (std::pair<int, float> jw2 : neighbor_weights.mat[k]) {
				int j = jw2.first; float w2 = jw2.second;
				neighbor_weights_pv.addValue(i, j, -w1*w2);
			}
		}
	}
}


void State::smooth(float h, bool pv) {
	if (h == 0.0f)
		return;
	LilMatrix lil = pv ? neighbor_weights_pv : neighbor_weights;
	vec3 *dpdt = new vec3[n];
	lil.matvecmul(&vertices[0], dpdt);
	for (int i = 0; i < n; i++)
		lil.addValue(i, i, -1.0f/h);
	CsrMatrix csr(lil);
	csr *= -1.0f;

	auto solver = lil.isSymmetric() ? &CsrMatrix::cg : &CsrMatrix::bicgstab;
	float tol = 1e-4f;  // makes difference when h is large

	float *dxdt = new float[n], *dx = new float[n];
	for (int i = 0; i < n; i++)
		dxdt[i] = dpdt[i].x, dx[i] = 0.0f;
	int xk = (csr.*solver)(dxdt, dx, n, tol);

	float *dydt = new float[n], *dy = new float[n];
	for (int i = 0; i < n; i++)
		dydt[i] = dpdt[i].y, dy[i] = 0.0f;
	int yk = (csr.*solver)(dydt, dy, n, tol);

	float *dzdt = new float[n], *dz = new float[n];
	for (int i = 0; i < n; i++)
		dzdt[i] = dpdt[i].z, dz[i] = 0.0f;
	int zk = (csr.*solver)(dzdt, dz, n, tol);

	for (int i = 0; i < n; i++)
		vertices[i] += vec3(dx[i], dy[i], dz[i]);

	delete dxdt; delete dx;
	delete dydt; delete dy;
	delete dzdt; delete dz;
	delete dpdt;
	printf("%d %d %d ", xk, yk, zk);
}


void State::draw(Viewport *viewport, vec3 color) const {

	std::vector<vec3> vertices_e;
	std::vector<vec3> normals;
	std::vector<ivec3> indices;

	for (ivec3 t : triangles) {
		int l = (int)vertices_e.size();
		vec3 v0 = vertices[t.x],
			v1 = vertices[t.y],
			v2 = vertices[t.z];
		vec3 n = normalize(cross(v1-v0, v2-v0));
		vertices_e.push_back(v0);
		vertices_e.push_back(v1);
		vertices_e.push_back(v2);
		for (int i = 0; i < 3; i++)
			normals.push_back(n);
		indices.push_back(ivec3(l, l+1, l+2));
	}

	viewport->drawVBO(
		vertices_e,
		normals,
		indices,
		color);

}
