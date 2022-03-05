#pragma once

#include <functional>

#include <GL/glew.h>
#include <glm/glm.hpp>
using namespace glm;

#include "sparse.h"


namespace Integrators {
	class Euler;
	class Midpoint;
	class RungeKutta;
	class ImplicitEuler;
}

class State {

	// parameters
	float k;  // ∂u/∂t = k * ∂²u/∂x²
	std::function<float(vec2)> initial_temp;  // initial temperature
	std::function<float(vec2, float)> heater;  // heater(xy, t)
	float time;

	// discretization
	float x0;  // minimum x
	float x1;  // maximum x
	float y0;  // minimum y
	float y1;  // maximum y
	int xn;  // number of sample points along x
	int yn;  // number of sample points along y
	int n;  // xn*yn
	float dx, dy;  // step along x, y
	vec2 *xy;  // positions
	float *u;  // temperatures

public:
	friend class Integrators::Euler;
	friend class Integrators::Midpoint;
	friend class Integrators::RungeKutta;
	friend class Integrators::ImplicitEuler;

	State() : xy(nullptr), u(nullptr) {}
	State(
		vec2 xy0, vec2 xy1, int xn, int yn,
		float k,
		std::function<float(vec2)> initial_temp,
		std::function<float(vec2, float)> heater
	);
	State::State(const State &other);
	~State();
	int getN() const {
		return this->n;
	}

	void getU(float *u) const;
	void getDudt(float *dudt) const;
	void getDpdu(LilMatrix *dpdu) const;

	void draw(Viewport *viewport, vec3 color) const;
};

State::State(
	vec2 xy0, vec2 xy1, int xn, int yn,
	float k,
	std::function<float(vec2)> initial_temp,
	std::function<float(vec2, float)> heater
) {
	// parameters
	this->k = k;
	this->initial_temp = initial_temp;
	this->heater = heater;
	this->time = 0.0f;

	// discretization
	this->x0 = xy0.x;
	this->x1 = xy1.x;
	this->y0 = xy0.y;
	this->y1 = xy1.y;
	this->xn = xn;
	this->yn = yn;
	this->n = xn * yn;
	this->dx = (x1-x0)/(xn-1);
	this->dy = (y1-y0)/(yn-1);

	// initialize
	this->xy = new vec2[xn*yn];
	this->u = new float[xn*yn];
	for (int xi = 0; xi < xn; xi++)
		for (int yi = 0; yi < yn; yi++) {
			int i = xi*yn+yi;
			xy[i] = vec2(x0+xi*dx, y0+yi*dy);
			u[i] = this->initial_temp(xy[i]);
		}
}

State::State(const State &other) {
	*this = other;
	this->xy = new vec2[n];
	this->u = new float[n];
	std::memcpy(this->xy, other.xy, sizeof(vec2)*n);
	std::memcpy(this->u, other.u, sizeof(float)*n);
}

State::~State() {
	if (xy) delete xy; xy = nullptr;
	if (u) delete u; u = nullptr;
}


void State::getU(float *u) const {
	std::memcpy(u, this->u, n*sizeof(float));
}

void State::getDudt(float *dudt) const {

	// heater
	for (int i = 0; i < n; i++)
		dudt[i] = this->heater(xy[i], time);

	// Laplacian
	for (int xi = 0; xi < xn; xi++)
		for (int yi = 0; yi < yn; yi++) {
			int xi0 = max(xi-1, 0);
			int xi1 = min(xi+1, xn-1);
			int yi0 = max(yi-1, 0);
			int yi1 = min(yi+1, yn-1);
			float u = this->u[xi*yn+yi];
			float ux0 = this->u[xi0*yn+yi];
			float ux1 = this->u[xi1*yn+yi];
			float uy0 = this->u[xi*yn+yi0];
			float uy1 = this->u[xi*yn+yi1];
			float d2udx2 = (ux0+ux1-2.0f*u) / (dx*dx);
			float d2udy2 = (uy0+uy1-2.0f*u) / (dy*dy);
			dudt[xi*yn+yi] += this->k * (d2udx2+d2udy2);
		}
}

void State::getDpdu(LilMatrix *dpdu) const {
	for (int xi = 0; xi < xn; xi++)
		for (int yi = 0; yi < yn; yi++) {
			int pi = xi*yn+yi;
			int xi0 = max(xi-1, 0);
			int xi1 = min(xi+1, xn-1);
			int yi0 = max(yi-1, 0);
			int yi1 = min(yi+1, yn-1);
			// d2udx2 = (ux0+ux1-2.0*u) / dx**2
			dpdu->addValue(pi, xi0*yn+yi, k/(dx*dx));
			dpdu->addValue(pi, xi1*yn+yi, k/(dx*dx));
			dpdu->addValue(pi, xi*yn+yi, -2.0f*k/(dx*dx));
			// d2udy2 = (uy0+uy1-2.0*u) / dy**2k/(dx*dx)
			dpdu->addValue(pi, xi*yn+yi0, k/(dy*dy));
			dpdu->addValue(pi, xi*yn+yi1, k/(dy*dy));
			dpdu->addValue(pi, xi*yn+yi, -2.0f*k/(dy*dy));
		}
}


void State::draw(Viewport *viewport, vec3 color) const {

	// setup vertice/color/indice buffers
	int m = (xn-1)*(yn-1);
	vec3 *vertices = new vec3[4*m];
	vec3 *normals = new vec3[4*m];
	ivec3 *indices = new ivec3[2*m];
	for (int xi = 0; xi < xn-1; xi++)
		for (int yi = 0; yi < yn-1; yi++) {
			// vertices
			int v0[4] = {
				xi*yn+yi,
				(xi+1)*yn+yi,
				xi*yn+(yi+1),
				(xi+1)*yn+(yi+1)
			};
			int v1 = 4*(xi*(yn-1)+yi);
			vec3 *vp = &vertices[v1];
			for (int t = 0; t < 4; t++)
				vp[t] = vec3(xy[v0[t]], u[v0[t]]);
			// normal
			vec3 *cp = &normals[v1];
			vec3 n = normalize(cross(vp[3]-vp[0], vp[2]-vp[1]));
			cp[0] = cp[1] = cp[2] = cp[3] = n;
			// indice
			ivec3 *ip = &indices[2*(xi*(yn-1)+yi)];
			ip[0] = ivec3(v1+0, v1+1, v1+3);
			ip[1] = ivec3(v1+3, v1+2, v1+0);
		}

	viewport->drawVBO(
		std::vector<vec3>(vertices, vertices+4*m),
		std::vector<vec3>(normals, normals+4*m),
		std::vector<ivec3>(indices, indices+2*m),
		color);

	delete vertices;
	delete normals;
	delete indices;
}
