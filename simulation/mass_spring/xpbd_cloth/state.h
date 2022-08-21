#pragma once

#include <functional>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
using namespace glm;


// Large Steps in Cloth Simulation
// https://www.cs.cmu.edu/~baraff/papers/sig98.pdf#Forces


struct Vertex {
	vec3 x = vec3(0.0);  // position
	vec3 v = vec3(0.0);  // velocity
	vec3 a = vec3(0.0);  // acceleration
	float inv_m = 0.0;  // reciprocal of mass
	float k_drag = 0.0;  // viscous drag coefficient, a=-kv
};


struct TriangleConstraint {
	int vi[3] = { -1, -1, -1 };  // indices of vertices
	vec2 duv1, duv2;  // span of parameter space: [1]-[0], [2]-[0]
	mat2 inv_dudv; // inverse(mat2(duv1, duv2))
	float rest_area;  // area in parameter space
	float k_stretch = 1.0, k_shear = 1.0;  // stiffness constants
	float d_stretch = 0.0, d_shear = 0.0;  // damping constants
	float bu = 1.0, bv = 1.0;  // >1 => lengthen, <1 => tighten
};


class State {
public:

	// geometry/constraints
	float t;
	std::vector<Vertex> vertices;
	std::vector<TriangleConstraint> triangles;

	// called by the constructor
	// calculates the orientation of the triangles based on the initial state
	void normalizeTriangles();

	static vec3 g;  // acceleration due to gravity

	// constructors
	State(const State &other) { *this = other; }
	State(std::vector<Vertex> vertices, std::vector<TriangleConstraint> triangles);
	State(std::vector<vec3> vertices, std::vector<ivec3> triangles,
		float tot_m, float k_drag,
		float k_stretch, float k_shear, float damp);

	// draw on viewport
	void draw(Viewport *viewport, vec3 color) const;

	// dynamics
	void calcExternalAcceleration(bool requires_drag);
	void calcConstraintAcceleration();
	void getStretchConstraint(const TriangleConstraint *pc,
		float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]);
	void getShearConstraint(const TriangleConstraint *pc,
		float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]);
	float getEnergy();
};


/* Constructors */

vec3 State::g = vec3(0, 0, -9.8);

State::State(
	std::vector<Vertex> vertices, std::vector<TriangleConstraint> triangles
) {
	this->t = 0.0f;
	this->vertices = vertices;
	this->triangles = triangles;
	this->normalizeTriangles();
}

State::State(
	std::vector<vec3> vertices, std::vector<ivec3> triangles,
	float tot_m, float k_drag = 0.1f,
	float k_stretch = 1e3f, float k_shear = 1e3f, float damp = 0.2f
) {
	this->t = 0.0f;
	this->vertices.resize(vertices.size());
	for (int i = 0; i < (int)vertices.size(); i++) {
		Vertex *v = &this->vertices[i];
		v->x = vertices[i];
		v->inv_m = tot_m / (float)vertices.size();
		v->k_drag = k_drag;
	}
	this->triangles.resize(triangles.size());
	for (int i = 0; i < (int)triangles.size(); i++) {
		TriangleConstraint *t = &this->triangles[i];
		*(ivec3*)(t->vi) = triangles[i];
		t->k_stretch = k_stretch, t->k_shear = k_shear;
		t->d_stretch = k_stretch * damp, t->d_shear = k_shear * damp;
	}
	this->normalizeTriangles();
}

void State::normalizeTriangles() {
	for (int i = 0; i < (int)triangles.size(); i++) {
		TriangleConstraint *t = &triangles[i];
		vec3 v[3];
		for (int j = 0; j < 3; j++) v[j] = vertices[t->vi[j]].x;
		vec3 du = v[1] - v[0], dv = v[2] - v[0];
		float a = acos(dot(du, dv) / (length(du)*length(dv)));
		t->duv1 = length(du) * vec2(1, 0);
		t->duv2 = length(dv) * vec2(cos(a), sin(a));
		t->inv_dudv = inverse(mat2(t->duv1, t->duv2));
		t->rest_area = 0.5f / abs(determinant(t->inv_dudv));
	}
}


/* Rendering */

void State::draw(Viewport *viewport, vec3 color) const {

	std::vector<vec3> vertices_e;
	std::vector<vec3> normals;
	std::vector<ivec3> indices;

	for (int i = 0; i < (int)triangles.size(); i++) {
		const TriangleConstraint *t = &triangles[i];
		int l = (int)vertices_e.size();
		vec3 v0 = vertices[t->vi[0]].x,
			v1 = vertices[t->vi[1]].x,
			v2 = vertices[t->vi[2]].x;
		vec3 n = normalize(cross(v1-v0, v2-v0));
		vertices_e.push_back(v0);
		vertices_e.push_back(v1);
		vertices_e.push_back(v2);
		for (int i = 0; i < 3; i++)
			normals.push_back(n);
		indices.push_back(ivec3(l, l+1, l+2));
	}

	viewport->drawVBO(vertices_e, normals, indices, color);
}


/* Dynamics */

// non-constraint acceleration
void State::calcExternalAcceleration(bool requires_drag) {
	// gravity + viscous drag
	for (int i = 0; i < (int)vertices.size(); i++) {
		Vertex *v = &vertices[i];
		v->a = v->inv_m == 0.0 ? vec3(0.0) : this->g;
		if (requires_drag) v->a -= v->k_drag * v->v;
	}
}

// constraint accelerations
void State::calcConstraintAcceleration() {
	// update constraints
	float ks, kd, c; int mi[3]; vec3 dcdx[3];
	auto updateConstraint = [&](int num_masses) {
		for (int k = 0; k < num_masses; k++) {
			Vertex *v = &vertices[mi[k]];
			vec3 fs = -ks * c * dcdx[k];
			vec3 fd = -kd * dot(dcdx[k], v->v) * dcdx[k];
			v->a += v->inv_m * (fs + fd);
		}
	};
	// triangle constraints
	for (int j = 0; j < (int)triangles.size(); j++) {
		getStretchConstraint(&triangles[j], &ks, &kd, &c, mi, dcdx);
		updateConstraint(3);
		getShearConstraint(&triangles[j], &ks, &kd, &c, mi, dcdx);
		updateConstraint(3);
	}
}

// evaluate a stretch constraint
void State::getStretchConstraint(
	const TriangleConstraint *pc,
	float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]
) {
	*ks = pc->k_stretch, *kd = pc->d_stretch;
	for (int i = 0; i < 3; i++) xi[i] = pc->vi[i];
	vec3 dx1 = vertices[xi[1]].x - vertices[xi[0]].x;
	vec3 dx2 = vertices[xi[2]].x - vertices[xi[0]].x;
	mat2x3 w = mat2x3(dx1, dx2) * pc->inv_dudv;
	vec3 wu = w[0], wv = w[1];
	float lwu = length(wu), lwv = length(wv);
	vec3 nwu = lwu==0. ? wu : wu / lwu;
	vec3 nwv = lwv==0. ? wv : wv / lwv;
	// constraint
	float a = pow(pc->rest_area, 0.0f);
	*c = a * (lwu - pc->bu + lwv - pc->bv);
	// dc/dx = dc/d|w| * d|w|/d[Δx] * d[Δx]/dx
	vec3 dcdx1 = nwu * pc->inv_dudv[0][0] + nwv * pc->inv_dudv[1][0];
	vec3 dcdx2 = nwu * pc->inv_dudv[0][1] + nwv * pc->inv_dudv[1][1];
	dcdx[0] = -a * (dcdx1 + dcdx2);
	dcdx[1] = a * dcdx1;
	dcdx[2] = a * dcdx2;
}

// evaluate a shear constraint
void State::getShearConstraint(
	const TriangleConstraint *pc,
	float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]
) {
	*ks = pc->k_stretch, *kd = pc->d_stretch;
	//*ks = *kd = 0.0f;
	for (int i = 0; i < 3; i++) xi[i] = pc->vi[i];
	vec3 dx1 = vertices[xi[1]].x - vertices[xi[0]].x;
	vec3 dx2 = vertices[xi[2]].x - vertices[xi[0]].x;
	mat2x3 w = mat2x3(dx1, dx2) * pc->inv_dudv;
	vec3 wu = w[0], wv = w[1];
	// constraint
	float a = pow(pc->rest_area, 0.0f);
	*c = a * dot(wu, wv);
	// dc/dx = dc/dw * dw/d[Δx] * d[Δx]/dx
	vec3 dcdx1 = wv * pc->inv_dudv[0][0] + wu * pc->inv_dudv[1][0];
	vec3 dcdx2 = wv * pc->inv_dudv[0][1] + wu * pc->inv_dudv[1][1];
	dcdx[0] = -a * (dcdx1 + dcdx2);
	dcdx[1] = a * dcdx1;
	dcdx[2] = a * dcdx2;
}

// get energy
float State::getEnergy() {
	// gravitational potential energy and kinetic energy
	float eg = 0.0f, ek = 0.0f;
	for (int i = 0; i < (int)vertices.size(); i++) {
		Vertex* v = &vertices[i];
		if (v->inv_m != 0.0f) {
			eg += -dot(this->g, v->x) / v->inv_m;
			ek += 0.5f / v->inv_m * dot(v->v, v->v);
		}
	}
	// constraint energy
	float ec = 0.0f;
	for (int j = 0; j < (int)triangles.size(); j++) {
		// stretch constraints
		float ks, kd, c; int xi[3]; vec3 dcdx[3];
		this->getStretchConstraint(&triangles[j], &ks, &kd, &c, xi, dcdx);
		ec += 0.5f * ks * c * c;
	}
	ec = 0.0f;
	//printf("%f %f %f\n", eg, ek, ec);
	return eg + ek + ec;
}
