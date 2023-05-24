#pragma once

#include <functional>
#include <vector>
#include <unordered_map>

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
};

struct EdgeConstraint {
	int ai[2] = { -1, -1 };  // indices of adjacent vertices
	int oi[2] = { -1, -1 };  // indices of opposite vertices
	float rest_l;  // rest length
	float inv_l;  // reciprocal of length
	float k_stretch = 1.0, k_shear = 1.0;  // stiffness constants
	float d_stretch = 0.0, d_shear = 0.0;  // damping constants
};


class State {

	// called by the constructor
	// calculates common edges shared by triangles
	void calcEdges();

	// called by the constructor
	// calculates the orientation of the triangles and edges based on the initial state
	void normalizeConstraints();

public:

	// geometry/constraints
	float t;
	std::vector<Vertex> vertices;
	std::vector<TriangleConstraint> triangles;
	std::vector<EdgeConstraint> edges;

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
	static bool enableTriangleStretchConstraint;
	static bool enableTriangleShearConstraint;
	static bool enableSpringConstraint;
	void calcExternalAcceleration(bool requires_drag);
	void getTriangleStretchConstraint(const TriangleConstraint *pc,
		float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]);
	void getTriangleShearConstraint(const TriangleConstraint *pc,
		float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]);
	void getSpringConstraint(const EdgeConstraint *pc,
		float *ks, float *kd, float *c, int xi[2], vec3 dcdx[2]);
	float getEnergy();
};


/* Constructors */

vec3 State::g = vec3(0, 0, -9.8);

#if 0
bool State::enableTriangleStretchConstraint = true;
bool State::enableTriangleShearConstraint = true;
bool State::enableSpringConstraint = false;
#else
bool State::enableTriangleStretchConstraint = false;
bool State::enableTriangleShearConstraint = false;
bool State::enableSpringConstraint = true;
#endif

State::State(
	std::vector<Vertex> vertices, std::vector<TriangleConstraint> triangles
) {
	this->t = 0.0f;
	this->vertices = vertices;
	this->triangles = triangles;
	this->calcEdges();
	this->normalizeConstraints();
}

State::State(
	std::vector<vec3> vertices, std::vector<ivec3> triangles,
	float tot_m, float k_drag = 0.1f,
	float k_stretch = 1.0f, float k_shear = 1.0f, float damp = 0.2f
) {
	this->t = 0.0f;
	// vertices
	this->vertices.resize(vertices.size());
	for (int i = 0; i < (int)vertices.size(); i++) {
		Vertex *v = &this->vertices[i];
		v->x = vertices[i];
		v->inv_m = 1.0f / (tot_m / (float)vertices.size());
		v->k_drag = k_drag;
	}
	// triangles
	this->triangles.resize(triangles.size());
	for (int i = 0; i < (int)triangles.size(); i++) {
		TriangleConstraint *t = &this->triangles[i];
		*(ivec3*)(t->vi) = triangles[i];
		t->k_stretch = k_stretch, t->k_shear = k_shear;
		t->d_stretch = k_stretch * damp, t->d_shear = k_shear * damp;
	}
	// edges
	this->calcEdges();
	for (int i = 0; i < (int)edges.size(); i++) {
		EdgeConstraint *e = &this->edges[i];
		e->k_stretch = k_stretch, e->k_shear = k_shear;
		e->d_stretch = k_stretch * damp, e->d_shear = k_shear * damp;
	}
	// normalize constraints
	this->normalizeConstraints();
}


void State::calcEdges() {
	std::unordered_map<uint64_t, ivec2> edgemap;
	for (int ti = 0; ti < (int)triangles.size(); ti++) {
		for (int vi = 0; vi < 3; vi++) {
			uint64_t v1 = (uint64_t)triangles[ti].vi[vi];
			uint64_t v2 = (uint64_t)triangles[ti].vi[(vi + 1) % 3];
			int v3 = triangles[ti].vi[(vi + 2) % 3];
			uint64_t ei = min(v1, v2) | (max(v1, v2) << 32);
			ivec2 e = ivec2(-1);
			if (edgemap.find(ei) != edgemap.end()) e = edgemap[ei];
			if (e.x == -1) e.x = v3;
			else e.y = v3;
			edgemap[ei] = e;
		}
	}
	this->edges.clear();
	this->edges.reserve(edgemap.size());
	for (std::pair<uint64_t, ivec2> edge : edgemap) {
		ivec2 ai = *(ivec2*)&edge.first;
		ivec2 oi = edge.second;
		//printf("%d %d  %d %d\n", ai.x, ai.y, oi.x, oi.y);
		EdgeConstraint ec;
		*(ivec2*)ec.ai = ai;
		*(ivec2*)ec.oi = oi;
		this->edges.push_back(ec);
	}
}

void State::normalizeConstraints() {
	// triangle constraints
	for (int i = 0; i < (int)triangles.size(); i++) {
		TriangleConstraint *t = &triangles[i];
		vec3 v[3];
		for (int j = 0; j < 3; j++) v[j] = vertices[t->vi[j]].x;
		vec3 dx1 = v[1] - v[0], dx2 = v[2] - v[0];
		vec3 e1 = normalize(dx1);
		vec3 e2 = normalize(dx2 - dot(dx2, e1) * e1);
		mat2x3 r = mat2x3(e1, e2);
		t->duv1 = dx1 * r;
		t->duv2 = dx2 * r;
		t->inv_dudv = inverse(mat2(t->duv1, t->duv2));
		t->rest_area = 0.5f / abs(determinant(t->inv_dudv));
	}
	// edge constraints
	for (int i = 0; i < (int)edges.size(); i++) {
		EdgeConstraint *e = &edges[i];
		int ai[2] = { e->ai[0], e->ai[1] };
		int oi[2] = { e->oi[0], e->oi[1] };
		vec3 dx = vertices[ai[1]].x - vertices[ai[0]].x;
		e->rest_l = length(dx);
		e->inv_l = 1.0f / e->rest_l;
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


// Calculate the strain of a deformed triangle and its gradient
void calcStrain(mat2 inv_dudv, vec3 dx1, vec3 dx2, mat2 &strain,
	vec3 &de00dx1, vec3 &de00dx2, vec3 &de11dx1, vec3 &de11dx2, vec3 &de01dx1, vec3 &de01dx2
) {
	// 3D to 2D
	vec3 e1 = normalize(dx1);
	vec3 e2d = dx2 - dot(dx2, e1) * e1;
	vec3 e2 = normalize(e2d);
	mat3x2 r = transpose(mat2x3(e1, e2));
	vec2 duv1 = r * dx1, duv2 = r * dx2;
	auto d_normalize_x_dx = [](vec3 x) -> mat3 {
		return (mat3(dot(x, x)) - outerProduct(x, x)) / pow(dot(x, x), 1.5f);
	};
	mat3 de2ddx = d_normalize_x_dx(e2d);
	mat3 de1dx1 = d_normalize_x_dx(dx1);
	mat3 de1dx2 = mat3(0);
	mat3 de2dx1 = -de1dx1 * (outerProduct(dx2, e1) + mat3(dot(e1, dx2))) * de2ddx;
	mat3 de2dx2 = (mat3(1) - outerProduct(e1, e1)) * de2ddx;
	// deformation gradient
	// dudx[0][0] = invDudv[0][0] e1 dx1 + invDudv[0][1] e1 dx2 - 1
	// dudx[1][0] = invDudv[1][0] e1 dx1 + invDudv[1][1] e1 dx2
	// dudx[0][1] = invDudv[0][0] e2 dx1 + invDudv[0][1] e2 dx2
	// dudx[1][1] = invDudv[1][0] e2 dx1 + invDudv[1][1] e2 dx2 - 1
	mat2 dudx = mat2(duv1, duv2) * inv_dudv - mat2(1);
	mat2x3 ddudxi0dx1 = mat2x3(de1dx1 * dx1 + e1, de1dx1 * dx2);  // e1 dx1, e1 dx2
	mat2x3 ddudxi1dx1 = mat2x3(de2dx1 * dx1 + e2, de2dx1 * dx2);  // e2 dx1, e2 dx2
	vec3 ddudxdx1[2][2] = {
		{ ddudxi0dx1 * inv_dudv[0], ddudxi1dx1 * inv_dudv[0] },
		{ ddudxi0dx1 * inv_dudv[1], ddudxi1dx1 * inv_dudv[1] }
	};
	mat2x3 ddudxi0dx2 = mat2x3(de1dx2 * dx1, de1dx2 * dx2 + e1);  // e1 dx1, e1 dx2
	mat2x3 ddudxi1dx2 = mat2x3(de2dx2 * dx1, de2dx2 * dx2 + e2);  // e2 dx1, e2 dx2
	vec3 ddudxdx2[2][2] = {
		{ ddudxi0dx2 * inv_dudv[0], ddudxi1dx2 * inv_dudv[0] },
		{ ddudxi0dx2 * inv_dudv[1], ddudxi1dx2 * inv_dudv[1] }
	};
	// strain tensor (symmetric)
	// 2 strain[0][0] = 2 dudx[0][0] + dudx[0][0]² + dudx[0][1]²
	// 2 strain[1][1] = 2 dudx[1][1] + dudx[1][0]² + dudx[1][1]²
	// 2 strain[0][1] = dudx[0][1] + dudx[1][0] + dudx[0][0] dudx[1][0] + dudx[0][1] dudx[1][1]
	strain = 0.5f * (transpose(dudx) + dudx + transpose(dudx) * dudx);
	de00dx1 = ddudxdx1[0][0] + dudx[0][0] * ddudxdx1[0][0] + dudx[0][1] * ddudxdx1[0][1];
	de00dx2 = ddudxdx2[0][0] + dudx[0][0] * ddudxdx2[0][0] + dudx[0][1] * ddudxdx2[0][1];
	de11dx1 = ddudxdx1[1][1] + dudx[1][1] * ddudxdx1[1][1] + dudx[1][0] * ddudxdx1[1][0];
	de11dx2 = ddudxdx2[1][1] + dudx[1][1] * ddudxdx2[1][1] + dudx[1][0] * ddudxdx2[1][0];
	de01dx1 = 0.5f * (ddudxdx1[0][1] + ddudxdx1[1][0] +
		(ddudxdx1[0][0] * dudx[1][0] + ddudxdx1[1][0] * dudx[0][0]) +
		(ddudxdx1[1][1] * dudx[0][1] + ddudxdx1[0][1] * dudx[1][1]));
	de01dx2 = 0.5f * (ddudxdx2[0][1] + ddudxdx2[1][0] +
		(ddudxdx2[0][0] * dudx[1][0] + ddudxdx2[1][0] * dudx[0][0]) +
		(ddudxdx2[1][1] * dudx[0][1] + ddudxdx2[0][1] * dudx[1][1]));
}

// Evaluate a stretch constraint
void State::getTriangleStretchConstraint(
	const TriangleConstraint *pc,
	float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]
) {
	if (!enableTriangleStretchConstraint) {
		*ks = *kd = *c = 0.0f;
		dcdx[0] = dcdx[1] = dcdx[2] = vec3(0.0f);
		return;
	}
	*ks = pc->k_stretch - (2.f/3.f)*pc->k_shear;
	*kd = pc->d_stretch - (2.f/3.f)*pc->d_shear;
	for (int i = 0; i < 3; i++) xi[i] = pc->vi[i];
	vec3 dx1 = vertices[xi[1]].x - vertices[xi[0]].x;
	vec3 dx2 = vertices[xi[2]].x - vertices[xi[0]].x;
#if 0
	// Baraff cloth simulation paper, bad
	mat2x3 w = mat2x3(dx1, dx2) * pc->inv_dudv;
	vec3 wu = w[0], wv = w[1];
	float lwu = length(wu), lwv = length(wv);
	vec3 nwu = lwu==0. ? wu : wu / lwu;
	vec3 nwv = lwv==0. ? wv : wv / lwv;
	*c = lwu - 1.0f + lwv - 1.0f;
	// dc/dx = dc/d|w| * d|w|/d[Δx] * d[Δx]/dx
	vec3 dcdx1 = nwu * pc->inv_dudv[0][0] + nwv * pc->inv_dudv[1][0];
	vec3 dcdx2 = nwu * pc->inv_dudv[0][1] + nwv * pc->inv_dudv[1][1];
#else
	// continuum formulation, slow
	mat2 strain;
	vec3 de00dx1, de00dx2, de11dx1, de11dx2, de01dx1, de01dx2;
	calcStrain(pc->inv_dudv, dx1, dx2, strain,
		de00dx1, de00dx2, de11dx1, de11dx2, de01dx1, de01dx2);
	*c = strain[0][0] + strain[1][1];
	vec3 dcdx1 = de00dx1 + de11dx1;
	vec3 dcdx2 = de00dx2 + de11dx2;
#endif
	dcdx[0] = -(dcdx1 + dcdx2);
	dcdx[1] = dcdx1;
	dcdx[2] = dcdx2;
	float a = pow(pc->rest_area, 1.0f);
	*ks *= a, *kd *= a;
}

// Evaluate a shear constraint
void State::getTriangleShearConstraint(
	const TriangleConstraint *pc,
	float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]
) {
	if (!enableTriangleShearConstraint) {
		*ks = *kd = *c = 0.0f;
		dcdx[0] = dcdx[1] = dcdx[2] = vec3(0.0f);
		return;
	}
	*ks = pc->k_shear, *kd = pc->d_shear;
	for (int i = 0; i < 3; i++) xi[i] = pc->vi[i];
	vec3 dx1 = vertices[xi[1]].x - vertices[xi[0]].x;
	vec3 dx2 = vertices[xi[2]].x - vertices[xi[0]].x;
#if 0
	// Baraff cloth simulation paper, bad
	mat2x3 w = mat2x3(dx1, dx2) * pc->inv_dudv;
	vec3 wu = w[0], wv = w[1];
	*c = dot(wu, wv);
	// dc/dx = dc/dw * dw/d[Δx] * d[Δx]/dx
	vec3 dcdx1 = wv * pc->inv_dudv[0][0] + wu * pc->inv_dudv[1][0];
	vec3 dcdx2 = wv * pc->inv_dudv[0][1] + wu * pc->inv_dudv[1][1];
#else
	// continuum formulation, slow
	mat2 strain;
	vec3 de00dx1, de00dx2, de11dx1, de11dx2, de01dx1, de01dx2;
	calcStrain(pc->inv_dudv, dx1, dx2, strain,
		de00dx1, de00dx2, de11dx1, de11dx2, de01dx1, de01dx2);
	float eu =  // raw energy term
		2.0f * strain[0][0] * strain[0][0] +
		2.0f * strain[1][1] * strain[1][1] +
		4.0f * strain[0][1] * strain[1][0];  // or ϵ[0][1]²
	*c = sqrt(max(eu, 1e-12f));
	vec3 dcdx1 = (
		2.0f * strain[0][0] * de00dx1 +
		2.0f * strain[1][1] * de11dx1 +
		4.0f * strain[0][1] * de01dx1
		) / (*c);
	vec3 dcdx2 = (
		2.0f * strain[0][0] * de00dx2 +
		2.0f * strain[1][1] * de11dx2 +
		4.0f * strain[0][1] * de01dx2
		) / (*c);
#endif
	dcdx[0] = -(dcdx1 + dcdx2);
	dcdx[1] = dcdx1;
	dcdx[2] = dcdx2;
	float a = pow(pc->rest_area, 1.0f);
	*ks *= a, *kd *= a;
}


// Hooken springs for testing
void State::getSpringConstraint(const EdgeConstraint *pc,
	float *ks, float *kd, float *c, int xi[2], vec3 dcdx[2]
) {
	if (!enableSpringConstraint) {
		*ks = *kd = *c = 0.0f;
		dcdx[0] = dcdx[1] = vec3(0.0f);
		return;
	}
	*ks = pc->k_stretch, *kd = pc->d_stretch;
	*ks *= 10.0f, *kd *= 10.0f;
	for (int i = 0; i < 2; i++) xi[i] = pc->ai[i];
	vec3 dx = vertices[xi[1]].x - vertices[xi[0]].x;
	float l = length(dx);
	*c = pc->inv_l * l - 1.0f;
	dcdx[0] = -pc->inv_l * dx / l;
	dcdx[1] = -dcdx[0];
}


// Calculate the energy of the system (constraint + gravitational + kinetic)
float State::getEnergy() {
	// gravitational potential energy and kinetic energy
	float eg = 0.0f, ek = 0.0f, totm = 0.0f;
	for (int i = 0; i < (int)vertices.size(); i++) {
		Vertex* v = &vertices[i];
		if (v->inv_m != 0.0f) {
			totm += 1.0f / v->inv_m;
			eg += -dot(this->g, v->x) / v->inv_m;
			ek += 0.5f / v->inv_m * dot(v->v, v->v);
		}
	}
	// constraint energy
	float ec = 0.0f;
	for (int j = 0; j < (int)triangles.size(); j++) {
		float ks, kd, c; int xi[3]; vec3 dcdx[3];
		this->getTriangleStretchConstraint(&triangles[j], &ks, &kd, &c, xi, dcdx);
		this->getTriangleShearConstraint(&triangles[j], &ks, &kd, &c, xi, dcdx);
		ec += 0.5f * ks * c * c;
	}
	for (int j = 0; j < (int)edges.size(); j++) {
		float ks, kd, c; int xi[2]; vec3 dcdx[2];
		this->getSpringConstraint(&edges[j], &ks, &kd, &c, xi, dcdx);
		ec += 0.5f * ks * c * c;
	}
	//printf("%f %f %f\n", eg, ek, ec);
	return eg + ec;
	return eg + ek + ec;
}
