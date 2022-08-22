#pragma once

#include <random>

#include "state.h"


struct BaseSolver {
	State state;
	BaseSolver(const State &state) : state(state) { }
	virtual void update(float) = 0;
};


// Reference solver
struct EulerCromer : BaseSolver {
	EulerCromer(const State &state) : BaseSolver(state) { }
	void update(float dt);
};
void EulerCromer::update(float dt) {
	state.calcExternalAcceleration(true);
	state.calcConstraintAcceleration();
	for (int i = 0; i < (int)state.vertices.size(); i++) {
		Vertex *v = &state.vertices[i];
		v->v += v->a * dt;
		v->x += v->v * dt;
	}
	state.t += dt;
}


// Implicit non-iterative solver. Found this working.
// Might not be accurate for highly nonlinear constraints.
struct XPBDNonIterative : BaseSolver {
	XPBDNonIterative(const State &state) : BaseSolver(state) {}
	void update(float dt);
};
void XPBDNonIterative::update(float dt) {
	auto vertices = &state.vertices;
	auto triangles = &state.triangles;

	// backup initial positions
	vec3 *x0 = new vec3[vertices->size()];
	for (int i = 0; i < (int)vertices->size(); i++)
		x0[i] = state.vertices[i].x;

	// external acceleration
	state.calcExternalAcceleration(false);

	// implicitly solve for viscous drag
	for (int i = 0; i < (int)vertices->size(); i++) {
		Vertex *v = &(*vertices)[i];
		float k = v->k_drag;
		v->a -= k * v->v / (1.0f + k*dt);
	}

	// predicted positions
	for (int i = 0; i < (int)vertices->size(); i++) {
		Vertex *v = &(*vertices)[i];
		v->x += (v->v + v->a * dt) * dt;
	}

	// XPBD solver
	int *mi; Vertex* masses[3];
	float ks, kd, c; vec3 dcdx[3];
	auto xpbdStep = [&](int num_masses) {
		float kt = kd / dt + ks;
		float mc2 = 0.0f;
		for (int k = 0; k < num_masses; k++)
			mc2 += masses[k]->inv_m * dot(dcdx[k], dcdx[k]);
		for (int k = 0; k < num_masses; k++) {
			vec3 v = (masses[k]->x - x0[mi[k]]) / dt;
			float dl = -(ks * c + kd * dot(dcdx[k], v)) / (kt * mc2 + 1.0f / (dt*dt));
			masses[k]->x += masses[k]->inv_m * dcdx[k] * dl;
		}
	};

	// implicitly solve for constraints
	// Jacobi-like step is unstable. Shuffled Gauss-Seidel increases energy slightly.
	int *iter_order = new int[triangles->size()];
	for (int i = 0; i < (int)triangles->size(); i++)
		iter_order[i] = i;
	std::random_shuffle(iter_order, iter_order+(int)triangles->size());
	for (int ji = 0; ji < (int)triangles->size(); ji++) {
		int j = iter_order[ji];
		TriangleConstraint *trig = &(*triangles)[j];
		mi = trig->vi;
		for (int _ = 0; _ < 3; _++) masses[_] = &(*vertices)[mi[_]];
		state.getStretchConstraint(trig, &ks, &kd, &c, mi, dcdx);
		xpbdStep(3);
		state.getShearConstraint(trig, &ks, &kd, &c, mi, dcdx);
		xpbdStep(3);
	}
	delete iter_order;

	// update velocity and time
	for (int i = 0; i < vertices->size(); i++) {
		Vertex *v = &(*vertices)[i];
		v->v = (v->x - x0[i]) / dt;
	}
	state.t += dt;
	delete x0;
}



// XPBD: Position-Based Simulation of Compliant Constrained Dynamics
// https://matthias-research.github.io/pages/publications/XPBD.pdf

// Can't get this working for some reason.
// With damping, it does not converge to the reference solution.

#if 1

// XPBD solver
struct XPBDSolver : BaseSolver {
	XPBDSolver(const State &state) : BaseSolver(state) {}
	void update(float dt);
};
void XPBDSolver::update(float dt) {
	auto vertices = &state.vertices;
	auto triangles = &state.triangles;

	// backup initial positions
	vec3 *x0 = new vec3[vertices->size()];
	for (int i = 0; i < (int)vertices->size(); i++)
		x0[i] = state.vertices[i].x;

	// external acceleration
	state.calcExternalAcceleration(false);

	// implicitly solve for viscous drag
	for (int i = 0; i < (int)vertices->size(); i++) {
		Vertex *v = &(*vertices)[i];
		float k = v->k_drag;
		v->a -= k * v->v / (1.0f + k*dt);
	}

	// predicted positions
	for (int i = 0; i < (int)vertices->size(); i++) {
		Vertex *v = &(*vertices)[i];
		v->x += (v->v + v->a * dt) * dt;
	}

	// XPBD solver
	int *mi; Vertex* masses[3];
	float ks, kd, c; vec3 dcdx[3];
#if 1
	// assume ∇Cᵀλ = 0
	auto xpbdStep = [&](int num_masses, float &l) {
		float alpha = 1.0f / (dt * dt);
		float kt = ks + kd / dt;
		float mc2 = 0.0f, dcv = 0.0f;
		for (int k = 0; k < num_masses; k++) {
			mc2 += masses[k]->inv_m * dot(dcdx[k], dcdx[k]);
			masses[k]->v = (masses[k]->x - x0[mi[k]]) / dt;
			dcv += dot(dcdx[k], masses[k]->v);
		}
		float dl = -(ks * c + kd * dcv + alpha * l) / (kt * mc2 + alpha);
		l += dl;
		for (int k = 0; k < num_masses; k++) {
			masses[k]->x += masses[k]->inv_m * dcdx[k] * dl;
			//masses[k]->x += masses[k]->inv_m * dcdx[k] * l;
		}
	};
#else
	// ∇Cᵀλ ≠ 0
	auto xpbdStep = [&](int num_masses, float &l) {
		float alpha = 1.0f / (dt * dt);
		float kt = ks + kd / dt;
		// calculate C_t
		float dcv = 0.0f;
		for (int k = 0; k < num_masses; k++) {
			masses[k]->v = (masses[k]->x - x0[mi[k]]) / dt;
			dcv += dot(dcdx[k], masses[k]->v);
		}
		float ct = -(ks * c + kd * dcv + alpha * l);
		// calculate Δλ (Cramer's rule)
		float m[9], kc[9], cn[9], cl[9];
		for (int k = 0; k < num_masses; k++) {
			((vec3*)m)[k] = vec3(1.0f / max(masses[k]->inv_m, 1e-4f));
			((vec3*)kc)[k] = kt * dcdx[k];
			((vec3*)cn)[k] = -dcdx[k];
			((vec3*)cl)[k] = dcdx[k] * l;
		}
		float det1 = alpha, det2 = ct;
		for (int i = 0; i < 3 * num_masses; i++) {
			//if (m[i] >= 1.0f / 1.01e-4f) continue;
			float k = -kc[i] / m[i];
			kc[i] += k * m[i];
			det1 += cn[i] * m[i];
			det2 += cl[i] * m[i];
		}
		float dl = det2 / det1;
		l += dl;
		// update positions
		for (int k = 0; k < num_masses; k++) {
			masses[k]->x += masses[k]->inv_m * dcdx[k] * l;
		}
	};
#endif

	// initialize Lagrange multipliers
	float *l_trig_b = new float[triangles->size()]; // bulk
	float *l_trig_s = new float[triangles->size()]; // shear
	for (int i = 0; i < (int)triangles->size(); i++)
		l_trig_b[i] = l_trig_s[i] = 0.0f;

	// iteratively solve for constraints
	int *iter_order = new int[triangles->size()];
	for (int i = 0; i < (int)triangles->size(); i++)
		iter_order[i] = i;
	for (int i = 0; i < 8; i++) {
		// triangles
		std::random_shuffle(iter_order, iter_order+(int)triangles->size());
		for (int ji = 0; ji < (int)triangles->size(); ji++) {
			int j = iter_order[ji];
			TriangleConstraint *trig = &(*triangles)[j];
			mi = trig->vi;
			for (int _ = 0; _ < 3; _++) masses[_] = &(*vertices)[mi[_]];
			state.getStretchConstraint(trig, &ks, &kd, &c, mi, dcdx);
			xpbdStep(3, l_trig_b[j]);
			state.getShearConstraint(trig, &ks, &kd, &c, mi, dcdx);
			xpbdStep(3, l_trig_s[j]);
		}
	}
	delete iter_order;

	delete l_trig_b; delete l_trig_s;

	// update velocity and time
	for (int i = 0; i < vertices->size(); i++) {
		Vertex *v = &(*vertices)[i];
		v->v = (v->x - x0[i]) / dt;
	}
	state.t += dt;
	delete x0;
}

#endif
