#pragma once

#include "state.h"
#include "sparse.h"


class Integrator {

protected:
	std::string name;  // class name
	State* state;  // state
	int n;  // dimension of vector

public:
	Integrator() {
		this->name = "Integrator";
		this->state = nullptr;
		this->n = 0;
	}
	~Integrator() {
		this->state = nullptr;
	}
	std::string getName() const {
		return this->name;
	}
	const State* getState() const {
		return this->state;
	}

	virtual void update(float dt) = 0;
};


class Euler : public Integrator {
public:
	Euler(State* state) {
		this->name = "Euler";
		this->state = state;
		this->n = state->getN();
	}
	~Euler() {
		this->state = nullptr;
	}

	void update(float dt) {
		float *dudt = new float[n]; state->getDudt(dudt);
		for (int i = 0; i < n; i++) state->u[i] += dudt[i] * dt;
		state->time += dt;
		delete dudt;
	}
};


class Midpoint : public Integrator {
public:
	Midpoint(State* state) {
		this->name = "Midpoint";
		this->state = state;
		this->n = state->getN();
	}
	~Midpoint() {
		this->state = nullptr;
	}

	void update(float dt) {
		float t0 = state->time;
		float *u0 = new float[n]; state->getU(u0);
		float *p0 = new float[n]; state->getDudt(p0);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + 0.5f*p0[i]*dt;
		state->time = t0 + 0.5f*dt;
		float *p1 = new float[n]; state->getDudt(p1);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + p1[i]*dt;
		state->time = t0 + dt;

		delete u0; delete p0; delete p1;
	}
};


class RungeKutta : public Integrator {
public:
	RungeKutta(State* state) {
		this->name = "RungeKutta";
		this->state = state;
		this->n = state->getN();
	}
	~RungeKutta() {
		this->state = nullptr;
	}

	void update(float dt) {
		float t0 = state->time;
		float *u0 = new float[n]; state->getU(u0);

		float *k1 = new float[n]; state->getDudt(k1);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + 0.5f*k1[i]*dt;
		state->time = t0 + 0.5f*dt;
		float *k2 = new float[n]; state->getDudt(k2);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + 0.5f*k2[i]*dt;
		state->time = t0 + 0.5f*dt;
		float *k3 = new float[n]; state->getDudt(k3);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + k3[i]*dt;
		state->time = t0 + dt;
		float *k4 = new float[n]; state->getDudt(k4);

		for (int i = 0; i < n; i++)
			state->u[i] = u0[i] + (k1[i]+2.0f*k2[i]+2.0f*k3[i]+k4[i])*dt/6.0f;
		state->time = t0 + dt;

		delete u0;
		delete k1; delete k2; delete k3; delete k4;
	}
};


class ImplicitEuler : public Integrator {
public:
	ImplicitEuler(State* state) {
		this->name = "ImplicitEuler";
		this->state = state;
		this->n = state->getN();
	}
	~ImplicitEuler() {
		this->state = nullptr;
	}

	void update(float dt) {
		LilMatrix lil(n);
		for (int i = 0; i < n; i++) lil.addValue(i, i, -1.0f/dt);
		state->getDpdu(&lil);
		CsrMatrix csr(lil); csr *= -1.0;

		float *dudt = new float[n]; state->getDudt(dudt);
		float *du = new float[n];
		for (int i = 0; i < n; i++) {
			//du[i] = dudt[i] * dt;
			du[i] = 0.0f;
		}
		int iter_count = csr.cg(dudt, du, n, 0.001f);
		printf("%d/%d\n", iter_count, n);
		for (int i = 0; i < n; i++) state->u[i] += du[i];
		state->time += dt;
		delete dudt; delete du;
	}
};
