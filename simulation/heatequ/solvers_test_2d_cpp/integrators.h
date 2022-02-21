#pragma once

#include "state.h"


class Integrator {

protected:
	State* state;  // state
	int n;  // dimension of vector
	int evalCount;  // derivate evaluations per step

public:
	Integrator() {
		this->state = nullptr;
		this->n = 0;
		this->evalCount = -1;
	}
	~Integrator() {
		this->state = nullptr;
	}
	int getEvalCount() const {
		return this->evalCount;
	}
	const State* getState() const {
		return this->state;
	}

	virtual void update(float dt) = 0;
};


class Euler : public Integrator {
public:
	Euler(State* state) {
		this->state = state;
		this->n = state->getN();
		this->evalCount = 1;
	}
	~Euler() {
		this->state = nullptr;
	}

	void update(float dt) {
		float *dudt = new float[n];
		state->calcDudt(dudt);
		for (int i = 0; i < n; i++) state->u[i] += dudt[i] * dt;
		state->time += dt;
		delete dudt;
	}
};


class Midpoint : public Integrator {
public:
	Midpoint(State* state) {
		this->state = state;
		this->n = state->getN();
		this->evalCount = 2;
	}
	~Midpoint() {
		this->state = nullptr;
	}

	void update(float dt) {
		float t0 = state->time;
		float *u0 = new float[n]; state->getU(u0);
		float *p0 = new float[n]; state->calcDudt(p0);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + 0.5f*p0[i]*dt;
		state->time = t0 + 0.5f*dt;
		float *p1 = new float[n]; state->calcDudt(p1);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + p1[i]*dt;
		state->time = t0 + dt;

		delete u0; delete p0; delete p1;
	}
};


class RungeKutta : public Integrator {
public:
	RungeKutta(State* state) {
		this->state = state;
		this->n = state->getN();
		this->evalCount = 4;
	}
	~RungeKutta() {
		this->state = nullptr;
	}

	void update(float dt) {
		float t0 = state->time;
		float *u0 = new float[n]; state->getU(u0);

		float *k1 = new float[n]; state->calcDudt(k1);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + 0.5f*k1[i]*dt;
		state->time = t0 + 0.5f*dt;
		float *k2 = new float[n]; state->calcDudt(k2);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + 0.5f*k2[i]*dt;
		state->time = t0 + 0.5f*dt;
		float *k3 = new float[n]; state->calcDudt(k3);

		for (int i = 0; i < n; i++) state->u[i] = u0[i] + k3[i]*dt;
		state->time = t0 + dt;
		float *k4 = new float[n]; state->calcDudt(k4);

		for (int i = 0; i < n; i++)
			state->u[i] = u0[i] + (k1[i]+2.0f*k2[i]+2.0f*k3[i]+k4[i])*dt/6.0f;
		state->time = t0 + dt;

		delete u0;
		delete k1; delete k2; delete k3; delete k4;
	}
};
