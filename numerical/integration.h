// numerical integration header/template


#ifndef __INC_INTEGRATION_H

#define __INC_INTEGRATION_H


#include "geometry.h"



/* Single-variable integrals in bounded interval */
/* Sampling + Interpolation based numerical integrators */

// standard Simpson's method, requires N to be even
// Estimated error: (b-a)⁴/180 |f"'(b)-f"'(a)| N⁻⁴
// 3rd derivative of the function needs be continuous to get the expected accuracy
template<typename T, typename Fun>
T NIntegrate_Simpson(Fun f, double a, double b, int N) {
	double dx = (b - a) / N;
	T s(0.);
	for (int i = 1; i < N; i += 2) s += f(a + i * dx);
	s *= 2;
	for (int i = 2; i < N; i += 2) s += f(a + i * dx);
	s = s * 2. + f(a) + f(b);
	return s * (dx / 3.);
}

// Gaussian quadrature with 2 samples per subinterval
// Estimated error: (b-a)⁴/270 |f"'(b)-f"'(a)| N⁻⁴
template<typename T, typename Fun>
T NIntegrate_Gaussian2(Fun f, double a, double b, int N) {
	double dx = (b - a) / (N >>= 1);
	T s(0.);
	const double d0 = 0.21132486540518711775, d1 = 0.78867513459481288225;  // (1±1/sqrt(3))/2
	for (int i = 0; i < N; i++) {
		s += f(a + (i + d0)*dx) + f(a + (i + d1)*dx);
	}
	return s * (dx / 2.);
}

// standard trapzoid method
// Estimated error: (b-a)²/12 |f'(b)-f'(a)| N⁻²
template<typename T, typename Fun>
T NIntegrate_trapzoid(Fun f, double a, double b, int N) {
	double dx = (b - a) / N;
	T s(0.);
	for (int i = 1; i < N; i++) s += f(a + i * dx);
	s += (f(a) + f(b))*.5;
	return s * dx;
}

// rectangle method, sample at midpoints
// Estimated error: (b-a)²/24 |f'(b)-f'(a)| N⁻²
// more accurate and portable than the trapzoid method
template<typename T, typename Fun>
T NIntegrate_midpoint(Fun f, double a, double b, int N) {
	double dx = (b - a) / N;
	a += .5*dx;  // midpoint
	T s(0.);
	for (int i = 0; i < N; i++) s += f(a + i * dx);
	return s * dx;
}

#if 1
// experimental method
template<typename T, typename Fun>
T NIntegrate_rect_rand(Fun f, double a, double b, int N) {
	double dx = (b - a) / N;
	T s(0.);
	unsigned seed = 1;
	for (int i = 0; i < N; i++) {
		double random = (seed = seed * 1664525u + 1013904223u) * (1. / 4294967296.);  // between 0 and 1
		s += f(a + (i + random) * dx);
	}
	return s * dx;
}
#endif




/* Line integrals respect to the arc length of parametric curves */
/* Some integrators have samples outside the interval (numerical differentiation) */
/* Those integrators (themselves) are relatively slow;
** If the function to be integrated is relatively cheap, use numerical differentiation + definite integral */

// Integral[f(t)dS, a, b], dS=length(dp/dt)dt;  O(N⁻²)
template<typename T, typename vec, typename Fun/*T(double)*/, typename fun/*vec(double)*/>
T NIntegrate_AL_midpoint_t(Fun f, fun p, double a, double b, int N) {
	double dt = (b - a) / N, t;
	a += .5*dt;
	T r(0.);
	vec p0 = p(a - dt), pc = p(a), p1;
	for (int i = 0; i < N; i++) {
		t = a + i * dt;
		p1 = p(t + dt);
		r += f(t) * length(p1 - p0);
		p0 = pc, pc = p1;
	}
	return r * .5;
}
// Integral[f(p(t))dS, a, b];  same as the previous one
template<typename T, typename vec, typename Fun/*T(vec)*/, typename fun/*vec(double)*/>
T NIntegrate_AL_midpoint_p(Fun f, fun p, double a, double b, int N) {
	double dt = (b - a) / N, t;
	a += .5*dt;
	T r(0.);
	vec p0 = p(a - dt), pc = p(a), p1;
	for (int i = 0; i < N; i++) {
		t = a + i * dt;
		p1 = p(t + dt);
		// the ↓ only thing different from the previous function
		r += f(pc) * length(p1 - p0);
		p0 = pc, pc = p1;
	}
	return r * .5;
}
// Integral[f(t)dS, a, b];  derived from Simpson, O(N⁻⁴)
template<typename T, typename vec, typename Fun/*T(double)*/, typename fun/*vec(double)*/>
T NIntegrate_AL_Simpson_t(Fun f, fun p, double a, double b, int N) {
	double dt = (b - a) / (N >>= 1);
	T r(0.);
	vec p00 = p(a - dt), pc0 = p(a - .5*dt), p0 = p(a), pc = p(a + .5*dt), p1 = p(a + dt), pc1, p11;  // position samples with delta 0.5*dt
	T v0 = f(a) * length(-p1 + pc * 8. - pc0 * 8. + p00), vc, v1;  // function value samples with delta 0.5*dt
	for (int i = 1; i <= N; i++) {
		double t = a + i * dt;
		pc1 = p(t + .5*dt), p11 = p(t + dt);
		vc = f(t - .5*dt) * length(-pc1 + p1 * 8. - p0 * 8. + pc0);  // finite difference with O(h⁴) error
		v1 = f(t) * length(-p11 + pc1 * 8. - pc * 8. + p0);
		r += v0 + vc * 4. + v1;
		p00 = p0, pc0 = pc, p0 = p1, pc = pc1, p1 = p11; v0 = v1;
	}
	return r * (1. / 36.);
}
// Integral[f(p(t))dS, a, b];
template<typename T, typename vec, typename Fun/*T(vec)*/, typename fun/*vec(double)*/>
T NIntegrate_AL_Simpson_p(Fun f, fun p, double a, double b, int N) {
	// same as the previous one except it uses f(p) instead of f(t)
	double dt = (b - a) / (N >>= 1);
	T r(0.);
	vec p00 = p(a - dt), pc0 = p(a - .5*dt), p0 = p(a), pc = p(a + .5*dt), p1 = p(a + dt), pc1, p11;
	T v0 = f(p0) * length(-p1 + pc * 8. - pc0 * 8. + p00), vc, v1;
	for (int i = 1; i <= N; i++) {
		double t = a + i * dt;
		pc1 = p(t + .5*dt), p11 = p(t + dt);
		vc = f(pc) * length(-pc1 + p1 * 8. - p0 * 8. + pc0);
		v1 = f(p1) * length(-p11 + pc1 * 8. - pc * 8. + p0);
		r += v0 + vc * 4. + v1;
		p00 = p0, pc0 = pc, p0 = p1, pc = pc1, p1 = p11; v0 = v1;
	}
	return r * (1. / 36.);
}




#endif // __INC_INTEGRATION_H
