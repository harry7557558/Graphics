
// To-do:
// General polynomial solver
// Newton's iteration in one dimension
// Multivariable numerical root-finding

// Remove debug code in solveTrigPoly() after well-tested



#ifndef __INC_ROOTFINDING_H

#define __INC_ROOTFINDING_H


#include <cmath>
#include <algorithm>

#ifndef PI
#define PI 3.1415926535897932384626
#endif



// These solvers may be accelerated a lot if manually write them inline and do simplifications according to the circumstance.
// It is recommended to perform one Newton iteration for analytical solutions when accuracy is required.


// Solve equation ax^3+bx^2+cx+d=0
// return 1: two or three real roots, r, u, v;
// return 0: one real root r and two complex roots u+vi, u-vi;
bool solveCubic(double a, double b, double c, double d, double &r, double &u, double &v) {
	b /= (3.*a), c /= a, d /= a;
	double p = (1. / 3.) * c - b * b;
	double q = (.5*c - b * b) * b - .5*d;
	a = q * q + p * p * p;
	if (a >= 0) {  // 1 root
		a = sqrt(a);
		u = q + a; u = u > 0. ? pow(u, 1. / 3.) : -pow(-u, 1. / 3.);  // u = cbrt(q + a)
		v = q - a; v = v > 0. ? pow(v, 1. / 3.) : -pow(-v, 1. / 3.);  // v = cbrt(q - a)
		r = u + v;
		v = sqrt(.75)*(u - v);
		u = -.5 * r - b;
		r -= b;
		return 0;
	}
	else {  // 3 roots
		c = pow(q*q - a, 1. / 6.);
		u = (1. / 3.)* atan(sqrt(-a) / q); if (q < 0.) u += PI / 3.;
		d = c * sin(u), c *= cos(u);
		r = 2. * c - b;
		c = -c;
		d *= sqrt(3.);
		u = c - d - b;
		v = u + 2. * d;
		return 1;
	}
}

// Find all real roots of a quartic equation, return the number of real roots
// R may not be already sorted
int solveQuartic(double k4, double k3, double k2, double k1, double k0, double R[4]) {
	// Modified from https://www.shadertoy.com/view/3lj3DW
	k3 /= (4.*k4), k2 /= (6.*k4), k1 /= (4.*k4), k0 /= k4;
	double c2 = k2 - k3 * k3;
	double c1 = k1 + k3 * (2.0*k3*k3 - 3.0*k2);
	double c0 = (1. / 3.) * (k0 + k3 * (k3*(c2 + k2)*3.0 - 4.0*k1));
	double q = c2 * c2 + c0;
	double r = c2 * c2*c2 - 3.0*c0*c2 + c1 * c1;
	double h = r * r - q * q*q;

	if (h > 0.0) {  // 2 roots
		h = sqrt(h);
		double s = r + h; s = s > 0. ? pow(s, 1. / 3.) : -pow(-s, 1. / 3.);  // s = cbrt(r + h)
		double u = r - h; u = u > 0. ? pow(u, 1. / 3.) : -pow(-u, 1. / 3.);  // u = cbrt(r - h)

		double x = s + u + 4.0*c2;
		double y = s - u;
		double ks = x * x + y * y*3.0;
		double k = sqrt(ks);

		double m = .5 *sqrt(6. / (k + x));
		double b = -2.0*c1*(k + x) / (ks + x * k) - k3;
		R[0] = -m * y + b;
		R[1] = m * y + b;
		return 2;
	}
	else {  // 4 or 0 roots
		double sQ = sqrt(q);
		double w = sQ * cos(acos(-r / (sQ*q)) / 3.0);

		double d2 = -w - c2; if (d2 < 0.0) return 0;
		double d1 = sqrt(d2);
		double h1 = sqrt(w - 2.0*c2 + c1 / d1);
		double h2 = sqrt(w - 2.0*c2 - c1 / d1);

		R[0] = -d1 - h1 - k3;
		R[1] = -d1 + h1 - k3;
		R[2] = d1 - h2 - k3;
		R[3] = d1 + h2 - k3;
		return 4;
	}
}

// Use Newton's iteration once to reduce float-point inaccuracy
double refineRoot_cubic(double c3, double c2, double c1, double c0, double x) {
	double y = c0 + x * (c1 + x * (c2 + x * c3));
	double dy = c1 + x * (2.*c2 + x * 3.*c3);
	double dx = y / dy;
	if (dx*dx < 1e-6) x -= dx;
	return x;
}
double refineRoot_quartic(double c4, double c3, double c2, double c1, double c0, double x) {
	double y = c0 + x * (c1 + x * (c2 + x * (c3 + x * c4)));
	double dy = c1 + x * (2.*c2 + x * (3.*c3 + x * 4.*c4));
	double dx = y / dy;
	if (dx*dx < 1e-4) x -= dx;
	return x;
}

// this one handles the case with zero higher coefficients
int solveQuartic_dg(double k4, double k3, double k2, double k1, double k0, double R[]) {
	if (k4 != 0.0) return solveQuartic(k4, k3, k2, k1, k0, R);
	if (k3 != 0.0) return 2 * (int)solveCubic(k3, k2, k1, k0, R[0], R[1], R[2]) + 1;
	if (k2 != 0.0) {
		double d = k1 * k1 - 4.*k0*k2;
		if (d < 0) return 0;
		d = sqrt(d); double m = -.5 / k2;
		R[0] = (k1 - d) * m;
		R[1] = (k1 + d) * m;
		return 2;
	}
	if (k1 != 0.0) {
		R[0] = -k0 / k1;
		return 1;
	}
	return -1;
}



// Solve a*cos(x)+b*sin(x)+c=0 in non-degenerated case
// solutions in [-π,π]; return false if no real solution
bool solveTrigL(double a, double b, double c, double r[2]) {
	double d = a * a + b * b - c * c;
	if (d < 0.) {
		r[0] = r[1] = NAN;
		return false;
	}
	d = sqrt(d);
	double k = 1. / (a - c);
	r[0] = 2 * atan((b + d) * k);
	r[1] = 2 * atan((b - d) * k);
	return true;
}



/* DEBUGGING */
// Solve k4*x^4+k3*x^3+k2*x^2+k1*x+k0 + c1*cos(w*x)+c2*sin(w*x) = 0
// return the first root greater than x_min; if not found, return NAN
double solveTrigPoly(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x_min) {
	// original algorithm, may have better ways
	// substitude u = w*x+phi, x=(u-phi)/w
	if (w < 0) w = -w, c2 = -c2;
	double m = sqrt(c1*c1 + c2 * c2), phi = atan2(c1, c2);  // c1*cos(w*x)+c2*sin(w*x)=m*sin(u)
	double iw = 1. / w, iw2 = iw * iw, iw3 = iw2 * iw, iw4 = iw2 * iw2;
	double phi2 = phi * phi, phi3 = phi2 * phi, phi4 = phi2 * phi2;
	double k[5] = {
		k0 - k1 * phi*iw + k2 * phi2*iw2 - k3 * phi3*iw3 + k4 * phi4*iw4,
		k1*iw - 2 * k2*phi*iw2 + 3 * k3*phi2*iw3 - 4 * k4*phi3*iw4,
		k2*iw2 - 3 * k3*phi*iw3 + 6 * k4*phi2*iw4,
		k3*iw3 - 4 * k4*phi*iw4, k4*iw4 };
	double u_min = w * x_min + phi;
	// The equation becomes: k[0]+k[1]*u+k[2]*u^2+k[3]*u^3+k[4]*u^4+m*sin(u)=0

	// least-square fitted quartic polynomial for sin phases
	// splitting [0,2π] into 4 intervals with width π/2, and translate it to fit it to [0,π/2]
	// calculated in Python using 40 decimal places, should have no problem with double
	const double S_PH[4][5] = {
		{ 0, 0.9968319281669103658219, 0.0187293883640716203885, -0.202740721717267666000, 0.0285388589493544049933 },
		{ 1, 0.0026141240860088173279, -0.514162882698694506993, 0.0234257824830134182761, 0.0285388589493544049933 },
		{ 0, -0.996831928166910365821, -0.018729388364071620388, 0.2027407217172676660005, -0.028538858949354404993 },
		{ -1, -0.002614124086008817327, 0.5141628826986945069933, -0.023425782483013418276, -0.028538858949354404993 },
	};

	static int id = 0; id++;
	printf("{type:'folder',id:'%d',title:'%d',collapsed:false},", id, k4 ? 4 : k3 ? 3 : k2 ? 2 : k1 ? 1 : k0 ? 0 : -1);
	printf("{type:'expression',folderId:'%d',latex:'%.12lgx^4%+.12lgx^3%+.12lgx^2%+.12lgx%+.12lg%+.12lg\\\\cos(%.12lgx)%+.12lg\\\\sin(%.12lgx)',color:'#000'},", id, k4, k3, k2, k1, k0, c1, w, c2, w);
	printf("{type:'expression',folderId:'%d',latex:'%.12lgx^4%+.12lgx^3%+.12lgx^2%+.12lgx%+.12lg%+.12lg\\\\sin(x)',color:'#000'},", id, k[4], k[3], k[2], k[1], k[0], m);
	printf("{type:'expression',folderId:'%d',latex:'%.12lgx^4%+.12lgx^3%+.12lgx^2%+.12lgx%+.12lg',color:'#888'},", id, k[4], k[3], k[2], k[1], k[0]);

	// find intervals that possibly have roots
	// find intervals that the absolute value of the polynomial is no more than m
	printf("{type:'expression',folderId:'%d',latex:'\\\\left|y\\\\right|=%lg',color:'#888'},", id, m);
	double B_[8];
	int N = solveQuartic_dg(k[4], k[3], k[2], k[1], k[0] - m, B_);
	N += solveQuartic_dg(k[4], k[3], k[2], k[1], k[0] + m, &B_[N]);
	if (N < 2) return NAN;  // polynomial is out of range
	if (N & 1) throw(N);  // should never happen
	std::sort(B_, B_ + N);

	// check these intervals
	typedef struct { double t0, t1; } interval;
	interval *B = (interval*)&B_[0];
	N /= 2;
	fprintf(stderr, "N=%d\n", N);
	for (int n = 0; n < N; n++) {
		printf("{type:'expression',folderId:'%d',latex:'%lg<x<%lg',color:'#a80'},", id, B[n].t0, B[n].t1);
		if (B[n].t1 > n) {
			// search roots between t0 and t1
			double t0 = B[n].t0; if (t0 < u_min) t0 = u_min;
			double t1 = B[n].t1;
			double dt = .5*PI;
			int i0 = int(floor(t0 / dt)), i1 = int(ceil(t1 / dt));
			int dbg_count = 0;
			for (int i = i0; i < i1; i++) {
				dbg_count++;
				// construct a quartic approximation of the function
				// translate the sine phase by t, and plus polynomial k
				const double* ph = S_PH[(unsigned)i % 4];
				double t = i * dt, t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
				double q[5] = {
					ph[0] - ph[1] * t + ph[2] * t2 - ph[3] * t3 + ph[4] * t4,
					ph[1] - 2 * ph[2] * t + 3 * ph[3] * t2 - 4 * ph[4] * t3,
					ph[2] - 3 * ph[3] * t + 6 * ph[4] * t2,
					ph[3] - 4 * ph[4] * t, ph[4]
				};
				for (int i = 0; i <= 4; i++) q[i] = q[i] * m + k[i];
				printf("{type:'expression',folderId:'%d',latex:'%.12lgx^4%+.12lgx^3%+.12lgx^2%+.12lgx%+.12lg',color:'#000',hidden:true},", id, q[4], q[3], q[2], q[1], q[0]);
				// solve the quartic to get an approximation of the root
				double r[4]; int nr = solveQuartic_dg(q[4], q[3], q[2], q[1], q[0], r);
				double u = INFINITY;
				for (int i = 0; i < nr; i++) {
					double ut = r[i];
					// perform Newton's iteration
					for (int iter = 0; iter < 2; iter++) {
						double v = k[0] + ut * (k[1] + ut * (k[2] + ut * (k[3] + ut * k[4]))) + m * sin(ut);
						double dv = k[1] + ut * (2.*k[2] + ut * (3.*k[3] + ut * 4.*k[4])) + m * cos(ut);
						double du = v / dv;
						if (abs(du) < dt) ut -= du;
					}
					// check if the root is legal
					double td = ut - t;
					if (td > 0. && td < dt) {
						if (td < u && ut > t0 && ut < t1) u = ut;  // this one will work
					}
				}
				if (u != INFINITY) {  // done!
					return (u - phi)*iw;  // substitude back
				}
			}
			fprintf(stderr, "%d iter(s)\n", dbg_count);  // should not exceed 4
		}
	}

	// not found
	return NAN;
}

#ifdef _DEBUG
// debug solveTrigPoly()
#include "numerical/random.h"
int main() {
	freopen("stdout.txt", "w", stdout);
	for (int T = 69; T < 100; T++) {
		fprintf(stderr, "T=%d\n", T);
		_SRAND(T);
		double k[5];
		for (int i = 0; i < 5; i++) k[i] = 0;
		int N = randi(0, 4) + 1;
		fprintf(stderr, "%d\n", N);
		for (int i = 0; i <= N; i++) k[i] = randf_n(1);
		vec2 p = rand2_n(5.0);
		double w = randf_n(2.0);
		for (unsigned i = 0; i < -1; i++) {
			double t0 = -INFINITY;
			scanf("%lf", &t0);
			printf("s=Calc.getState();s['expressions']['list']=[");
			double r = solveTrigPoly(k[4], k[3], k[2], k[1], k[0], p.x, p.y, w, t0);
			printf("];Calc.setState(s);\n");
			fprintf(stderr, "%.15lf\n", r);
			fflush(stdout);
			if (t0 == -INFINITY) break;
		}
		return 0;
	}
}
#endif


#endif // __INC_ROOTFINDING_H

