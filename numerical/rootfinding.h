// currently only implemented solutions for cubic and quartic equations

// To-do:
// Polynomial solver
// Newton's iteration in one dimension
// Multivariable numerical root-finding
// Add solution for a*cos(x)+b*sin(x)+c=0



#ifndef __INC_ROOTFINDING_H

#define __INC_ROOTFINDING_H


#include <cmath>

#ifndef PI
#define PI 3.1415926535897932384626
#endif



// In practical use, it is recommand to write the cubic and quartic solvers inline
// and do simplifications according to particular circumstance, instead of calling the solver with coefficients.

// Also, it is recommand to perform one Newton iteration for analytical solution when accuracy is required.


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



#endif // __INC_ROOTFINDING_H

