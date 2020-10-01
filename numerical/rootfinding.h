
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

// In all functions: a multiple root is counted as multiple single roots


// Solve equation ax^3+bx^2+cx+d=0
// return 1: two or three real roots, r, u, v;
// return 0: one real root r ~and two complex roots u+vi, u-vi~;
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
		//v = sqrt(.75)*(u - v);
		//u = -.5 * r - b;
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
	if (k4*k4 > 1e-10) return solveQuartic(k4, k3, k2, k1, k0, R);  // the numbers become extremely large when dividing by a small coefficient
	if (k3*k3 > 1e-16) return 2 * (int)solveCubic(k3, k2, k1, k0, R[0], R[1], R[2]) + 1;
	if (k2*k2 > 1e-16) {
		double d = k1 * k1 - 4.*k0*k2;
		if (d < 0) return 0;
		d = sqrt(d); double m = -.5 / k2;
		R[0] = (k1 - d) * m;
		R[1] = (k1 + d) * m;
		return 2;
	}
	if (k1) {
		R[0] = -k0 / k1;
		return 1;
	}
	return -1;
}




// Solve an arbitrary degree polynomial using bisection method
// O(N²) or O(N²logN), numerically stable when the polynomial does not have multiple roots

// N: the degree of the polynomial
// C: lowest degree term comes first, allocate to N+1
// R: allocate to at least N for safety
// the result will be naturally sorted in increasing order
int solvePolynomial_bisect(int N, const double* C, double* R,
	double x0 = -1e18, double x1 = 1e18, double eps = 1e-15) {

	// low-degree cases
	if (N == 0)
		return 0;
	if (N == 1) {
		R[0] = -C[0] / C[1];
		return 1;
	}

	// find the roots of its derivative
	double* Cd = new double[N];
	for (int i = 1; i <= N; i++) {
		Cd[i - 1] = C[i] * i;
	}
	double *Rd = new double[N - 1];
	int NRd = solvePolynomial_bisect(N - 1, Cd, Rd, x0, x1, eps);

	// polynomial evaluation
	auto evalPolynomial = [&](const double* C, int N, double x)->double {
		double r = 0;
		for (int i = N; i >= 0; i--) r = r * x + C[i];
		return r;
	};

	// bisection search root finding
	auto binarySearch = [&](double x0, double x1)->double {
		double y0 = evalPolynomial(C, N, x0), y1 = evalPolynomial(C, N, x1);
		if ((y0 < 0) == (y1 < 0)) return NAN;
		for (int i = 0; i < 256; i++) {
			double x = 0.5 * (x0 + x1);
			double y = evalPolynomial(C, N, x);
			if ((y < 0) ^ (y0 < 0)) y1 = y, x1 = x;
			else y0 = y, x0 = x;
			if (x1 - x0 < eps) break;
		}
		return 0.5*(x0 + x1);
	};

	// roots must exist in between when sorted
	// according to differential mean value theorem
	int NR = 0;
	double r;
	if (NRd == 0) {
		if (!std::isnan(r = binarySearch(x0, x1))) R[NR++] = r;
	}
	else {
		if (!std::isnan(r = binarySearch(x0, Rd[0]))) R[NR++] = r;
		for (int i = 1; i < NRd; i++) {
			if (!std::isnan(r = binarySearch(Rd[i - 1], Rd[i]))) R[NR++] = r;
		}
		if (!std::isnan(r = binarySearch(Rd[NRd - 1], x1))) R[NR++] = r;
	}

	delete Cd; delete Rd;
	return NR;
}

// optimized for solving quintic polynomials
int solveQuintic_bisect(const double C[6], double R[5],
	double x0 = -1e18, double x1 = 1e18, double eps = 1e-15) {

	// find the roots of its derivative
	double Cd[5] = { C[1], 2.*C[2], 3.*C[3], 4.*C[4], 5.*C[5] };
	double _Rd[4]; double *Rd = _Rd;
	int NRd = solveQuartic(Cd[4], Cd[3], Cd[2], Cd[1], Cd[0], Rd);
	// sort the roots
	for (int j = NRd; j > 0; j--) {
		for (int i = 1; i < j; i++) {
			if (Rd[i - 1] > Rd[i]) {
				double t = Rd[i]; Rd[i] = Rd[i - 1], Rd[i - 1] = t;
			}
		}
	}
	//for (int i = 0; i < NRd; i++) for (int _ = 0; _ < 2; _++) Rd[i] = refineRoot_quartic(Cd[4], Cd[3], Cd[2], Cd[1], Cd[0], Rd[i]);
	while (NRd && Rd[0] < x0) Rd++, NRd--;
	while (NRd && Rd[NRd - 1] > x1) NRd--;
	/*{
		int NRd1 = solvePolynomial_bisect(4, Cd, Rd, x0, x1, eps);
		if (NRd1 != NRd) {
			for (int i = 0; i < 6; i++) fprintf(stderr, "%+.16lf*x^{%d}", C[i], i);
			fprintf(stderr, "\n"); fflush(stderr);
		}
	}*/

	// find the root between two numbers
	auto findRootBetween = [&](double x0, double x1, bool isBetweenExtremum)->double {
		double y0 = C[0] + x0 * (C[1] + x0 * (C[2] + x0 * (C[3] + x0 * (C[4] + x0 * C[5]))));
		double y1 = C[0] + x1 * (C[1] + x1 * (C[2] + x1 * (C[3] + x1 * (C[4] + x1 * C[5]))));
		if ((y0 < 0) == (y1 < 0)) return NAN;
#if 0
		// try Newton iteration
		double x;
#if 0
		if (isBetweenExtremum) {
			// start with a good initial guess
			x = y0 / (y0 - y1);
			//x = .5*(1. + pow(x, 1. / 3.) - pow(1. - x, 1. / 3.));
			//x = (((-.4731919*x + 0.70978785)*x - .190129035)*x - .0232334575) / ((x + 0.05088618)*(1.05088618 - x)) + .5;
			x = ((1.15304589*x - 1.72956883)*x + 1.33465084)*x + 0.12093605;
			x = x0 + (x1 - x0)*x;
		}
		else
#endif
			x = 0.5*(x0 + x1);
		for (int i = 0; i < 16; i++) {
			double y = C[0] + x * (C[1] + x * (C[2] + x * (C[3] + x * (C[4] + x * C[5]))));
			double dy = (((Cd[4] * x + Cd[3])*x + Cd[2])*x + Cd[1])*x + Cd[0];
			double dx = y / dy;
			x -= dx;
			if (!(x > x0 && x < x1)) break;
			if (abs(dx) < eps) return x;
		}
		// bisection search
		for (int i = 0; i < 80; i++) {
			double x = 0.5 * (x0 + x1);
			double y = C[0] + x * (C[1] + x * (C[2] + x * (C[3] + x * (C[4] + x * C[5]))));
			if ((y < 0) ^ (y0 < 0)) y1 = y, x1 = x;
			else y0 = y, x0 = x;
			if (x1 - x0 < eps) break;
		}
		return 0.5*(x0 + x1);
#else
		double x = 0.5 * (x0 + x1);
		for (int i = 0; i < 60; i++) {
			double y = C[0] + x * (C[1] + x * (C[2] + x * (C[3] + x * (C[4] + x * C[5]))));
			double dy = (((Cd[4] * x + Cd[3])*x + Cd[2])*x + Cd[1])*x + Cd[0];
			double dx = y / dy;
			double x_new = x - dx;
			if (x_new > x0 && x_new < x1) {
				x = x_new;
				if (abs(dx) < eps) return x;
			}
			else {
				if ((y < 0) ^ (y0 < 0)) y1 = y, x1 = x;
				else y0 = y, x0 = x;
				x = 0.5*(x0 + x1);
				if (x1 - x0 < eps) return x;
			}
		}
		return 0.5*(x0 + x1);
#endif
	};

	// roots must exist in between when sorted
	// according to differential mean value theorem
	int NR = 0;
	double r;
	if (NRd == 0) {
		if (!std::isnan(r = findRootBetween(x0, x1, false))) R[NR++] = r;
	}
	else {
		if (!std::isnan(r = findRootBetween(x0, Rd[0], false))) R[NR++] = r;
		for (int i = 1; i < NRd; i++) {
			if (!std::isnan(r = findRootBetween(Rd[i - 1], Rd[i], true))) R[NR++] = r;
		}
		if (!std::isnan(r = findRootBetween(Rd[NRd - 1], x1, false))) R[NR++] = r;
	}

	return NR;
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







// Solve k2*x^2+k1*x+k0 + c1*cos(w*x)+c2*sin(w*x) = 0
// return the smallest root greater than x_min, NAN if not found
// numerical solution, sometimes misses roots; may be optimized further
int solveQuadratic(double a, double b, double c, double R[2]) {  // standard quadratic solver
	if (a*a < 1e-20) {
		R[0] = -c / b;
		return 1;
	}
	double d = b * b - 4 * a*c;
	if (d < 0.) return 0;
	d = sqrt(d) / (2.*a);
	b /= -2.*a;
	R[0] = b - d, R[1] = b + d;
	return 2;
}
double solveTrigQuadratic(double k2, double k1, double k0, double c1, double c2, double w, double x_min) {
	// degenerated cases
	if (w < 0) w = -w, c2 = -c2;
	if (w*w < 1e-16) {
		double r[2];
		int nr = solveQuadratic(k2, k1, k0 + c1, r);
		double mt = NAN;
		for (int i = 0; i < nr; i++) {
			if (r[i] > x_min && !(r[i] > mt)) mt = r[i];
		}
		return mt;
	}
	if (k2*k2 < 1e-24 && k1*k1 < 1e-24) {
		double r[2];
		if (!solveTrigL(c1, c2, k0, r)) return NAN;
		x_min *= w;
		double mt = NAN;
		for (int i = 0; i < 2; i++) {
			r[i] += (2 * PI)*ceil((.5 / PI)*(x_min - r[i]));
			if (r[i] > x_min && !(r[i] > mt)) mt = r[i];
		}
		return mt / w;
	}

	double w2 = w * w;
	double m = sqrt(c1*c1 + c2 * c2);

	// split [0,2π] into 32 equal intervals
	const double SP_32[32][3] = { { 0.00000000000000000000, 1.00320083126428648821, -0.04896359868473793220 }, { 0.19509032201612826785, 0.98387634616046915379, -0.14500915241586180911 }, { 0.38268343236508977173, 0.92674204483791664154, -0.23548208574171368088 }, { 0.55557023301960222474, 0.83399356645516932392, -0.31690557457238660565 }, { 0.70710678118654752440, 0.70919518302253026482, -0.38615055989493642802 }, { 0.83146961230254523708, 0.55714282642757604459, -0.44055599575645271203 }, { 0.92387953251128675613, 0.38367978346230678067, -0.47803111176769732600 }, { 0.98078528040323044913, 0.19547214158868254887, -0.49713576023664532142 }, { 1.00000000000000000000, -0.00024738506415482473, -0.49713576023664532142 }, { 0.98078528040323044913, -0.19595740484771187072, -0.47803111176769732600 }, { 0.92387953251128675613, -0.38413689145715004898, -0.44055599575645271203 }, { 0.83146961230254523708, -0.55755421275434054939, -0.38615055989493642802 }, { 0.70710678118654752440, -0.70954503833538655616, -0.31690557457238660565 }, { 0.55557023301960222474, -0.83426844601064545436, -0.23548208574171368088 }, { 0.38268343236508977173, -0.92693138516884989407, -0.14500915241586180911 }, { 0.19509032201612826785, -0.98397287102412504440, -0.04896359868473793220 }, { -0.00000000000000000000, -1.00320083126428648821, 0.04896359868473793220 }, { -0.19509032201612826785, -0.98387634616046915379, 0.14500915241586180911 }, { -0.38268343236508977173, -0.92674204483791664154, 0.23548208574171368088 }, { -0.55557023301960222474, -0.83399356645516932392, 0.31690557457238660565 }, { -0.70710678118654752440, -0.70919518302253026482, 0.38615055989493642802 }, { -0.83146961230254523708, -0.55714282642757604459, 0.44055599575645271203 }, { -0.92387953251128675613, -0.38367978346230678067, 0.47803111176769732600 }, { -0.98078528040323044913, -0.19547214158868254887, 0.49713576023664532142 }, { -1.00000000000000000000, 0.00024738506415482473, 0.49713576023664532142 }, { -0.98078528040323044913, 0.19595740484771187072, 0.47803111176769732600 }, { -0.92387953251128675613, 0.38413689145715004898, 0.44055599575645271203 }, { -0.83146961230254523708, 0.55755421275434054939, 0.38615055989493642802 }, { -0.70710678118654752440, 0.70954503833538655616, 0.31690557457238660565 }, { -0.55557023301960222474, 0.83426844601064545436, 0.23548208574171368088 }, { -0.38268343236508977173, 0.92693138516884989407, 0.14500915241586180911 }, { -0.19509032201612826785, 0.98397287102412504440, 0.04896359868473793220 }, };
	const double CP_32[32][3] = { { 1.00000000000000000000, -0.00024738506415482473, -0.49713576023664532142 }, { 0.98078528040323044913, -0.19595740484771187072, -0.47803111176769732600 }, { 0.92387953251128675613, -0.38413689145715004898, -0.44055599575645271203 }, { 0.83146961230254523708, -0.55755421275434054939, -0.38615055989493642802 }, { 0.70710678118654752440, -0.70954503833538655616, -0.31690557457238660565 }, { 0.55557023301960222474, -0.83426844601064545436, -0.23548208574171368088 }, { 0.38268343236508977173, -0.92693138516884989407, -0.14500915241586180911 }, { 0.19509032201612826785, -0.98397287102412504440, -0.04896359868473793220 }, { -0.00000000000000000000, -1.00320083126428648821, 0.04896359868473793220 }, { -0.19509032201612826785, -0.98387634616046915379, 0.14500915241586180911 }, { -0.38268343236508977173, -0.92674204483791664154, 0.23548208574171368088 }, { -0.55557023301960222474, -0.83399356645516932392, 0.31690557457238660565 }, { -0.70710678118654752440, -0.70919518302253026482, 0.38615055989493642802 }, { -0.83146961230254523708, -0.55714282642757604459, 0.44055599575645271203 }, { -0.92387953251128675613, -0.38367978346230678067, 0.47803111176769732600 }, { -0.98078528040323044913, -0.19547214158868254887, 0.49713576023664532142 }, { -1.00000000000000000000, 0.00024738506415482473, 0.49713576023664532142 }, { -0.98078528040323044913, 0.19595740484771187072, 0.47803111176769732600 }, { -0.92387953251128675613, 0.38413689145715004898, 0.44055599575645271203 }, { -0.83146961230254523708, 0.55755421275434054939, 0.38615055989493642802 }, { -0.70710678118654752440, 0.70954503833538655616, 0.31690557457238660565 }, { -0.55557023301960222474, 0.83426844601064545436, 0.23548208574171368088 }, { -0.38268343236508977173, 0.92693138516884989407, 0.14500915241586180911 }, { -0.19509032201612826785, 0.98397287102412504440, 0.04896359868473793220 }, { 0.00000000000000000000, 1.00320083126428648821, -0.04896359868473793220 }, { 0.19509032201612826785, 0.98387634616046915379, -0.14500915241586180911 }, { 0.38268343236508977173, 0.92674204483791664154, -0.23548208574171368088 }, { 0.55557023301960222474, 0.83399356645516932392, -0.31690557457238660565 }, { 0.70710678118654752440, 0.70919518302253026482, -0.38615055989493642802 }, { 0.83146961230254523708, 0.55714282642757604459, -0.44055599575645271203 }, { 0.92387953251128675613, 0.38367978346230678067, -0.47803111176769732600 }, { 0.98078528040323044913, 0.19547214158868254887, -0.49713576023664532142 }, };

	// find the interval(s) with roots
	double _B[4];
	int N = solveQuadratic(k2, k1, k0 + m, _B);
	N += solveQuadratic(k2, k1, k0 - m, &_B[N]);
	if (N < 2) return NAN;
	std::sort(_B, _B + N);
	typedef struct { double t0, t1; } interval;
	interval *B = (interval*)&_B[0];
	N >>= 1;

	// check these intervals
	for (int n = 0; n < N; n++) {
		if (B[n].t1 > x_min) {
			// search roots between t0 and t1
			double t0 = B[n].t0; if (t0 < x_min) t0 = x_min;
			double t1 = B[n].t1;
			double dt = .0625*PI / w;
			int i0 = int(floor(t0 / dt)), i1 = int(ceil(t1 / dt));
			for (int i = i0; i < i1; i++) {
				// create a quadratic approximation of the trigonometric part
				const double *sp = SP_32[(unsigned)i % 32], *cp = CP_32[(unsigned)i % 32];
				double ph[3] = {  // trigonometric coefficients c1*cos(w*x)+c2*sin(w*x)
					c1 * cp[0] + c2 * sp[0],
					c1 * w*cp[1] + c2 * w*sp[1],
					c1 * w2*cp[2] + c2 * w2*sp[2]
				};
				// translate trigonometric polynomial and add k
				double t = i * dt;
				double q[3] = {
					k0 + ph[0] - ph[1] * t + ph[2] * t*t,
					k1 + ph[1] - 2 * ph[2] * t,
					k2 + ph[2]
				};
				// solve the quadratic to get an approximation of the root
				double r[2]; int nr = solveQuadratic(q[2], q[1], q[0], r);
				double x = INFINITY;
				for (int j = 0; j < nr; j++) {
					// Newton iteration
					double xt = r[j];
					double v, dv, dx;
					for (int iter = 0; iter < 2; iter++) {
						double wx = w * xt, cwx = cos(wx), swx = sin(wx);
						v = k0 + xt * (k1 + xt * k2) + c1 * cwx + c2 * swx;
						dv = k1 + xt * 2.*k2 - c1 * w* swx + c2 * w*cwx;
						dx = v / dv;
						if (abs(dx) < dt) xt -= dx;
						if (dx*dx < 1e-24) break;
					}
					// check if the root is legal
					if (dx*dx < 1e-7) {
						double xd = xt - t;
						if (xd > 0. && xd < dt) {
							if (xt < x && xt > t0 && xt < t1) x = xt;  // this one should work
						}
					}
				}
				if (x != INFINITY) {
					return x;
				}
			}
		}
	}
	return NAN;
}




/* DEBUGGING */
// Solve k4*x^4+k3*x^3+k2*x^2+k1*x+k0 + c1*cos(w*x)+c2*sin(w*x) = 0

// Ideas for small w:
//  - Use cubic approximations instead of quartic
//  - See how quadratic approximation works out
//  - Implement separated quadratic, cubic, and quartic solvers
//     - Should have no problem when passing zero k2 to quadratic solver
//  - Don't use the substitution solution

// return the first root greater than x_min; if not found, return NAN
double solveTrigPoly(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x_min) {
	// original algorithm, may have better ways
	// substitute u = w*x+phi, x=(u-phi)/w
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

	// least-square fitted quartic polynomial for sine phases
	// splitting [0,2π] into 4 intervals with width π/2, and translate it to fit it to [0,π/2]
	// calculated in Python using 40 decimal places, should have no problem with double
	const double S_PH[4][5] = {
		{ 0, 0.9968319281669103658219, 0.0187293883640716203885, -0.202740721717267666000, 0.0285388589493544049933 },
		{ 1, 0.0026141240860088173279, -0.514162882698694506993, 0.0234257824830134182761, 0.0285388589493544049933 },
		{ 0, -0.996831928166910365821, -0.018729388364071620388, 0.2027407217172676660005, -0.028538858949354404993 },
		{ -1, -0.002614124086008817327, 0.5141628826986945069933, -0.023425782483013418276, -0.028538858949354404993 },
	};

#ifdef _DEBUG
	static int id = 0; id++;
	printf("{type:'folder',id:'%d',title:'%d',collapsed:false},", id, k4 ? 4 : k3 ? 3 : k2 ? 2 : k1 ? 1 : k0 ? 0 : -1);
	printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf%+.12lf\\\\cos(%.12lfx)%+.12lf\\\\sin(%.12lfx)',color:'#000'},", id, k4, k3, k2, k1, k0, c1, w, c2, w);
	printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf%+.12lf\\\\sin(x)',color:'#000'},", id, k[4], k[3], k[2], k[1], k[0], m);
	printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf',color:'#888',lineStyle:'DASHED'},", id, k[4], k[3], k[2], k[1], k[0]);
#endif

	// find intervals that possibly have roots
	// find intervals that the absolute value of the polynomial is no more than m
#ifdef _DEBUG
	printf("{type:'expression',folderId:'%d',latex:'\\\\left|y\\\\right|=%lf',color:'#888'},", id, m);
#endif
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
#ifdef _DEBUG
	fprintf(stderr, "N=%d\n", N);
#endif
	for (int n = 0; n < N; n++) {
#ifdef _DEBUG
		printf("{type:'expression',folderId:'%d',latex:'%lf<x<%lf',color:'#a80'},", id, B[n].t0, B[n].t1);
#endif
		if (B[n].t1 > u_min) {
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
#ifdef _DEBUG
				printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf',color:'#000',hidden:true},", id, q[4], q[3], q[2], q[1], q[0]);
#endif
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
						if (ut < u && ut > t0 && ut < t1) u = ut;  // this one will work
					}
				}
				if (u != INFINITY) {  // done!
					// substitute back
					double x = (u - phi) / w;
					return x;
				}
			}
#ifdef _DEBUG
			fprintf(stderr, "%d iter(s)\n", dbg_count);  // should not exceed 4
#endif
		}
	}

	// not found
	return NAN;
}
// designed to work for small w; without substitution
// problem occurs when a small w encounters k4=0
// test: 0.1x slower than solveTrigPoly()
double solveTrigPoly_smallw(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x_min) {
	// the case when w=0
	if (w*w < 1e-16) {
		double r[4];
		int n = solveQuartic_dg(k4, k3, k2, k1, k0 + c1, r);
		double mx = NAN;
		for (int i = 0; i < n; i++) {
			if (r[i] > x_min && !(r[i] > mx)) mx = r[i];
		}
		return mx;
	}
	if (w < 0) w = -w, c2 = -c2;
	double w2 = w * w, w3 = w2 * w, w4 = w2 * w2;

	// splitting [0,2π] into 8 intervals with width π/4 (a more accurate approximation, hope it works)
	const double SP_8[8][5] = { { 0, 0.9997531243564286396045, 0.0028483114986132009966, -0.177251591139356185196, 0.0158188893387660080116 }, { 0.7071067811865475244008, 0.7069949604196201832864, -0.352215472731225637930, -0.123116519536749247730, 0.0381901771833280581761 }, { 1, 0.0000887371984277537483, -0.500956209912763528027, 0.0031385394583132777973, 0.0381901771833280581761 }, { 0.7071067811865475244008, -0.706869467070116661448, -0.356243593482427661890, 0.1275550846047389923689, 0.0158188893387660080116 }, { 0, -0.999753124356428639604, -0.002848311498613200996, 0.1772515911393561851965, -0.015818889338766008011 }, { -0.707106781186547524400, -0.706994960419620183286, 0.3522154727312256379308, 0.1231165195367492477308, -0.038190177183328058176 }, { -1, -0.000088737198427753748, 0.5009562099127635280279, -0.003138539458313277797, -0.038190177183328058176 }, { -0.707106781186547524400, 0.7068694670701166614489, 0.3562435934824276618900, -0.127555084604738992368, -0.015818889338766008011 }, };
	const double CP_8[8][5] = { { 1, 0.0000887371984277537483, -0.500956209912763528027, 0.0031385394583132777973, 0.0381901771833280581761 }, { 0.7071067811865475244008, -0.706869467070116661448, -0.356243593482427661890, 0.1275550846047389923689, 0.0158188893387660080116 }, { 0, -0.999753124356428639604, -0.002848311498613200996, 0.1772515911393561851965, -0.015818889338766008011 }, { -0.707106781186547524400, -0.706994960419620183286, 0.3522154727312256379308, 0.1231165195367492477308, -0.038190177183328058176 }, { -1, -0.000088737198427753748, 0.5009562099127635280279, -0.003138539458313277797, -0.038190177183328058176 }, { -0.707106781186547524400, 0.7068694670701166614489, 0.3562435934824276618900, -0.127555084604738992368, -0.015818889338766008011 }, { 0, 0.9997531243564286396045, 0.0028483114986132009966, -0.177251591139356185196, 0.0158188893387660080116 }, { 0.7071067811865475244008, 0.7069949604196201832864, -0.352215472731225637930, -0.123116519536749247730, 0.0381901771833280581761 }, };

#ifdef _DEBUG
	static int id = 0; id++;
	printf("{type:'folder',id:'%d',title:'%d',collapsed:false},", id, k4 ? 4 : k3 ? 3 : k2 ? 2 : k1 ? 1 : k0 ? 0 : -1);
	printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf%+.12lf\\\\cos(%.12lfx)%+.12lf\\\\sin(%.12lfx)',color:'#000'},", id, k4, k3, k2, k1, k0, c1, w, c2, w);
	printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf',color:'#888',lineStyle:'DASHED'},", id, k4, k3, k2, k1, k0);
#endif

	// interval finding
	double B_[8];
	double m = sqrt(c1*c1 + c2 * c2);
	int N = solveQuartic_dg(k4, k3, k2, k1, k0 - m, B_);
	N += solveQuartic_dg(k4, k3, k2, k1, k0 + m, &B_[N]);
	if (N < 2) return NAN;
	if (N & 1) throw(N);
	std::sort(B_, B_ + N);
	typedef struct { double t0, t1; } interval;
	interval *B = (interval*)&B_[0];
	N >>= 1;

#ifdef _DEBUG
	printf("{type:'expression',folderId:'%d',latex:'\\\\left|y\\\\right|=%lf',color:'#888'},", id, m);
#endif
	for (int n = 0; n < N; n++) {
#ifdef _DEBUG
		printf("{type:'expression',folderId:'%d',latex:'%lf<x<%lf',color:'#a80'},", id, B[n].t0, B[n].t1);
#endif
		if (B[n].t1 > x_min) {
			// search roots between t0 and t1
			double t0 = B[n].t0; if (t0 < x_min) t0 = x_min;
			double t1 = B[n].t1;
			double dt = .25*PI / w;
			int i0 = int(floor(t0 / dt)), i1 = int(ceil(t1 / dt));
			int dbg_count = 0;
			for (int i = i0; i < i1; i++) {
				dbg_count++;
				// create a quartic approximation of the function using quartic approximations of sin and cos
				const double *sp = SP_8[(unsigned)i % 8], *cp = CP_8[(unsigned)i % 8];
				double ph[5] = {  // trigonometric coefficients c1*cos(w*x)+c2*sin(w*x)
					c1 * cp[0] + c2 * sp[0],
					c1 * w*cp[1] + c2 * w*sp[1],
					c1 * w2*cp[2] + c2 * w2*sp[2],
					c1 * w3*cp[3] + c2 * w3*sp[3],
					c1 * w4*cp[4] + c2 * w4*sp[4]
				};
				double t = i * dt, t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
				double q[5] = {  // translate trigonometric polynomial and add k
					k0 + ph[0] - ph[1] * t + ph[2] * t2 - ph[3] * t3 + ph[4] * t4,
					k1 + ph[1] - 2 * ph[2] * t + 3 * ph[3] * t2 - 4 * ph[4] * t3,
					k2 + ph[2] - 3 * ph[3] * t + 6 * ph[4] * t2,
					k3 + ph[3] - 4 * ph[4] * t,
					k4 + ph[4]
				};
#ifdef _DEBUG
				printf("{type:'expression',folderId:'%d',latex:'%.12lfx^4%+.12lfx^3%+.12lfx^2%+.12lfx%+.12lf',color:'#000',hidden:true},", id, q[4], q[3], q[2], q[1], q[0]);
#endif
				// solve the quartic to get an approximation of the root
				double r[4]; int nr = solveQuartic_dg(q[4], q[3], q[2], q[1], q[0], r);
				double x = INFINITY;
				for (int j = 0; j < nr; j++) {
					// get the root
					double xt = r[j];
					for (int iter = 0; iter < 2; iter++) {
						double v = k0 + xt * (k1 + xt * (k2 + xt * (k3 + xt * k4))) + c1 * cos(w*xt) + c2 * sin(w*xt);
						double dv = k1 + xt * (2.*k2 + xt * (3.*k3 + xt * 4.*k4)) - c1 * w* sin(w*xt) + c2 * w*cos(w*xt);
						double dx = v / dv;
						if (abs(dx) < dt) xt -= dx;
					}
					// check if the root is legal
					double xd = xt - t;
					if (xd > 0. && xd < dt) {
						if (xt < x && xt > t0 && xt < t1) x = xt;  // this one should work
					}
				}
				if (x != INFINITY) {
					return x;
				}
			}
#ifdef _DEBUG
			fprintf(stderr, "%d iter(s)\n", dbg_count);  // should not exceed 8
#endif
		}
	}

	// not found
	return NAN;
}

// iter=1 should be enough; default value is 2
double refineRoot_TrigQuadratic(double k2, double k1, double k0, double c1, double c2, double w, double x, int iter = 2) {
	if (isnan(x)) return x;
	while (iter--) {
		double cx = cos(w*x), sx = sin(w*x);
		double y = k0 + x * (k1 + x * k2) + c1 * cx + c2 * sx;
		double dy = k1 + x * 2. * k2 + w * (-c1 * sx + c2 * cx);
		double dx = y / dy;
		if (dx*dx < 1e2) x -= dx;
		if (dx*dx < 1e-24) break;
	}
	return x;
}
double refineRoot_TrigPoly(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x, int iter = 2) {
	if (isnan(x)) return x;
	while (iter--) {
		double y = k0 + x * (k1 + x * (k2 + x * (k3 + x * k4))) + c1 * cos(w*x) + c2 * sin(w*x);
		double dy = k1 + x * (2 * k2 + x * (3 * k3 + x * 4 * k4)) - w * c1*sin(w*x) + w * c2*cos(w*x);
		double dx = y / dy;
		if (dx*dx < 1e2) x -= dx;
		if (dx*dx < 1e-24) break;
	}
	return x;
}
double solveTrigQuadratic_refine(double k2, double k1, double k0, double c1, double c2, double w, double x_min, int iter = 2) {
	double x = solveTrigQuadratic(k2, k1, k0, c1, c2, w, x_min);
	return refineRoot_TrigQuadratic(k2, k1, k0, c1, c2, w, x, iter);
}
double solveTrigQuartic_refine(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x_min, int iter = 2) {
	double x = solveTrigPoly_smallw(k4, k3, k2, k1, k0, c1, c2, w, x_min);
	return refineRoot_TrigPoly(k4, k3, k2, k1, k0, c1, c2, w, x, iter);
}




#endif // __INC_ROOTFINDING_H

