/* INCOMPLETE */
// function minimization experiment
// macro _DEBUG_OPTIMIZATION is defined in _debug_optimization_2d.cpp

// To-do:
// Bracket minimum in one dimension
// Conjugate gradient
// Powell's method
// Downhill simplex in high dimension
// Downhill simplex with simulated annealing


#ifndef __INC_OPTIMIZATION_H

#define __INC_OPTIMIZATION_H


#include "linearsystem.h"
#include "eigensystem.h"

#include "geometry.h"



// One-dimensional optimization
// These functions receive parameters as references instead of returning a number
// because they are often used in multivariable optimizations and the function may be expensive to evaluate.

// a,b are initial guesses, c is a point between a and b, evaled indicates whether Fa,Fb are already evaluated
// warn that if the minimum is already between a and b, this routine will increase the interval size
// not well-tested, possible have bug or infinite loop
template<typename Fun> void bracketMinimum_golden(Fun f, double &a, double &b, double *c_ = nullptr, double *Fa = nullptr, double *Fb = nullptr, double *Fc = nullptr, bool evaled = false) {
	const double gold = 1.6180339887498949;
	double fa = evaled && Fa ? *Fa : f(a);
	double fb = evaled && Fb ? *Fb : f(b);
	double c = NAN, fc = NAN;
	if (fa < fb) {
		while (1) {
			c = a, fc = fa;
			a = b - gold * (b - a), fa = f(a);
			if (fc <= fa && fc <= fb) break;
			//if (fa < fc && fc < fb) b = c, fb = fc;
		}
	}
	else {
		while (1) {
			c = b, fc = fb;
			b = a + gold * (b - a), fb = f(b);
			if (fc <= fa && fc <= fb) break;
			//if (fa > fc && fc > fb) a = c, fa = fc;
		}
	}
	if (Fa) *Fa = a; if (Fb) *Fb = b;
	if (c_) *c_ = c; if (Fc) *Fc = fc;
}

// minimize a 1d function where a minimum is bracketed x0 and x1
template<typename Fun> double GoldenSectionSearch_1d(Fun F, double &x0, double &x1, double &y0, double &y1, double eps = 1e-12) {
	const double g1 = 0.6180339887498949, g0 = 1.0 - g1;
	double t0 = g1 * x0 + g0 * x1;
	double t1 = g0 * x0 + g1 * x1;
	y0 = F(t0), y1 = F(t1);
	for (int i = 0; i < 64; i++) {
		if (y0 < y1) {
			x1 = t1, y1 = y0;
			t1 = t0, t0 = g1 * x0 + g0 * x1;
			y0 = F(t0);
		}
		else {
			x0 = t0, y0 = y1;
			t0 = t1, t1 = g0 * x0 + g1 * x1;
			y1 = F(t1);
		}
		if (x1 - x0 < eps) break;
	}
	return y0 < t1 ? t0 : t1;
}

// Brent's method for minimizing functions in 1-dimension
template<typename Fun> double Brent_minimize_1d(Fun F, double x0, double xc, double x1, double epsilon, double *minval = nullptr, int max_iter = 100) {
	// try to understand the code copied from Numerical Recipes
	// @x0,@x1: minimum is bracketed between x0 and x1
	// @x: point with least value so far
	// @w: point with second least value
	// @v: previous w
	// @u: evaluated most recently
	double x = xc > x0 && xc < x1 ? xc : 0.5*(x0 + x1), fx = F(x);
	double w = x, v = x, fw = fx, fv = fx, u, fu;
	double e = 0.;  // distance moved on the step before last
	double dx;
	for (int iter = 1; iter <= max_iter; iter++) {
		double xm = 0.5*(x0 + x1);
		double tol1 = epsilon * abs(x) + 1e-10;
		double tol2 = 2.*tol1;
		if (abs(x - xm) + .5*(x1 - x0) <= tol2) {
			if (minval) *minval = fx;
			return x;
		}
		if (abs(e) > tol1) {  // try parabolic fit
			double r = (x - w)*(fx - fv), q = (x - v)*(fx - fw);
			double p = (x - v)*q - (x - w)*r;
			q = 2.*(q - r);
			if (q > 0.) p = -p;
			q = abs(q);
			double etemp = e;
			e = dx;
			// determine if accept the parabolic fit
			if (abs(p) >= abs(.5*q*etemp) || p <= q * (x0 - x) || p >= q * (x1 - x)) {
				dx = 0.3819660112501 * (e = (x > xm ? x0 - x : x1 - x));  // golden section
			}
			else {  // accept parabolic fit
				dx = p / q;
				u = x + dx;
				if (u - x0 < tol2 || x1 - u < tol2)
					dx = xm > x ? tol1 : -tol1;
			}
		}
		else {
			// golden section
			dx = 0.3819660112501 * (e = (x >= xm ? x0 - x : x1 - x));
		}
		// newly evaluated
		u = abs(dx) > tol1 ? x + dx : x + (dx > 0. ? tol1 : -tol1);
		fu = F(u);
		// update samples
		if (fu < fx) {
			if (u >= x) x0 = x; else x1 = x;
			v = w, w = x, x = u;
			fv = fw, fw = fx, fx = fu;
		}
		else {
			if (u < x) x0 = u; else x1 = u;
			if (fu < fw || w == x) {
				v = w, fv = fw;
				w = u, fw = fu;
			}
			else if (fu <= fv || v == x || v == w) {
				v = u, fv = fu;
			}
		}
	}
	// iteration limit exceeded
	if (minval) *minval = fx;
	return x;
}






/* Numerical Differentiation */

// numerical differentiation in 2d
// not necessary when the analytical gradient is given
template<typename Fun> vec2 nGrad(Fun F, vec2 x, double e = .0001) {
	return (.5 / e) * vec2(F(x + vec2(e, 0)) - F(x - vec2(e, 0)), F(x + vec2(0, e)) - F(x - vec2(0, e)));
}
template<typename Fun> void nGrad2(Fun F, vec2 x, double &Fx, vec2 &grad, vec2 &grad2, double &dxy, double e = .0001) {
	double D[3][3];
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		D[i][j] = F(x + e * vec2(j - 1, i - 1));
	Fx = D[1][1];
	grad = (.5 / e) * vec2(D[1][2] - D[1][0], D[2][1] - D[0][1]);
	grad2 = (1. / (e*e)) * vec2(D[1][2] + D[1][0] - 2.*Fx, D[2][1] + D[0][1] - 2.*Fx);
	dxy = (.25 / (e*e)) * ((D[0][0] + D[2][2]) - (D[0][2] + D[2][0]));
}

// numerical differentiation in higher dimensions
// F: double F(const double *x);
// 2N samples
template<typename Fun> void NGrad(int N, Fun F, const double *x, double *grad, double e = .0001) {
	double *p = new double[N];
	for (int i = 0; i < N; i++) p[i] = x[i];
	for (int i = 0; i < N; i++) {
		p[i] = x[i] - e;
		double a = F(p);
		p[i] = x[i] + e;
		double b = F(p);
		p[i] = x[i];
		grad[i] = (.5 / e)*(b - a);
	}
	delete p;
}
// grad2: standard Hessian matrix
// 2N²+1 samples - an analytical derivative is highly recommended
template<typename Fun> void NGrad2(int N, Fun F, const double *x, double *Fx, double *grad, double *grad2, double e = .0001) {
	double *p = new double[N];
	for (int i = 0; i < N; i++) p[i] = x[i];
	double f = F(p); if (Fx) *Fx = f;
	for (int i = 0; i < N; i++) {
		// gradient
		p[i] = x[i] - e; double a = F(p);
		p[i] = x[i] + e; double b = F(p);
		if (grad) grad[i] = (.5 / e)*(b - a);
		// second derivative
		grad2[i*N + i] = (1. / (e*e))*(a + b - 2.*f);
		// other derivatives
		for (int j = 0; j < i; j++) {
			p[i] = x[i] + e;
			p[j] = x[j] + e; a = F(p);
			p[j] = x[j] - e; double c = F(p);
			p[i] = x[i] - e; b = F(p);
			p[j] = x[j] + e; double d = F(p);
			grad2[i*N + j] = grad2[j*N + i] = (.25 / (e*e))*((a + b) - (c + d));
			p[j] = x[j];
		}
		p[i] = x[i];
	}
	delete p;
}






/* Newton's Iteration */

// Non-standard methods based on Newton's iteration
// Fails when iterates to a point with discontinuous or zero gradient

// this method performs Newton's iteration in the gradient direction
template<typename Fun> vec2 Newton_Gradient_2d(Fun F, vec2 x0) {
	const double e = 0.0001;
	vec2 x = x0;
	double F0x = INFINITY;
	for (int i = 0; i < 10000; i++) {
		double Fx = F(x);
		double xp = F(x + vec2(e, 0)), xm = F(x - vec2(e, 0));
		double yp = F(x + vec2(0, e)), ym = F(x - vec2(0, e));
		vec2 g = (.5 / e)*vec2(xp - xm, yp - ym), eg = normalize(g);  // gradient
		double gm = (1. / (e*e)) * (F(x + e * eg) + F(x - e * eg) - 2.*Fx);  // second derivative in the gradient direction
		vec2 dx = g * abs(1. / gm);  // taking abs prevents it from reaching a maxima
		if (!(dx.sqr() > 1e-12 && abs(Fx - F0x) > 1e-8)) {
			// break the balance when it reaches a saddle point (doesn't always work)
			// this makes it generally faster but more like to stuck on valleys
			if (Fx > xp) x += vec2(e, 0);
			else if (Fx > xm) x -= vec2(e, 0);
			else if (Fx > yp) x += vec2(0, e);
			else if (Fx > ym) x -= vec2(0, e);
			// termination - too small step, too small difference, or NAN
			else if (!(dx.sqr() > 1e-16 && abs(Fx - F0x) > 1e-12)) {
				if (0.0*dot(dx, dx) == 0.0) return x - dx;
				else return x;
			}
		}
		else x -= dx;
		F0x = Fx;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 128,0,128 }));
		drawDot(x, 5, COLOR({ 128,0,128 }));
#endif
	}
	return x;
}

// this method uses the Newton-Raphson method to find a point with zero-gradient
// it might be a local maximum, a minimum, or a saddle point
template<typename Fun> vec2 Newton_Iteration_2d(Fun F, vec2 x0) {
	vec2 x = x0;
	double z0 = INFINITY, z;
	for (int i = 0; i < 10000; i++) {
		vec2 g, g2; double gxy;
		nGrad2(F, x, z, g, g2, gxy, 0.0001);
		double m = 1. / (g2.x*g2.y - gxy * gxy);
		vec2 dx = m * vec2(g.x*g2.y - g.y*gxy, g2.x*g.y - g.x*gxy);
		if (!(dx.sqr() > 1e-16 && abs(z - z0) > 1e-12)) {
			if (0.0*dx.x == 0.0*dx.y) return x - dx;
			return x;
		}
		x -= dx;
		z0 = z;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 128,128,0 }));
		drawDot(x, 5, COLOR({ 128,128,0 }));
#endif
	}
	return x;
}

// this version doesn't get maxima and saddle but is slower
// ideas of improvement include checking signs of eigenvalues and detecting infinite loops
template<typename Fun> vec2 Newton_Iteration_2d_(Fun F, vec2 x0) {
	const double e = 0.0001;
	vec2 x = x0;
	double z0 = INFINITY, z;
	for (int i = 0; i < 10000; i++) {
		// numerical differentiation
		double D[3][3];
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			D[i][j] = F(x + e * vec2(j - 1, i - 1));
		z = D[1][1];
		vec2 g = (.5 / e) * vec2(D[1][2] - D[1][0], D[2][1] - D[0][1]);
		vec2 g2 = (1. / (e*e)) * vec2(D[1][2] + D[1][0] - 2.*z, D[2][1] + D[0][1] - 2.*z);
		double gxy = (.25 / (e*e)) * ((D[0][0] + D[2][2]) - (D[0][2] + D[2][0]));
		// solve the linear system
		double m = 1. / (g2.x*g2.y - gxy * gxy);
		vec2 dx = m * vec2(g.x*g2.y - g.y*gxy, g2.x*g.y - g.x*gxy);
		if (2.*dot(dx, g) < dot(dx, vec2(g2.x*dx.x + gxy * dx.y, gxy*dx.x + g2.y*dx.y))) dx = -dx;  // prevent reaching maxima
		if (!(dx.sqr() > 1e-12 && abs(z - z0) > 1e-8)) {
			// prevent reaching saddle points
			// this may cause it to enter an infinite loop
			int mi = -1, mj = -1;
			double mz = z;
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
				if (D[i][j] < mz) mz = D[i][j], mi = i, mj = j;
			if (mi != -1) x += e * vec2(mj - 1, mi - 1);
			else if (!(dx.sqr() > 1e-16 && abs(z - z0) > 1e-12)) {  // termination
				if (0.0*dx.x == 0.0*dx.y) return x - dx;
				else return x;
			}
		}
		if (0.0*dx.x == 0.0*dx.y) x -= dx;
		else break;
		z0 = z;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 160,40,40 }));
		drawDot(x, 5, COLOR({ 160,40,40 }));
#endif
	}
	return x;
}

// same function as Newton_Iteration_2d but uses analytical gradient, faster and more stable
// F: double F(vec2 x, vec2 *grad, vec2 *grad2, double *dxy);
template<typename Fun> vec2 Newton_Iteration_2d_ad(Fun F, vec2 x0) {
	vec2 x = x0;
	double z0 = INFINITY, z;
	for (int i = 0; i < 10000; i++) {
		vec2 g, g2; double gxy;
		z = F(x, &g, &g2, &gxy);
		// calculate Newton step
		double m = 1. / (g2.x*g2.y - gxy * gxy);
		vec2 dx = m * vec2(g.x*g2.y - g.y*gxy, g2.x*g.y - g.x*gxy);
		// make sure it will reach a minimum
		if (dot(dx, vec2(g2.x*dx.x + gxy * dx.y, gxy*dx.x + g2.y*dx.y)) > 2.*dot(dx, g)) dx = -dx;  // if not decent direction, go opposite
		else if (dx.sqr() < 1e-12 && g2.x*g2.y < gxy*gxy) {  // break the balance when it converges to a saddle point
			x -= dx; dx = vec2(0.0);
			x += max(length(dx), 1e-6)*normalize(vec2(-g2.x + g2.y + sqrt(dot(g2, g2) - 2.*g2.x*g2.y + 4.*gxy*gxy), -2.*gxy));
		}
		// test termination condition
		else if (!(dx.sqr() > 1e-16 && abs(z - z0) > 1e-12)) {  // Note that sometimes this terminates early due to coincidence
			if (0.0*dx.x == 0.0*dx.y) return x - dx;
			return x;
		}
		x -= dx;
		z0 = z;
	}
	return x;
}

// Optimization in higher dimensions

// one iteration: O(N²) samples, O(N³) complexity; quadratic convergence
// return true when (possible) succeed
template<typename Fun> bool Newton_Iteration_Minimize(int N, Fun F, const double* x0, double* xm, bool checkSaddle = false, int MAX_ITER = 10000) {
	if (xm != x0) for (int i = 0; i < N; i++) xm[i] = x0[i];
	double *g = new double[N], *g2 = new double[N*N];
	double *dx = new double[N];
	double y0 = INFINITY, y;
	bool converging = false;
	for (int i = 0; i < MAX_ITER; i++) {
		NGrad2(N, F, xm, &y, g, g2);
		for (int i = 0; i < N; i++) dx[i] = g[i];
		solveLinear(N, g2, dx);
		if (quamul(N, dx, g2, dx) > 2.0*vecdot(N, dx, g)) {  // make sure it is a decent direction
			for (int i = 0; i < N; i++) dx[i] = -dx[i];
		}
		double m = 0; for (int i = 0; i < N; i++) m += dx[i] * dx[i];
		if (0.0*m == 0.0) {
			for (int i = 0; i < N; i++) xm[i] -= dx[i];
		}
		if (!(m > 1e-12 && abs(y - y0) > 1e-8)) {
			if (!converging) {
				if (checkSaddle) {  // test positive-definiteness (slow)
					EigenPairs_Jacobi(N, g2, g, g2);
					double mg = INFINITY; int mi = -1;
					for (int i = 0; i < N; i++) {
						if (g[i] < mg) mg = g[i], mi = i;
					}
					if (mg < 0) {  // break the balance when it reaches a saddle point
						m = max(sqrt(m), 1e-4);
						for (int i = 0; i < N; i++) xm[i] -= m * g2[mi*N + i];
					}
					else converging = true;
				}
				else converging = true;
			}
			if (converging && !(m > 1e-16 && abs(y - y0) > 1e-12)) {  // termination
				delete g; delete g2; delete dx;
				return true;
			}
		}
		y0 = y;
	}
	delete g; delete g2; delete dx;
	return false;
}






/* Downhill Simplex */

// fun(vec2): function to minimize
// P0: initial simplex/triangle
// Breaks when the optimizer makes no improvement more than accur_eps for more than noimporv_break times
// or iteration steps exceeds max_iter
// Each iteration step requires 2 or 1 function evaluations
template<typename Fun> vec2 downhillSimplex_2d(Fun fun, vec2 P0[3],
	double accur_eps = 1e-6, int noimporv_break = 10, int max_iter = 1000) {

	struct sample {
		vec2 p;
		double val;
	} S[3];
	for (int i = 0; i < 3; i++) {
		S[i].p = P0[i];
		S[i].val = fun(S[i].p);
	}

	double old_minval = INFINITY;
	int noimporv_count = 0;

	for (int iter = 0; iter < max_iter; iter++) {

		// sort in increasing order
		sample temp;
		if (S[0].val > S[1].val) temp = S[0], S[0] = S[1], S[1] = temp;
		if (S[1].val > S[2].val) temp = S[1], S[1] = S[2], S[2] = temp;
		if (S[0].val > S[1].val) temp = S[0], S[0] = S[1], S[1] = temp;

#ifdef _DEBUG_OPTIMIZATION
		// debug output
		drawLine(S[0].p, S[1].p, COLOR{ 0,120,0 });
		drawLine(S[1].p, S[2].p, COLOR{ 0,120,0 });
		drawLine(S[2].p, S[0].p, COLOR{ 0,120,0 });
#endif

		// termination condition
		if (S[0].val < old_minval - accur_eps) {
			noimporv_count = 0;
			old_minval = S[0].val;
		}
		else if (++noimporv_count > noimporv_break) {
			return S[0].p;
		}

		// reflection
		sample refl;
		vec2 center = (S[0].p + S[1].p) * .5;
		refl.p = center * 2. - S[2].p;
		refl.val = fun(refl.p);
#ifdef _DEBUG_OPTIMIZATION
		drawDot(refl.p, 2, COLOR{ 0,120,0 });
#endif
		if (refl.val >= S[0].val && refl.val < S[1].val) {
			S[2] = refl;
			continue;
		}

		// expansion
		if (refl.val < S[0].val) {
			sample expd;
			expd.p = center + (center - S[2].p)*2.;
			expd.val = fun(expd.p);
#ifdef _DEBUG_OPTIMIZATION
			drawDot(expd.p, 2, COLOR{ 0,120,0 });
#endif
			if (expd.val < refl.val)
				S[2] = expd;
			else
				S[2] = refl;
			continue;
		}

		// contraction
		sample ctrct;
		ctrct.p = center + .5*(S[2].p - center);
		ctrct.val = fun(ctrct.p);
#ifdef _DEBUG_OPTIMIZATION
		drawDot(ctrct.p, 2, COLOR{ 0,120,0 });
#endif
		if (ctrct.val < S[2].val) {
			S[2] = ctrct;
			continue;
		}

		// compression
		S[1].p = S[0].p + (S[1].p - S[0].p)*.5;
		S[2].p = S[0].p + (S[2].p - S[0].p)*.5;
		S[1].val = fun(S[1].p), S[2].val = fun(S[2].p); // may only need 1 evals?
#ifdef _DEBUG_OPTIMIZATION
		drawDot(S[1].p, 2, COLOR{ 0,120,0 });
		drawDot(S[2].p, 2, COLOR{ 0,120,0 });
#endif
	}

	return S[0].val < S[1].val && S[0].val < S[2].val ? S[0].p
		: S[1].val < S[2].val ? S[1].p : S[2].p;
}


// K: number of dimensions, usually >2
// @fun(double[K]): function to minimize
// @P[K+1][K]: initial simplex; an array of double* that can be swapped easily
// @val[K+1]: initial values
// @initialized: set to false if val is not initialized
// @least_square: set this to true if it is minimizing a least-square loss function (different termination condition, see code for details)
// @terminate_count_break: break if the number of consecutive iterations with termination condition satisfied reaches this number
// @refl_c,expd_c,ctrct_c,shrnk_c: factors of reflection/expansion/contraction/shrinking
// return the index to the minimum value; Eg. fun(P[id])=val(id)
template<typename Fun> int downhillSimplex(
	int K, Fun fun, double* const P[], double val[], bool initialized,
	double epsilon = 1e-8, bool least_square = false, int terminate_count_break = 1, int min_iter = 10, int max_iter = 2000,
	double refl_c = NAN, double expd_c = NAN, double ctrct_c = NAN, double shrnk_c = NAN) {

	// suggested parameters
	// https://www.webpages.uidaho.edu/~fuchang/res/ANMS.pdf
	if (!(refl_c > 0.)) refl_c = 1.;
	if (!(expd_c > 0.)) expd_c = 1. + 2. / K;
	if (!(ctrct_c > 0.)) ctrct_c = .75 - .5 / K;
	if (!(shrnk_c > 0.)) shrnk_c = 1. - 1. / K;

	// initialize val[]
	if (!initialized) {
		for (int i = 0; i <= K; i++) {
			val[i] = fun(P[i]);
		}
	}
	double old_minval = INFINITY;
	int terminate_count = 0;

	// temp variables - I may not need this many
	double *ctr = new double[K],
		*refl = new double[K],
		*expd = new double[K],
		*ctrct = new double[K];

	// iteration
	for (int iter = 0; ; iter++) {

		// find the lowest, highest, and second highest points
		// (there is no need for sorting/re-ordering)
		int low_id, high_id, high2_id;
		high_id = val[0] > val[1] ? (low_id = high2_id = 1, 0) : (low_id = high2_id = 0, 1);
		for (int i = 2; i <= K; i++) {
			if (val[i] <= val[low_id]) low_id = i;
			if (val[i] > val[high_id]) high2_id = high_id, high_id = i;
			else if (val[i] > val[high2_id]) high2_id = i;
		}
		// no NAN please!
		if (low_id == high_id || low_id == high2_id || high_id == high2_id) throw("bug!");

#if 0
		// debug output
		printf("%d %lf\n", iter, val[low_id]);
		if (K == 2) STL.push_back(stl_triangle(
			vec3(P[0][0], P[0][1], val[0]),
			vec3(P[1][0], P[1][1], val[1]),
			vec3(P[2][0], P[2][1], val[2]),
			ColorFunctions::DarkBands(0.5*(cos(0.5*iter) + 1.))));
		auto addTriangle = [&](double* A, double* B, double* C) {
			STL.push_back(stl_triangle(*(vec3*)A, *(vec3*)B, *(vec3*)C,
				ColorFunctions::Rainbow(0.5*(sin(0.4*iter) + 1.))));
		};
		if (K == 3)
			addTriangle(P[0], P[1], P[2]),
			addTriangle(P[0], P[1], P[3]),
			addTriangle(P[0], P[2], P[3]),
			addTriangle(P[1], P[2], P[3]);
#endif

		// termination condition
		if (iter > min_iter &&
			(((least_square ?
				2.*(val[high_id] - val[low_id]) < epsilon * (val[high_id] + val[low_id] + 1e-4*epsilon)  // from Numerical Recipes
				: abs(val[high_id] - val[low_id]) < epsilon
				) ? true : terminate_count = 0) && ++terminate_count >= terminate_count_break)
			|| iter > max_iter) {
			delete ctr; delete refl; delete expd; delete ctrct;
			return low_id;
		}

		// centroid
		for (int i = 0; i < K; i++) ctr[i] = 0;
		for (int i = 0; i <= K; i++) if (i != high_id) {
			for (int _ = 0; _ < K; _++) ctr[_] += P[i][_];
		}
		for (int i = 0; i < K; i++) ctr[i] *= (1. / K);

		// reflection
		for (int i = 0; i < K; i++)
			refl[i] = ctr[i] + refl_c * (ctr[i] - P[high_id][i]);
		double refl_val = fun(refl);
		if (val[low_id] <= refl_val && refl_val < val[high2_id]) {
			for (int i = 0; i < K; i++) P[high_id][i] = refl[i];
			val[high_id] = refl_val;
			continue;
		}

		// expansion
		if (refl_val < val[low_id]) {
			for (int i = 0; i < K; i++)
				expd[i] = ctr[i] + expd_c * (ctr[i] - P[high_id][i]);
			double expd_val = fun(expd);
			double *new_vec = expd_val < refl_val ? expd : refl;
			double new_val = min(expd_val, refl_val);
			for (int i = 0; i < K; i++) P[high_id][i] = new_vec[i];
			val[high_id] = new_val;
			continue;
		}

		// contraction
		for (int i = 0; i < K; i++)
			ctrct[i] = ctr[i] - ctrct_c * (ctr[i] - P[high_id][i]);
		double ctrct_val = fun(ctrct);
		if (ctrct_val < val[high_id]) {
			for (int i = 0; i < K; i++) P[high_id][i] = ctrct[i];
			val[high_id] = ctrct_val;
			continue;
		}

		// compression/shrinking
		for (int i = 0; i <= K; i++) if (i != low_id) {
			for (int _ = 0; _ < K; _++)
				P[i][_] = P[low_id][_] + shrnk_c * (P[i][_] - P[low_id][_]);
			val[i] = fun(P[i]);
		}

	}

}


// setup initial simplex for Nelder-Mead
// @K: dimension
// @P0[K]: initial guess, it is ok that S[0]==P0
// @S[K+1][K]: initial simplex
// @r: radius of expansion
void setupInitialSimplex_axesAligned(int K, const double P0[], double* S[], double r) {
	for (int i = 0; i < K; i++) {
		for (int j = 0; j <= K; j++) S[j][i] = P0[i];
	}
	for (int j = 1; j <= K; j++) S[j][j - 1] += r;
}
void setupInitialSimplex_regular(int K, const double P0[], double* S[], double r) {
	// https://en.wikipedia.org/wiki/Simplex#Cartesian_coordinates_for_a_regular_n-dimensional_simplex_in_Rn
	double k1 = -(sqrt(K + 1) + 1) / pow(K, 1.5);
	double k0 = k1 + sqrt(1 + 1. / K);
	double k2 = 1. / sqrt(K);
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < K; j++)
			S[i + 1][j] = (i == j ? k1 : k0) * r + P0[j];
	}
	for (int i = 0; i < K; i++)  // just make sure it works when S[0]==P0
		S[0][i] = k2 * r + P0[i];
}






/* Powell Conjugate Direction */

template<typename Fun> vec2 PowellConjugateDirection_2d(Fun F, vec2 p0, double epsilon = 1e-6, double *val = nullptr, vec2 e0 = vec2(NAN), vec2 e1 = vec2(NAN)) {
	if (isnan(e0.sqr()) || isnan(e1.sqr())) {
		e0 = vec2(1, 0), e1 = vec2(0, 1);
		vec2 eps = pMax(vec2(epsilon), 1e-4*abs(p0));
		double fp = F(p0), fp10 = F(p0 + vec2(eps.x, 0)), fp01 = F(p0 + vec2(0, eps.y));
		vec2 grad = vec2(fp10 - fp, fp01 - fp) / eps;
		e0 = grad * (-fp / grad.sqr()), e1 = e0.rot();
	}
	auto line_min = [&](vec2 p, vec2 d) {
		double t0 = 0., t1 = length(d), tc;
		d = normalize(d);
		auto fun = [&](double t) {
#ifdef _DEBUG_OPTIMIZATION
			drawDot(p + d * t, 2, COLOR{ 200,80,0 });
#endif
			return F(p + d * t);
		};
		bracketMinimum_golden(fun, t0, t1, &tc);
		//t0 = -10, t1 = 10, tc = 0.;
		double mv, mt = Brent_minimize_1d(fun, t0, tc, t1, epsilon, &mv);
		return vec3(p + d * mt, mv);
	};
	for (int i = 0; i < 100; i++) {
		vec3 s1 = line_min(p0, e0); vec2 p1 = s1.xy();
		vec3 s2 = line_min(p1, e1);	vec2 p2 = s2.xy();
		vec3 s3 = line_min(p2, e0);	vec2 p3 = s3.xy();
#ifdef _DEBUG_OPTIMIZATION
		drawLine(p0, p1, COLOR{ 200,80,0 });
		drawLine(p1, p2, COLOR{ 200,80,0 });
		drawLine(p2, p3, COLOR{ 200,80,0 });
		drawDot(p0, 5, COLOR({ 200,80,0 }));
		printf("%lf\n", length(p3 - p0));  // quadratic convergence but each step is too expensive compare to MNS
#endif
		if (length(p3 - p0) < epsilon) {
			if (val) *val = s3.z;
			return p3;
		}
		if (p3 != p1) e1 = e0, e0 = p3 - p1, e1 = i == 0 ? e0.rot() : p2 - p0;
		else { vec2 e = e0; e0 = e1, e1 = e; }
		p0 = p3;
	}
	return p0;
}





#endif // __INC_OPTIMIZATION_H

