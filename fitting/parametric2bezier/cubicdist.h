// Serves for parametric2bezier.cpp
// Finding the distance from a point to a cubic parametric cuve


// Quintic equation solvers
#include "numerical/rootfinding.h"  // solveQuintic_bisect
int rootFinder_Bezier(double C[], double R[]);


// To-read: https://hal.inria.fr/file/index/docid/518379/filename/Xiao-DiaoChen2007c.pdf




// calculate the square of distance to a cubic parametric curve
// by finding the roots of a quintic polynomial
double CubicCurveDistance2(vec2 C[4], vec2 P) {
	vec2 c0 = C[0] - P, c1 = C[1], c2 = C[2], c3 = C[3];
	vec2 p0 = c0;
	vec2 p1 = c0 + c1 + c2 + c3;
	double md = min(p0.sqr(), p1.sqr());
	double k[6];
	k[5] = 3.*c3.sqr();
	k[4] = 5.*dot(c2, c3);
	k[3] = 4.*dot(c1, c3) + 2.*c2.sqr();
	k[2] = 3.*(dot(c0, c3) + dot(c1, c2));
	k[1] = 2.*dot(c0, c2) + c1.sqr();
	k[0] = dot(c0, c1);
	double R[5];
	int NR = solveQuintic_bisect(k, R, 0., 1., 1e-6, true);
	//int NR = rootFinder_Bezier(k, R);
	for (int i = 0; i < NR; i++) {
		double t = R[i];
		vec2 b = c0 + t * (c1 + t * (c2 + t * c3));
		md = min(md, b.sqr());
	}
	return md;
}








// Method described in Graphics Gems (seems to be slower)

#define N 5

// find all real roots in [0,1]
int rootFinder_Bezier_recurse(double[], double*, double, double, int);
int rootFinder_Bezier(double C[N + 1], double R[N]) {
	// convert power basis to Bernstein-Bezier form
	for (int j = 1; j <= N; j++) {
		double c = 1. / (N + 1 - j);
		double d = 1., e = c;
		for (int i = N; i >= j; i--) {
			C[i] = d * C[i] + e * C[i - 1];
			d -= c, e += c;
		}
	}
	// call recursive function
	return rootFinder_Bezier_recurse(C, R, 0., 1., 60);
}
// recursive part of the solver
int rootFinder_Bezier_recurse(double K[], double *R, double t0, double t1, int remain) {
	// count the positive coefficients
	int pc = 0;
	for (int i = 0; i <= N; i++) pc += K[i] > 0;
	// does not cross t-axis
	if (pc == 0 || pc == N + 1) return 0;
	// termination / cross exactly once
	if (remain == 0 || (pc == 1 || pc == N)) {
		if (K[N] < K[0]) return 0;
		const double eps = 1e-4;
		// linear interpolation - y=mt+b
		double m = (K[N] - K[0]) / (t1 - t0), b = K[0] - m * t0; m = 1. / m;
		double intpt = (K[0] * t1 - K[N] * t0) / (K[0] - K[N]);
		// recursion limit exceeded or too small
		if (remain == 0 || t1 - t0 < eps) {
			R[0] = intpt; return 1;
		}
		// calculate the maximum possible error - might have a bug
		double maxerr = 0.;
		for (int i = 1; i < N; i++) {
			double t = ((N - i)*t0 + i * t1) / N;
			double t_ = m * (K[i] - b);
			maxerr = std::max(maxerr, abs(t_ - t));
			if (!(2.*maxerr < eps)) break;
		}
		if (2.*maxerr < eps || abs(t1 - t0) < eps) {
			R[0] = intpt; return 1;
		}
	}
	// recursively split
	{
		// de Casteljau's algorithm splitting curve at the center
		// _K becomes the left part and K becomes the right part
		double _K[N + 1];
		for (int j = N; j >= 0; j--) {
			_K[N - j] = K[0];
			for (int i = 0; i < j; i++) {
				K[i] = 0.5*(K[i] + K[i + 1]);
			}
		}
		// recursively find root
		int N1 = rootFinder_Bezier_recurse(_K, R, t0, 0.5*(t0 + t1), remain - 1);
		int N2 = rootFinder_Bezier_recurse(K, &R[N1], 0.5*(t0 + t1), t1, remain - 1);
		return N1 + N2;
	}
}

#undef N




