// Fitting ellipse to planar point set experiment

// WHY IS FITTING SUCH SIMPLE SHAPE SO **** ????

/* To-do:
	 - handle the case when the matrix is not invertible
	 - try to render dashed/dotted lines because the colors are really messy
	 - think this problem: do fitted shapes change after translating the points?
*/

// Ironically, in this file, testing code is longer than fitting code.



// some modules are organized into the "numerical" folder
#include "numerical/eigensystem.h"
#include "numerical/geometry.h"
#include "numerical/random.h"
#include "numerical/rootfinding.h"
#include "numerical/optimization.h"


#include <stdio.h>
#include <chrono>
#include <algorithm>






// Ellipse fitting - v contains the coefficients of { x², xy, y², x, y, 1 }, {a,b,c,d,e,f}

// generate least square fitting matrix, should be positive (semi)definite
void generateMatrix(double M[6][6], const vec2 *P, int N) {
	for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) M[i][j] = 0;
	for (int n = 0; n < N; n++) {
		vec2 v = P[n];
		double k[6] = { v.x*v.x, v.x*v.y, v.y*v.y, v.x,v.y, 1. };
		for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) {
			M[i][j] += k[i] * k[j];
		}
	}
}



// Ellipse fitting with quadratic constraints, solve for eigenvalues

// Minimize vᵀMv, subject to vᵀv=1
void fitEllipse0(const vec2 *P, int N, double v[6]) {
	double M[6][6]; generateMatrix(M, P, N);
	double lambda; EigenPair_invIter(6, &M[0][0], &lambda, v);
	// if not ellipse, check other eigenvalues
	if (4.*v[0] * v[2] <= v[1] * v[1]) {
		double eigv[6], eigvec[6][6];
		EigenPairs_expand(6, &M[0][0], eigv, &eigvec[0][0]);
		for (int i = 1; i < 6; i++) {
			auto w = eigvec[i];
			if (4.*w[0] * w[2] > w[1] * w[1]) {
				for (int j = 0; j < 6; j++) v[j] = w[j];
				return;
			}
		}
	}
}

// Minimize vᵀMv, subject to vᵀCv=1
// To get expected result, C should be non-negative definite
// I'm sure there is a bug because it sometimes gets hyperbolas
void fitEllipse_ConstraintMatrix(const vec2 *P, int N, double v[6], const double C[6][6], bool checkEllipse = true) {
	double M[6][6]; generateMatrix(M, P, N);
	double I[6][6]; matinv(6, &M[0][0], &I[0][0]);
	double B[6][6]; matmul(6, &I[0][0], &C[0][0], &B[0][0]);  // B is not symmetric
	double u; EigenPair_powIter(6, &B[0][0], &u, v);
	if (checkEllipse) {
		// make sure it is an ellipse
		if (4.*v[0] * v[2] <= v[1] * v[1]) {
			double eigv[6], eigvec[6][6];
			EigenPairs_expand(6, &B[0][0], eigv, &eigvec[0][0]);
			for (int i = 1; i < 6; i++) {
				auto w = eigvec[i];
				if (4.*w[0] * w[2] > w[1] * w[1]) {
					for (int j = 0; j < 6; j++) v[j] = w[j];
					return;
				}
			}
		}
	}
}

// Minimize vᵀMv, subject to 4ac-b²=1
void fitEllipse1(const vec2 *P, int N, double v[6]) {
	const double C[6][6] = { {0,0,2,0,0,0},{0,-1,0,0,0,0},{2,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0} };
	fitEllipse_ConstraintMatrix(P, N, v, C);
}
// Minimize vᵀMv, subject to a²+c²=1
void fitEllipse2(const vec2 *P, int N, double v[6]) {
	const double C[6][6] = { {1,0,0,0,0,0},{0,0,0,0,0,0},{0,0,1,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0} };
	fitEllipse_ConstraintMatrix(P, N, v, C);
}
// Minimize vᵀMv, subject to a²+1/2b²+c²=1
void fitEllipse3(const vec2 *P, int N, double v[6]) {
	const double C[6][6] = { {1,0,0,0,0,0},{0,.5,0,0,0,0},{0,0,1,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0} };
	fitEllipse_ConstraintMatrix(P, N, v, C);
}

// Minimize vᵀMv/vᵀCv, where C is the sum of magnitude of gradients
// doesn't "shrink", visually good but checking hyperbolas makes it horrible
void fitEllipse_grad(const vec2 *P, int N, double v[6]) {
	double C[6][6] = { {0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0} };
	for (int i = 0; i < N; i++) {
		double x = P[i].x, y = P[i].y, x2 = x * x, y2 = y * y, xy = x * y;
		C[0][0] += 4 * x2, C[1][1] += x2 + y2, C[2][2] += 4 * y2, C[3][3] += 1, C[4][4] += 1;
		C[0][1] += 2 * xy, C[1][0] += 2 * xy; C[1][2] += 2 * xy, C[2][1] += 2 * xy;
		C[0][3] += 2 * x, C[3][0] += 2 * x; C[2][4] += 2 * y, C[4][2] += 2 * y; C[1][3] += y, C[3][1] += y; C[1][4] += x, C[4][1] += x;
	}
	fitEllipse_ConstraintMatrix(P, N, v, C, false);
}



// Ellipse fitting with linear constraints, solve a linear system

// Minimize vᵀMv, subject to a+c=1
void fitEllipse_ac1(const vec2 *P, int N, double v[6]) {
#if 0
	double M[6][6]; generateMatrix(M, P, N);
	double c[6] = { 1,0,1,0,0,0 };
	for (int i = 0; i < 6; i++) v[i] = c[i];
	solveLinear(6, &M[0][0], v);
#else
	double Sx4 = 0, Sx3y = 0, Sx2y2 = 0, Sxy3 = 0, Sy4 = 0;
	double Sx3 = 0, Sx2y = 0, Sxy2 = 0, Sy3 = 0;
	double Sx2 = 0, Sxy = 0, Sy2 = 0, Sx = 0, Sy = 0;
	for (int i = 0; i < N; i++) {
		double x = P[i].x, y = P[i].y, x2 = x * x, y2 = y * y, xy = x * y;
		Sx += x, Sy += y, Sx2 += x2, Sxy += xy, Sy2 += y2;
		Sx3 += x2 * x, Sx2y += x2 * y, Sxy2 += x * y2, Sy3 += y2 * y;
		Sx4 += x2 * x2, Sx3y += x2 * xy, Sx2y2 += xy * xy, Sxy3 += xy * y2, Sy4 += y2 * y2;
	}
	double M[5][5] = {
		Sx4 - 2 * Sx2y2 + Sy4, Sx3y - Sxy3, Sx3 - Sxy2, Sx2y - Sy3, Sx2 - Sy2,
		Sx3y - Sxy3, Sx2y2, Sx2y, Sxy2, Sxy,
		Sx3 - Sxy2, Sx2y, Sx2, Sxy, Sx,
		Sx2y - Sy3, Sxy2, Sxy, Sy2, Sy,
		Sx2 - Sy2, Sxy, Sx, Sy, double(N)
	};
	double B[5] = { Sx2y2 - Sy4, Sxy3, Sxy2, Sy3, Sy2 };
	solveLinear(5, &M[0][0], &B[0]);
	v[0] = -B[0], v[1] = -B[1], v[2] = 1. + B[0], v[3] = -B[2], v[4] = -B[3], v[5] = -B[4];
#endif
}

// Minimize vᵀMv, subject to f=1
void fitEllipse_f1(const vec2 *P, int N, double v[6]) {
#if 0
	double M[6][6]; generateMatrix(M, P, N);
	double c[6] = { 0,0,0,0,0,1 };
	for (int i = 0; i < 6; i++) v[i] = c[i];
	solveLinear(6, &M[0][0], v);
#else
	double Sx4 = 0, Sx3y = 0, Sx2y2 = 0, Sxy3 = 0, Sy4 = 0;
	double Sx3 = 0, Sx2y = 0, Sxy2 = 0, Sy3 = 0;
	double Sx2 = 0, Sxy = 0, Sy2 = 0, Sx = 0, Sy = 0;
	for (int i = 0; i < N; i++) {
		double x = P[i].x, y = P[i].y, x2 = x * x, y2 = y * y, xy = x * y;
		Sx += x, Sy += y, Sx2 += x2, Sxy += xy, Sy2 += y2;
		Sx3 += x2 * x, Sx2y += x2 * y, Sxy2 += x * y2, Sy3 += y2 * y;
		Sx4 += x2 * x2, Sx3y += x2 * xy, Sx2y2 += xy * xy, Sxy3 += xy * y2, Sy4 += y2 * y2;
	}
	double M[5][5] = {
		Sx4, Sx3y, Sx2y2, Sx3, Sx2y,
		Sx3y, Sx2y2, Sxy3, Sx2y, Sxy2,
		Sx2y2, Sxy3, Sy4, Sxy2, Sy3,
		Sx3, Sx2y, Sxy2, Sx2, Sxy,
		Sx2y, Sxy2, Sy3, Sxy, Sy2
	};
	double B[5] = { Sx2, Sxy, Sy2, Sx, Sy };
	solveLinear(5, &M[0][0], &B[0]);
	for (int i = 0; i < 5; i++) v[i] = B[i];
	v[5] = -1;
#endif
}



// Fitting quadratic curves by minimizing the sum of the square of the exact Euclidean distance
// As a reference, doesn't seem to be practical

// calculate the exact distance to a quadratic curve (slow)
double distanceToQuadratic(const double v[6], vec2 p) {
	double a = v[0], b = v[1], c = v[2], d = v[3], e = v[4], f = v[5];
	double x0 = p.x, y0 = p.y;
	// temp variables
	double u2 = b * e - 2 * c*d, u1 = 4 * c*x0 - 2 * b*y0 - 2 * d, u0 = 4 * x0;
	double v2 = b * d - 2 * a*e, v1 = 4 * a*y0 - 2 * b*x0 - 2 * e, v0 = 4 * y0;
	double w2 = 4 * a*c - b * b, w1 = 4 * a + 4 * c, w0 = 4;
	// coefficients of the quartic equation
	double c4 = a * u2*u2 + b * u2*v2 + c * v2*v2 + d * u2*w2 + e * v2*w2 + f * w2*w2;
	double c3 = a * (2 * u1*u2) + b * (u2*v1 + u1 * v2) + c * (2 * v1*v2) + d * (u2*w1 + u1 * w2) + e * (v2*w1 + v1 * w2) + f * (2 * w1*w2);
	double c2 = a * (2 * u0*u2 + u1 * u1) + b * (u2*v0 + u0 * v2 + u1 * v1) + c * (2 * v0*v2 + v1 * v1) + d * (u2*w0 + u0 * w2 + u1 * w1) + e * (v2*w0 + v0 * w2 + v1 * w1) + f * (2 * w0*w2 + w1 * w1);
	double c1 = a * (2 * u0*u1) + b * (u1*v0 + u0 * v1) + c * (2 * v0*v1) + d * (u0*w1 + u1 * w0) + e * (v0*w1 + v1 * w0) + f * (2 * w1*w0);
	double c0 = a * u0*u0 + b * u0*v0 + c * v0*v0 + d * u0*w0 + e * v0*w0 + f * w0*w0;
	// solve for the Lagrangian
	double r[4];
	int N = solveQuartic(c4, c3, c2, c1, c0, r);
	if (N == 0) {  // curve not exist
		// not sure if this can work
		// continuous but not differentiable
		double invdet = 1. / (4 * a*c - b * b);
		double x = invdet * (-2 * c*d + b * e);
		double y = invdet * (-2 * a*e + b * d);
		double val = a * x*x + b * x*y + c * y*y + d * x + e * y + f;
		return length(p - vec2(x, y)) + val;
	}
	// find the nearest root
	double md = INFINITY, ml;
	for (int i = 0; i < N; i++) {
		double l = r[i];
		double invdet = 1.0 / (w0 + l * (w1 + l * w2));
		double x = invdet * (u0 + l * (u1 + l * u2));
		double y = invdet * (v0 + l * (v1 + l * v2));
		double d = (x - x0)*(x - x0) + (y - y0)*(y - y0);
		if (d < md) md = d, ml = l;
	}
	// "refine" the root
	ml = refineRoot_quartic(c4, c3, c2, c1, c0, ml);
	return length(vec2(u0 + ml * (u1 + ml * u2), v0 + ml * (v1 + ml * v2)) / (w0 + ml * (w1 + ml * w2)) - p);
}

// Ellipse fitting using the Euclidean distance
void fitEllipse_Euclidian(const vec2 *P, int N, double v[6]) {
	// hope this curve doesn't go through the origin...
	auto E = [&](double c[5]) {
		double k[6] = { c[0], c[1], c[2], c[3], c[4], 1 };
		double s = 0;
		for (int i = 0; i < N; i++) {
			double d = distanceToQuadratic(k, P[i]);
			s += d * d;
		}
		return s;
	};
	double c[6];
	fitEllipse_grad(P, N, c);
	for (int i = 0; i < 5; i++) c[i] /= c[5];
	Newton_Iteration_Minimize(5, E, c, c, true, 128);
	double eg = E(c);
	for (int i = 0; i < 5; i++) v[i] = c[i]; v[5] = 1;
	fitEllipse0(P, N, c);
	for (int i = 0; i < 5; i++) c[i] /= c[5];
	Newton_Iteration_Minimize(5, E, c, c, true, 128);
	if (E(c) < eg) for (int i = 0; i < 5; i++) v[i] = c[i];
}







// ================================================================ Testing Code ================================================================



// visualizing ellipse fitting results
// contains graphing classes/variables/functions

#pragma region Visualization

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <libraries/stb_image_write.h>

// color structs
typedef unsigned char byte;
typedef struct { byte r, g, b; } COLOR;
COLOR mix(COLOR a, COLOR b, double d) {
	//auto f = [&](byte a, byte b) { return (byte)((1 - d)*a + d * b); };
	int k = int(d * 256); auto f = [&](byte a, byte b) { return (byte)(((256 - k)*a + k * b) >> 8); };
	return COLOR{ f(a.r,b.r), f(a.g,b.g), f(a.b,b.b) };
}

// image variables
#define W 600
#define H 400
COLOR canvas[W*H];
double buffer[W*H];
#define Scale 30
#define Center vec2(0, 0)
const vec2 fromCoord(0.5*W - Scale * Center.x, Scale*Center.y - (0.5 - 0.5*H));
const vec2 fromScreen(-0.5*W / Scale + Center.x, (0.5*H - 0.5) / Scale + Center.y);

// painting functions
void drawAxis(double width, COLOR col, bool grid = true) {
	width *= 0.5;
	// go through all pixels (may be slow)
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
			p = vec2(abs(p.x), abs(p.y));
			double d = min(p.x, p.y) * Scale - width;
			if (grid) d = min(d, Scale * (.5 - max(abs(fmod(p.x, 1.) - .5), abs(fmod(p.y, 1.) - .5))));
			if (d < 0) canvas[j*W + i] = col;
			else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
		}
	}
}
void drawDot(vec2 c, double r, COLOR col, bool hollow = true) {
	vec2 C = vec2(c.x, -c.y) * Scale + fromCoord;
	r -= 0.5;
	int i0 = max(0, (int)floor(C.x - r - 1)), i1 = min(W - 1, (int)ceil(C.x + r + 1));
	int j0 = max(0, (int)floor(C.y - r - 1)), j1 = min(H - 1, (int)ceil(C.y + r + 1));
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		vec2 p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale) + Center;
		double d = length(p - c) * Scale - r;
		if (hollow) d = abs(d);
		canvas[j*W + i] = mix(canvas[j*W + i], col, 0.75 * clamp(1. - d, 0., 1.));
	}
}
void drawQuadraticCurve(const double v[6], double width, COLOR col, bool hollow = false) {
	double r = 0.5*width;
	// initialize a value buffer
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
		double x = p.x, y = p.y;
		buffer[j*W + i] = v[0] * x*x + v[1] * x*y + v[2] * y*y + v[3] * x + v[4] * y + v[5];
	}
	// map the value of the buffer to the image
	for (int j = 1; j < H - 1; j++) for (int i = 1; i < W - 1; i++) {
		// calculate numerical gradient from neighbourhood values
		double dx = buffer[j*W + i + 1] - buffer[j*W + i - 1];
		double dy = buffer[j*W + i + W] - buffer[j*W + i - W];
		double m = .5*sqrt(dx * dx + dy * dy);  // magnitude of gradient
		double d = abs(buffer[j*W + i] / m) - r;  // divide by gradient to estimate distance
		if (hollow) d = abs(d);
		if ((d = 1. - d) > 0.) canvas[j*W + i] = mix(canvas[j*W + i], col, 0.8 * clamp(d, 0., 1.));
	}
}

// visualizing distance function
void visualizeDistance(const double c[6]) {
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
		double d0 = distanceToQuadratic(c, p);
		double d = Scale * abs(d0);
		double s = 128 * (0.2 * cos(0.5*d) + 1) * (1.0 - 0.7*exp(-.03*d));
		canvas[j*W + i] = mix(COLOR{ byte(1.6*s), byte(1.4*s), byte(1.2*s) }, COLOR{ 255,255,255 }, clamp(.4*(4. - d), 0., 1.));
		if (d0 < 0.) std::swap(canvas[j*W + i].r, canvas[j*W + i].b);
	}
}

// initialize and save image
void init() {
	for (int i = 0, l = W * H; i < l; i++) canvas[i] = COLOR{ 255,255,255 };
}
bool save(const char* path) {
	return stbi_write_png(path, W, H, 3, canvas, 3 * W);
}

#pragma endregion






// Test fitting methods with random data

// test eigenvalue calculation
bool checkEigenpair(const double M[6][6], double lambda, const double v[6]) {
	bool ok = true;
	double Ax[6]; matvecmul(6, &M[0][0], v, Ax);
	double m = 0; for (int i = 0; i < 6; i++) m += Ax[i] * Ax[i]; m = 1. / sqrt(m);
	double e;
	if ((e = abs(m * lambda - 1)) > 1e-6) {
		printf("#%d %lg\n", __LINE__, e);
		ok = false;
	}
	for (int i = 0; i < 6; i++) Ax[i] *= m;
	m = 0; for (int i = 0; i < 6; i++) m += Ax[i] * v[i];
	if ((e = abs(m - 1)) > 1e-6) {
		printf("#%d %lg\n", __LINE__, e);
		ok = false;
	}
	return ok;
}
void randomTest_eigen() {
	auto t0 = std::chrono::high_resolution_clock::now();

	double M[6][6];
	for (int i = 0; i < 1000; i++)
	{
		_IDUM = i;
		vec2 P[10];
		for (int i = 0; i < 10; i++) P[i] = rand2(1.0);
		generateMatrix(M, P, 10);

#if 0
		double lambda, lambda0, eigv[6];
		EigenPair_invIter(6, &M[0][0], &lambda, eigv);
		checkEigenpair(M, lambda, eigv);
#else
		double eigv[6], eigvec[6][6];
		EigenPairs_expand(6, &M[0][0], eigv, &eigvec[0][0]);
		for (int t = 0; t < 6; t++) {
			if (!checkEigenpair(M, eigv[t], eigvec[t])) {
				printf("%d_%d\n", i, t);
			}
		}
#endif
	}

	printf("%lfs elapsed\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
}

// generate point data in the pattern of an ellipse
void randomPointData(vec2 *P, int N) {
	// parameters of the ellipse
	double rx = abs(randf_n(1.) + 3.) + 1.;
	double ry = abs(randf_n(1.) + 3.) + 1.;
	double a = randf(0, 2.*PI);
	double x = randf(-4.0, 4.0);
	double y = randf(-3.5, 3.5);
	double ca = cos(a), sa = sin(a), rx2 = rx * rx, ry2 = ry * ry;
	double c[6];
	c[0] = ca * ca / rx2 + sa * sa / ry2, c[2] = sa * sa / rx2 + ca * ca / ry2;
	c[1] = 2. * sa * ca * (1. / ry2 - 1. / rx2);
	c[3] = c[4] = 0, c[5] = -1;
	// parameters of random number generator
	vec2 g = rand2_n(5.0);
	double v = randf_n(2.0); v = v * v + 2.0;
	double f = randf_n(0.4); f = f * f + 0.1;
	// generating random points
	for (int i = 0; i < N; i++) {
		vec2 p = g + rand2_n(v);
		// using iteration to make points close to the ellipse
		for (int t = 0; t < 6; t++) {
			double x = p.x, y = p.y;
			double z = c[0] * x*x + c[1] * x*y + c[2] * y*y + c[3] * x + c[4] * y + c[5];
			vec2 dz = vec2(2.*c[0] * x + c[1] * y + c[3], c[1] * x + 2.*c[2] * y + c[4]);
			p = p - dz * (z / dot(dz, dz));
		}
		// add noise
		P[i] = p + rand2_n(f);
	}
}

// use random data to test ellipse fitting methods
void randomTest_image() {
	freopen("tests\\test.txt", "w", stdout);
	for (int i = 0; i < 100; i++)
	{
		// generate point data
		_IDUM = i;
		const int N = 200;
		vec2 *P = new vec2[N]; randomPointData(P, N);
		// fitting
		double c0[6]; fitEllipse0(P, N, c0);
		double c1[6]; fitEllipse1(P, N, c1);
		double c2[6]; fitEllipse2(P, N, c2);
		double c3[6]; fitEllipse3(P, N, c3);
		double cg[6]; fitEllipse_grad(P, N, cg);
		double d0[6]; fitEllipse_ac1(P, N, d0);
		double d1[6]; fitEllipse_f1(P, N, d1);
		double ec[6]; fitEllipse_Euclidian(P, N, ec);  // EXTREMELY SLOW!!!!!!
		// write result to stdout
		auto printc = [](double *c) {
			printf("%c %lfx^2%+lfxy%+lfy^2%+lfx%+lfy%+lf=0\n", 4.*c[0] * c[2] > c[1] * c[1] ? 'E' : 'H',
				c[0], c[1], c[2], c[3], c[4], c[5]);
		};
		printf("Test %d\n", i);
		printc(c0); printc(c1); printc(c2); printc(c3); printc(cg);
		printc(d0); printc(d1); printc(ec);
		printf("\n");
#if 1
		// visualization
#if 1
		// visualize fitting result
		init();
		drawAxis(2, COLOR{ 232,232,232 });
		drawQuadraticCurve(ec, 8, COLOR{ 232,232,232 });  // reference; one should be able to identify failures visually (#97 is pretty obvious)
		drawQuadraticCurve(c0, 4, COLOR{ 192,255,128 });  // v² : light green
		drawQuadraticCurve(c1, 4, COLOR{ 232,232,128 });  // 4ac-b² : khaki color
		drawQuadraticCurve(c2, 4, COLOR{ 255,192, 64 });  // a²+c² :  orange
		drawQuadraticCurve(c3, 4, COLOR{ 255,168,128 });  // a²+b²/2+c² : pink
		drawQuadraticCurve(cg, 4, COLOR{ 192, 64,192 });  // v²/grad² : purple
		drawQuadraticCurve(d0, 3, COLOR{ 128,128,255 }, true);  // a+c : hollow blue
		drawQuadraticCurve(d1, 3, COLOR{ 255,128,128 }, true);  // f : hollow pink
		for (int i = 0; i < N; i++) {
			drawDot(P[i], 3, COLOR{ 0,0,0 });
		}
#else
		// debug distance - WHY IS MY ART SO BAD???
		visualizeDistance(cg);
		drawAxis(2, COLOR{ 232,232,232 }, false);
		for (int i = 0; i < N; i++) {
			drawDot(P[i], 3, COLOR{ 255,0,0 }, false);
		}
#endif
		// save image
		char s[] = "tests\\test00.png";
		s[10] = i / 10 + '0', s[11] = i % 10 + '0';
		save(s);
#endif
		delete P;
	}
}

// test the performance and stability of ellipse fitting methods
#define TEST_N 10000
void randomTest_numerical() {
	freopen("tests\\test.txt", "w", stdout);
	// generate test data
	int PN[TEST_N]; vec2 *P[TEST_N];
	for (int i = 0; i < TEST_N; i++) {
		_SRAND(i);
		PN[i] = (int)(randf_n(50) + 200); PN[i] = max(PN[i], 5);
		P[i] = new vec2[PN[i]]; randomPointData(P[i], PN[i]);
	}
	// fitting
	double v[TEST_N][6];
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < TEST_N; i++) {
		// place an ellipse fitting function there
		fitEllipse0(P[i], PN[i], v[i]);
	}
	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
	// check result
	double Err[TEST_N], sumErr = 0; int totS = 0;
	int HyperbolaCount = 0;
	for (int T = 0; T < TEST_N; T++) {
		double err = 0;
		for (int i = 0; i < PN[T]; i++) {
			double e = distanceToQuadratic(v[T], P[T][i]);
			err += e * e;
		}
		Err[T] = err / PN[T];
		if (0.0*Err[T] == 0.0) sumErr += Err[T], totS++;
		if (!(abs(v[T][1]) < 1e6 && abs(v[T][5]) < 1e6)) {
			printf("#%d %lfx^2%+lfxy%+lfy^2%+lfx%+lfy%+lf=0\n", T, v[T][0], v[T][1], v[T][2], v[T][3], v[T][4], v[T][5]);
		}
		HyperbolaCount += v[T][1] * v[T][1] > 4. * v[T][0] * v[T][2];
	}
	printf("%.3lfsecs, %d hyperbolas;\n", time_elapsed, HyperbolaCount);
	std::sort(Err, Err + TEST_N);
	printf("Average loss %.3lf; Minimum/Median/Maximum losses %.3lf, %.3lf, %.3lf, %.3lf, %.3lf.\n",
		sumErr / totS, Err[0], Err[TEST_N / 4], Err[TEST_N / 2], Err[3 * TEST_N / 4], Err[TEST_N - 1]);
}

// test if shape change after linear transforms
void randomTest_transform(void fitEllipse(const vec2*, int, double[])) {
	double sse = 0, ste = 0;
	auto normalize = [](double v[6]) {
		double m = 0; for (int i = 0; i < 6; i++) m += v[i] * v[i];
		m = 1.0 / sqrt(m); for (int i = 0; i < 6; i++) v[i] *= m;
	};
	auto normalizef = [](double v[6]) {
		double m = 0; for (int i = 0; i < 3; i++) m += v[i] * v[i];
		m = 1.0 / sqrt(m); for (int i = 0; i < 6; i++) v[i] *= m;
	};
	for (int i = 0; i < TEST_N; i++) {
		_IDUM = i;
		vec2 P[100]; randomPointData(P, 100);
		double v0[6]; fitEllipse(P, 100, v0); normalize(v0);
		double sc = randf(0.01, 100);
		vec2 PS[100]; for (int i = 0; i < 100; i++) PS[i] = P[i] * sc;
		double vs[6]; fitEllipse(PS, 100, vs);
		vs[0] *= sc * sc, vs[1] *= sc * sc, vs[2] *= sc * sc, vs[3] *= sc, vs[4] *= sc; normalize(vs);
		double e = 0; for (int i = 0; i < 6; i++) e += v0[i] * vs[i]; sse += (e - 1)*(e - 1);
		vec2 tr = rand2_n(20);
		vec2 PT[100]; for (int i = 0; i < 100; i++) PT[i] = P[i] + tr;
		double vt[6]; fitEllipse(PT, 100, vt);
		normalizef(v0); normalizef(vt);
		if (vt[0] * v0[0] < 0) for (int i = 0; i < 6; i++) vt[i] *= -1;
		vt[5] += vt[0] * tr.x*tr.x + vt[1] * tr.x*tr.y + vt[2] * tr.y*tr.y + vt[3] * tr.x + vt[4] * tr.y;
		vt[3] += 2 * vt[0] * tr.x + vt[1] * tr.y, vt[4] += 2 * vt[2] * tr.y + vt[1] * tr.x;
		normalize(v0); normalize(vt);
		e = 0; for (int i = 0; i < 6; i++) e += v0[i] * vt[i]; ste += (e - 1)*(e - 1);
	}
	printf("%lf %lf\n", sse / TEST_N, ste / TEST_N);
}





int main() {
	randomTest_image();
	return 0;
}



/*

Experiment result: (TEST_N=10000)

fitEllipse0 - Minimize vᵀMv, subject to vᵀv=1
0.213secs, 0 hyperbolas;
Average loss 0.313; Minimum/Median/Maximum losses 0.000, 0.015, 0.034, 0.115, 55.721.

fitEllipse1 - Minimize vᵀMv, subject to 4ac-b²=1
0.225secs, 1351 hyperbolas;
Average loss 0.132; Minimum/Median/Maximum losses 0.000, 0.016, 0.040, 0.131, 4.739.

fitEllipse2 - Minimize vᵀMv, subject to a²+c²=1
0.210secs, 1483 hyperbolas;
Average loss 0.115; Minimum/Median/Maximum losses 0.000, 0.015, 0.035, 0.103, 3.605.

fitEllipse3 - Minimize vᵀMv, subject to a²+1/2b²+c²=1
0.209secs, 693 hyperbolas;
Average loss 0.132; Minimum/Median/Maximum losses 0.000, 0.015, 0.037, 0.118, 16.775.

fitEllipse_grad - Minimize vᵀMv/vᵀCv, C is the sum of magnitude of gradients
0.238secs, 1075 hyperbolas;
Average loss 0.105; Minimum/Median/Maximum losses 0.000, 0.013, 0.030, 0.091, 3.507.

fitEllipse_ac1 - Minimize vᵀMv, subject to a+c=1
0.044secs, 1198 hyperbolas;
Average loss 0.114; Minimum/Median/Maximum losses 0.000, 0.015, 0.034, 0.099, 3.963.

fitEllipse_f1 - Minimize vᵀMv, subject to f=1
0.054secs, 2461 hyperbolas;
Average loss 0.147; Minimum/Median/Maximum losses 0.000, 0.016, 0.039, 0.127, 16.398.

*/


/*

Linear transformation test:

TEST_N = 10:
fitEllipse0: 0.091399 0.371707
fitEllipse1: 0.674457 0.000000
fitEllipse2: 0.000000 0.000000
fitEllipse3: 0.005112 0.000000
fitEllipse_grad: 0.000000 0.000000
fitEllipse_ac1: 0.000000 0.000000
fitEllipse_f1: 0.000000 0.001966

TEST_N = 100:
fitEllipse0: 0.103037 0.332775
fitEllipse1: 0.320260 0.159825
fitEllipse2: 0.016617 0.033467
fitEllipse3: 0.008571 0.076296
fitEllipse_grad: 0.000000 0.000000
fitEllipse_ac1: 0.000000 0.000000
fitEllipse_f1: 0.000000 0.198647

TEST_N = 10000:
fitEllipse0: 0.114073 0.274444
fitEllipse1: 0.350403 0.079191
fitEllipse2: 0.032357 0.025077
fitEllipse3: 0.023227 0.049633
fitEllipse_grad: 0.010400 0.000000
fitEllipse_ac1: 0.000000 0.000000
fitEllipse_f1: 0.000000 0.224596

Note that fitEllipse 1-3 checks hyperbolas; fitEllipse_grad uses numerical optimization that sometimes fails.

*/


// All methods except fitEllipse0 can get hyperbolas.
// Among these methods, fitEllipse_grad works visually best.


