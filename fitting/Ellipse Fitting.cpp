// Fitting ellipse to point set experiment (incomplete)

/* To-do:
	 - implement orthogonal least square fitting
	 - try other eigenvalue algorithms (Jacobi, QR)
	 - handle the case when the matrix is not invertible
*/


// some modules are organized into the "numerical" folder
#include "numerical/eigensystem.h"
#include "numerical/geometry.h"
#include "numerical/random.h"

#include <stdio.h>
#include <chrono>


// generate matrix for ellipse fitting, should be positive (semi)definite
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

// debug functions
void printMatrix(const double M[6][6], const char end[] = "\\\\\n") {  // latex output
	printf("\\begin{bmatrix}");
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			if (abs(M[i][j]) < 1e-6) printf("0");
			else printf("%.6g", M[i][j]);
			if (j < 5) putchar('&');
		}
		printf("\\\\");
	}
	printf("\\end{bmatrix}%s", end);
}
void printVector(const double v[6], const char end[] = "\n") {
	printf("(");
	for (int i = 0; i < 6; i++) printf("%lf%c", v[i], i == 5 ? ')' : ',');
	printf("%s", end);
}
void printPolynomial(const double C[], int N, const char end[] = "\n") {
	printf("%lgx^{%d}", C[0], N);
	for (int i = 1; i < N; i++) {
		printf("%+.16lf*x^{%d}", C[i], N - i);
	}
	printf("%+.16lf%s", C[N], end);
}




// ellipse fitting - v contains the coefficients of { x², xy, y², x, y, 1 }
void fitEllipse(const vec2 *P, int N, double v[6]) {
	// Minimize vᵀMv, subject to vᵀv=1
	double M[6][6]; generateMatrix(M, P, N);
	double lambda; EigenPair_invIter(M, lambda, v);
	// if not ellipse, check other eivenvalues
	if (4.*v[0] * v[2] <= v[1] * v[1]) {
		double eigv[6], eigvec[6][6];
		EigenPairs_expand(M, eigv, eigvec);
		for (int i = 1; i < 6; i++) {
			auto w = eigvec[i];
			if (4.*w[0] * w[2] > w[1] * w[1]) {
				for (int j = 0; j < 6; j++) v[j] = w[j];
				return;
			}
		}
	}
}
void fitEllipse1(const vec2 *P, int N, double v[6]) {
	// Minimize vᵀMv, subject to 4ac-b²=1
	// I'm sure there is a bug because it sometimes get hyperbolas
	double M[6][6]; generateMatrix(M, P, N);
	const double F[6][6] = { {0,0,2,0,0,0},{0,-1,0,0,0,0},{2,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0} };
	double I[6][6]; matinv(M, I);
	double B[6][6]; matmul(I, F, B);
	double u; EigenPair_powIter(B, u, v);
	if (4.*v[0] * v[2] <= v[1] * v[1]) {
		double eigv[6], eigvec[6][6];
		EigenPairs_expand(B, eigv, eigvec);
		for (int i = 1; i < 6; i++) {
			auto w = eigvec[i];
			if (4.*w[0] * w[2] > w[1] * w[1]) {
				for (int j = 0; j < 6; j++) v[j] = w[j];
				return;
			}
		}
	}
}





// visualizing ellipse fitting results
// contains graphing classes/variables/functions

#pragma region Visualization

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <libraries/stb_image_write.h>

// color structs
typedef unsigned char byte;
typedef struct { byte r, g, b; } COLOR;
COLOR mix(COLOR a, COLOR b, double d) {
	auto f = [&](byte a, byte b) { return (byte)((1 - d)*a + d * b); };
	//int k = d * 256; auto f = [&](byte a, byte b) { return (byte)(((256 - k)*a + k * b) >> 8); };
	return COLOR{ f(a.r,b.r), f(a.g,b.g), f(a.b,b.b) };
}

// image variables
#define W 900
#define H 600
COLOR canvas[W*H];
double buffer[W*H];
#define Scale 50
#define Center vec2(0, 0)
const vec2 fromCoord(0.5*W - Scale * Center.x, Scale*Center.y - (0.5 - 0.5*H));
const vec2 fromScreen(-0.5*W / Scale + Center.x, (0.5*H - 0.5) / Scale + Center.y);

// painting functions
void drawAxis(double width, COLOR col) {
	width *= 0.5;
	// go through all pixels (may be slow)
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
			p = vec2(abs(p.x), abs(p.y));
			double d = min(p.x, p.y) * Scale - width;
			//d = min(d, Scale * (.5 - max(abs(fmod(p.x, 1.) - .5), abs(fmod(p.y, 1.) - .5))));  // grid
			if (d < 0) canvas[j*W + i] = col;
			else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
		}
	}
}
void drawDot(vec2 c, double r, COLOR col) {
	vec2 C = vec2(c.x, -c.y) * Scale + fromCoord;
	r -= 0.5;
	int i0 = max(0, (int)floor(C.x - r - 1)), i1 = min(W - 1, (int)ceil(C.x + r + 1));
	int j0 = max(0, (int)floor(C.y - r - 1)), j1 = min(H - 1, (int)ceil(C.y + r + 1));
	for (int j = j0; j <= j1; j++) {
		for (int i = i0; i <= i1; i++) {
			vec2 p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale) + Center;
			double d = length(p - c) * Scale - r;
			canvas[j*W + i] = mix(canvas[j*W + i], col, 0.75 * clamp(1. - d, 0., 1.));  // shade by distance to anti-aliase
		}
	}
}
void drawQuadraticCurve(const double v[6], double width, COLOR col) {
	double r = 0.5*width;
	// initialize a value buffer
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
			double x = p.x, y = p.y;
			buffer[j*W + i] = v[0] * x*x + v[1] * x*y + v[2] * y*y + v[3] * x + v[4] * y + v[5];
		}
	}
	// map the value of the buffer to the image
	for (int j = 1; j < H - 1; j++) {
		for (int i = 1; i < W - 1; i++) {
			// calculate numerical gradient from neighbourhood values
			double dx = buffer[j*W + i + 1] - buffer[j*W + i - 1];
			double dy = buffer[j*W + i + W] - buffer[j*W + i - W];
			double m = .5*sqrt(dx * dx + dy * dy);  // magnitude of gradient
			double d = abs(buffer[j*W + i] / m) - r;  // divide by gradient to estimate distance
			if ((d = 1. - d) > 0.) canvas[j*W + i] = mix(canvas[j*W + i], col, 0.8 * clamp(d, 0., 1.));
		}
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






// random tests for debug

// test eigenvalue calculation
bool checkEigenpair(const double M[6][6], double lambda, const double v[6]) {
	bool ok = true;
	double Ax[6]; matmul(M, v, Ax);
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
		for (int i = 0; i < 10; i++) P[i] = randv(1.0);
		generateMatrix(M, P, 10);
		//printMatrix(M);

#if 0
		double lambda, lambda0, eigv[6];
		EigenPair_invIter(M, lambda, eigv);
		checkEigenpair(M, lambda, eigv);
#else
		double eigv[6], eigvec[6][6];
		EigenPairs_expand(M, eigv, eigvec);
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
	vec2 g = randv_n(5.0);
	double v = randf_n(2.0); v = v * v + 2.0;
	double f = randf_n(0.4); f = f * f + 0.1;
	// generating random points
	for (int i = 0; i < N; i++) {
		vec2 p = g + randv_n(v);
		// using iteration to make points close to the ellipse
		for (int t = 0; t < 6; t++) {
			double x = p.x, y = p.y;
			double z = c[0] * x*x + c[1] * x*y + c[2] * y*y + c[3] * x + c[4] * y + c[5];
			vec2 dz = vec2(2.*c[0] * x + c[1] * y + c[3], c[1] * x + 2.*c[2] * y + c[4]);
			p = p - dz * (z / dot(dz, dz));
		}
		// add noise
		P[i] = p + randv_n(f);
	}
}

// write lots of pictures to see fitting results
void randomTest_image() {
	for (int i = 0; i < 100; i++) {
		// generate point data
		_IDUM = i;
		const int N = 200;
		vec2 *P = new vec2[N]; randomPointData(P, N);
		// fitting
		double c0[6]; fitEllipse(P, N, c0);
		double c1[6]; fitEllipse1(P, N, c1);
		// visualization
		init();
		drawQuadraticCurve(c0, 8, COLOR{ 192,255,128 });
		drawQuadraticCurve(c1, 8, COLOR{ 255,192,128 });
		drawAxis(5, COLOR{ 255,0,0 });
		for (int i = 0; i < N; i++) {
			drawDot(P[i], 5, COLOR{ 0,0,255 });
		}
		// save image
		char s[] = "tests\\test00.png";
		s[10] = i / 10 + '0', s[11] = i % 10 + '0';
		save(s);
	}
}





int main() {
	randomTest_image(); exit(0);
	return 0;
}

