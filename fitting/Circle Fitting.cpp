// Fitting a circle to point set experiment

// To-do:
// try the parametric equation of circle C+r(cos(t),sin(t))


#include "numerical/optimization.h"

#define _RANDOM_H_BETTER_QUALITY
#include "numerical/random.h"

#include <stdio.h>




// ============================================================== Fitting ==============================================================


// Fit points to a circle with center C and radius r

// circle equations:
// (x-cx)²+(y-cy)²=r²
// x²+y²+ax+by+c=0
// |p-c|=r
// C+r(cos(t),sin(t))



// Minimize Σ[(C-P[i])²-R²]²
// It can be seem that R² = 1/N Σ(C-P[i])²
void fitCircle(const vec2* P, int N, vec2 &C, double &R) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0, Sx3 = 0, Sy3 = 0, Sx2y = 0, Sxy2 = 0;
	for (int i = 0; i < N; i++) {
		double x = P[i].x, y = P[i].y, x2 = x * x, y2 = y * y;
		Sx += x, Sy += y, Sx2 += x2, Sy2 += y2, Sxy += x * y;
		Sx3 += x2 * x, Sy3 += y2 * y, Sx2y += x2 * y, Sxy2 += x * y2;
	}
	// After a heavy simplification, this problem turns into solving a linear system
	double A[2][2] = {
		2 * N*Sx2 - 2 * Sx*Sx, 2 * N*Sxy - 2 * Sx*Sy,
		2 * N*Sxy - 2 * Sx*Sy, 2 * N*Sy2 - 2 * Sy*Sy
	};
	double b[2] = {
		N*Sx3 + N * Sxy2 - Sx * Sx2 - Sx * Sy2,
		N*Sy3 + N * Sx2y - Sy * Sx2 - Sy * Sy2
	};
	double inv_det = 1.0 / (A[0][0] * A[1][1] - A[0][1] * A[1][0]);
	C.x = inv_det * (b[0] * A[1][1] - b[1] * A[0][1]);
	C.y = inv_det * (b[1] * A[0][0] - b[0] * A[1][0]);
	R = sqrt((1. / N)* (N * C.x*C.x - 2 * Sx*C.x + Sx2 + N * C.y*C.y - 2 * Sy*C.y + Sy2));
}

// Minimize the same equation numerically
// Test the numerical optimizer
void fitCircle_Numerical(const vec2* P, int N, vec2 &C, double &R) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0, Sx3 = 0, Sy3 = 0, Sx2y = 0, Sxy2 = 0, Sx4 = 0, Sx2y2 = 0, Sy4 = 0;
	for (int i = 0; i < N; i++) {
		double x = P[i].x, y = P[i].y, x2 = x * x, y2 = y * y;
		Sx += x, Sy += y, Sx2 += x2, Sy2 += y2, Sxy += x * y,
			Sx3 += x2 * x, Sy3 += y2 * y, Sx2y += x2 * y, Sxy2 += x * y2,
			Sx4 += x2 * x2, Sx2y2 += x2 * y2, Sy4 += y2 * y2;
	}
	// The function to minimize
	int Calls = 0; // count call times
	auto E = [&](vec2 p)->double {
		Calls++;
		double a = p.x, b = p.y, a2 = a * a, a3 = a2 * a, a4 = a2 * a2, b2 = b * b, b3 = b2 * b, b4 = b2 * b2;
		double r2 = ((a2*N - 2 * a*Sx + Sx2) + (b2*N - 2 * b*Sy + Sy2)) / N;
		return (a4 + 2 * a2*b2 + b4 - 2 * a2*r2 - 2 * b2*r2 + r2 * r2)*N \
			+ (-4 * a3 - 4 * a*b2 + 4 * a*r2)*Sx + (6 * a2 + 2 * b2 - 2 * r2)*Sx2 - 4 * a*Sx3 + Sx4 \
			+ (-4 * a2*b - 4 * b3 + 4 * b*r2)*Sy + (2 * a2 + 6 * b2 - 2 * r2)*Sy2 - 4 * b*Sy3 + Sy4 \
			+ 8 * a*b*Sxy - 4 * b*Sx2y - 4 * a*Sxy2 + 2 * Sx2y2;
		// I believe this will become a quadratic form after simplification
	};
	C = Newton_Gradient_2d(E, vec2(Sx, Sy) / N);
	R = sqrt(((C.x*C.x*N - 2 * C.x*Sx + Sx2) + (C.y*C.y*N - 2 * C.y*Sy + Sy2)) / N);
	//printf("%d\n", Calls);  // this varies from 20+ to 150+; Newton_Iteration_2d only 9x3=27 (theorically 9 or 18)
}


// Minimize Σ[x²+y²+ax+by+c]² and determine C and R from a,b,c
// Result is exactly the same as the previous functions
void fitCircle_E(const vec2* P, int N, vec2 &C, double &R) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0, Sx3 = 0, Sy3 = 0, Sx2y = 0, Sxy2 = 0;
	for (int i = 0; i < N; i++) {
		double x = P[i].x, y = P[i].y, x2 = x * x, y2 = y * y;
		Sx += x, Sy += y, Sx2 += x2, Sy2 += y2, Sxy += x * y;
		Sx3 += x2 * x, Sy3 += y2 * y, Sx2y += x2 * y, Sxy2 += x * y2;
	}
	double A[2][2] = {
		N*Sx2 - Sx * Sx, N*Sxy - Sx * Sy,
		N*Sxy - Sx * Sy, N*Sy2 - Sy * Sy
	};
	double Y[2] = {
		Sx*(Sx2 + Sy2) - N * (Sx3 + Sxy2),
		Sy*(Sx2 + Sy2) - N * (Sx2y + Sy3)
	};
	double inv_det = 1.0 / (A[0][0] * A[1][1] - A[0][1] * A[1][0]);
	double a = inv_det * (Y[0] * A[1][1] - Y[1] * A[0][1]);
	double b = inv_det * (Y[1] * A[0][0] - Y[0] * A[1][0]);
	double c = (-1. / N) * (Sx2 + Sy2 + a * Sx + b * Sy);
	C.x = -0.5*a, C.y = -0.5*b;
	R = 0.5*sqrt(a*a + b * b - 4.*c);
}


// Minimize Σ[|C-P[i]|-R]² (numerically)
// Visually, this works better than the above methods (but much slower)
void fitCircle_O(const vec2* P, int N, vec2 &C, double &R) {
	double *d = new double[N];
	int Calls = 0;
	// Warn that this function sometimes does not contain a local minima
	auto E = [&](vec2 p) {
		Calls++;
		R = 0;
		for (int i = 0; i < N; i++) {
			R += d[i] = length(p - P[i]);  // R² = 1/N Σ|C-P[i]|
		}
		R /= N;
		double Err = 0;
		for (int i = 0; i < N; i++) {
			double e = d[i] - R;
			Err += e * e;
		}
		return Err;
	};
	// start from the result of the previous function
	// this significantly reduces the chance of failture
	fitCircle(P, N, C, R);
	//C = Newton_Gradient_2d(E, C);  // slower
	C = Newton_Iteration_2d_(E, C);  // faster but less stable
	// do not have to calculate R because it is already calculated
	delete d;
	// debug output
	printf("%d\t%.3lf %.3lf\t%.3lf\n", Calls, C.x, C.y, R);
	if (0.0*R != 0.0) fprintf(stderr, "Error! %d\n", __LINE__);
}

// Same as previous except it uses analytical differentiation, more than 3 times faster
void fitCircle_O_ad(const vec2* P, int N, vec2 &C, double &R) {
	int Calls = 0;
	// calculating the analytical gradient of this function is truely a nightmare......
	auto E = [&](vec2 p, vec2 *grad, vec2 *grad2, double *dxy) ->double {
		Calls++;
		vec2 Su(0.), Sd(0.);
		R = 0;
		double Sd2 = 0., Sdi = 0.;
		vec2 Si3(0); double Sx3 = 0;
		for (int i = 0; i < N; i++) {
			vec2 d = p - P[i]; Sd += d;
			double d2 = dot(d, d); Sd2 += d2;
			double dl = sqrt(d2); R += dl;
			double di = 1. / dl; Sdi += di;
			Su += d * di;
			double di3 = di * di * di;
			Si3 += (d*d)*di3, Sx3 += d.x*d.y*di3;
		}
		R /= N;
		if (grad) *grad = 2.*(Sd - R * Su);
		if (grad2) *grad2 = 2. * (vec2(N) - (Su*Su / N + R * (vec2(Sdi) - Si3)));
		if (dxy) *dxy = (-2. / N) * (Su.x*Su.y - N * R*Sx3);
		return Sd2 - N * R*R;
	};
	vec2 C0; double R0; fitCircle(P, N, C0, R0);
	C = Newton_Iteration_2d_ad(E, C0);
	// calculate R
	R = 0.0;
	for (int i = 0; i < N; i++) R += length(C - P[i]);
	R /= N;
	// debug output
	printf("%d\t%.3lf %.3lf\t%.3lf\n", Calls, C.x, C.y, R);
	if (0.0*R != 0.0) fprintf(stderr, "Error! %d\n", __LINE__);
	// In the test, this function fails 6 cases in 10000 cases.
	// All failtures are caused by the gradient information leading it to infinity.
	// Try to detect failture and restart iteration from a new startpoint when fails.
}





// ============================================================== Visualizing ==============================================================


// visualizing fitting results
// contains graphing classes/variables/functions

#pragma region Visualization

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <libraries/stb_image_write.h>

// color structs
typedef unsigned char byte;
typedef struct { byte r, g, b; } COLOR;
COLOR mix(COLOR a, COLOR b, double d) {
	auto f = [&](byte a, byte b) { return (byte)((1 - d)*a + d * b); };
	return COLOR{ f(a.r,b.r), f(a.g,b.g), f(a.b,b.b) };
}

// image variables
#define W 600
#define H 400
COLOR canvas[W*H];
#define Scale 30
#define fromCoord(x,y) (vec2(x,-y)*Scale+vec2(.5*W,.5*H-.5))
#define fromScreen(x,y) ((vec2(x,-y)+vec2(-.5*W,.5*H-.5))*(1./Scale))

// graphing functions - code for simplicity not performance
void drawAxis(double width, COLOR col) {
	width *= 0.5;
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = fromScreen(i, j);
		p = vec2(abs(p.x), abs(p.y));
		double d = min(p.x, p.y) * Scale - width;
		if (d < 0) canvas[j*W + i] = col;
		else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
	}
}
void drawDot(vec2 c, double r, COLOR col) {
	vec2 C = fromCoord(c.x, c.y);
	r -= 0.5;
	int i0 = max(0, (int)floor(C.x - r - 1)), i1 = min(W - 1, (int)ceil(C.x + r + 1));
	int j0 = max(0, (int)floor(C.y - r - 1)), j1 = min(H - 1, (int)ceil(C.y + r + 1));
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		vec2 p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale);
		double d = length(p - c) * Scale - r;
		canvas[j*W + i] = mix(canvas[j*W + i], col, 0.75 * clamp(1. - d, 0., 1.));
	}
}
void drawCircle(vec2 c, double r, double width, COLOR col) {
	width *= 0.5;
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = fromScreen(i, j);
		double d = abs(length(p - c) - r) * Scale - width;
		if (d < 0) canvas[j*W + i] = col;
		else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
	}
}

// initialize and save image
void init() { for (int i = 0, l = W * H; i < l; i++) canvas[i] = COLOR{ 255,255,255 }; }
bool save(const char* path) { return stbi_write_png(path, W, H, 3, canvas, 3 * W); }

#pragma endregion





// ============================================================== Testing ==============================================================


#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;


// generate point data in the pattern of a circle
void randomPointData(vec2 *P, int N) {
	// parameters of the line
	vec2 c = randv_n(2.0);
	double r = randf(1.0, 4.0);
	// parameters of random number generator
	double u = randf(0, 2.*PI), v = randf(0.5, 2.0);
	double f = randf(0.05, 0.4);
	// generating random points
	for (int i = 0; i < N; i++) {
		double t = u + randf_n(v);
		vec2 p = c + r * vec2(cos(t), sin(t));
		P[i] = p + randv_n(f);
	}
}

// write lots of pictures to see fitting results
void randomTest_image() {
	for (int i = 0; i < 1000; i++)
	{
		// generate point data
		_SRAND(i);
		int N = (int)(randf_n(80) + 200); N = max(N, 3);
		vec2 *P = new vec2[N]; randomPointData(P, N);
		// fitting and visualization
		init();
		vec2 c; double r;
		fitCircle_Numerical(P, N, c, r); drawCircle(c, r, 8, COLOR{ 192,192,192 });
		fitCircle(P, N, c, r); drawCircle(c, r, 5, COLOR{ 192,255,128 });
		fitCircle_E(P, N, c, r); drawCircle(c, r, 3, COLOR{ 160,232,255 });
		//fitCircle_O(P, N, c, r); drawCircle(c, r, 5, COLOR{ 255,232,160 });
		fitCircle_O_ad(P, N, c, r); drawCircle(c, r, 5, COLOR{ 255,232,160 });
		// visualization
		drawAxis(3, COLOR{ 0,0,255 });
		for (int i = 0; i < N; i++) drawDot(P[i], 3.5, COLOR{ 255,0,0 });
		// save image
		char s[] = "tests\\test0000.png";
		for (int d = 13, j = i; d >= 10; d--) s[d] = j % 10 + '0', j /= 10;
		save(s);
		delete P;
	}
}

// test the performance and stability of numerical methods
void randomTest_numerical() {
	freopen("tests\\test.txt", "w", stdout);  // write to file
	freopen("tests\\error.txt", "w", stderr);  // on my system this directly writes to console if this line is commented
	auto t0 = NTime::now();
	for (int i = 0; i < 10000; i++)
	{
		// the same generator as randomTest_image
		_SRAND(i);
		int N = (int)(randf_n(80) + 200); N = max(N, 3);
		vec2 *P = new vec2[N]; randomPointData(P, N);
		printf("%d\t%d\t", i, N);
		// fitting
		vec2 c; double r;
		//fitCircle_O(P, N, c, r);
		fitCircle_O_ad(P, N, c, r);
		if (!(r < 10) && N > 3) {  // (possible) failture
			fprintf(stderr, "%d\t%d\t%lf\t%lf\t%lf\t\n", i, N, c.x, c.y, r);
		}
		delete P;
	}
	fprintf(stderr, "%lfsecs elapsed\n", fsec(NTime::now() - t0).count());
}



int main() {
	//randomTest_image(); exit(0);
	randomTest_numerical(); exit(0);
	return 0;
}

