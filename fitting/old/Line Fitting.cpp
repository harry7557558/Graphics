// Fitting a straight to point set experiment

// To-do: 


#define _RANDOM_H_BETTER_QUALITY
#include "numerical/random.h"

#include <stdio.h>




// ============================================================== Fitting ==============================================================


// Fit points to a straight line ax+by+c=0


// standard linear regression
void linear_regression(const vec2* p, int N, double &a, double &b, double &c) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0;
	for (int i = 0; i < N; i++) {
		double x = p[i].x, y = p[i].y;
		Sx += x, Sy += y, Sx2 += x * x, Sy2 += y * y, Sxy += x * y;
	}
	a = Sxy * N - Sx * Sy, b = Sx * Sx - Sx2 * N;
	c = Sx2 * Sy - Sx * Sxy;
}


// orthogonal distance least square fitting
// results of the two methods should represent the same line

// derived using equation of straight line cos(t)x+sin(t)y-d=0
void fitLine_T(const vec2* p, int N, double &a, double &b, double &c) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0;
	for (int i = 0; i < N; i++) {
		double x = p[i].x, y = p[i].y;
		Sx += x, Sy += y, Sx2 += x * x, Sy2 += y * y, Sxy += x * y;
	}
	a = Sx * Sx - Sy * Sy + (Sy2 - Sx2) * N, b = 2.0 * (Sxy * N - Sx * Sy);
	double t = -atan(b / a);
	if (a*cos(t) < b*sin(t)) t += PI;
	t *= 0.5;
	a = cos(t), b = sin(t);
	c = (Sx * a + Sy * b) / (-N);
}

// derived using Lagrange multiplier
void fitLine_L(const vec2 *p, int N, double &a, double &b, double &c) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0;
	for (int i = 0; i < N; i++) {
		double x = p[i].x, y = p[i].y;
		Sx += x, Sy += y, Sx2 += x * x, Sy2 += y * y, Sxy += x * y;
	}
	double k = (N * (Sx2 - Sy2) - (Sx * Sx - Sy * Sy)) / (N * Sxy - Sx * Sy);  // k=a/b-b/a
	a = (N*Sxy > Sx*Sy ? -1 : 1)*sqrt(k*k + 4) + k;
	b = 2.;
	c = (Sx * a + Sy * b) / (-N);
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
#define W 900
#define H 600
COLOR canvas[W*H];
#define Scale 50
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
		canvas[j*W + i] = mix(canvas[j*W + i], col, 0.8 * clamp(1. - d, 0., 1.));
	}
}
void drawStraightLine(double a, double b, double c, double width, COLOR col) {
	double m = 1. / sqrt(a * a + b * b); a *= m, b *= m, c *= m;
	double r = 0.5*width;
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = fromScreen(i, j);
		double d = abs(a*p.x + b * p.y + c) * Scale - width;
		if ((d = 1. - d) > 0.) canvas[j*W + i] = mix(canvas[j*W + i], col, 1.0*clamp(d, 0, 1));
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





// ============================================================== Testing ==============================================================


// generate point data in the pattern of a straight line
void randomPointData(vec2 *P, int N) {
	// parameters of the line
	double t = randf(0, 2 * PI);
	vec2 d = vec2(cos(t), sin(t));
	vec2 p0 = rand2_n(2.0);
	// parameters of random number generator
	double v = randf(1.0, 3.0);
	double f = randf(0.0, 0.8); f = f * f + 0.2;
	// generating random points
	for (int i = 0; i < N; i++) {
		double t = randf_n(v);
		vec2 p = p0 + d * t;
		P[i] = p + rand2_n(f);
	}
}

// write lots of pictures to see fitting results
void randomTest_image() {
	for (int i = 0; i < 100; i++) {
		// generate point data
		_SRAND(i);
		const int N = 200;
		vec2 *P = new vec2[N]; randomPointData(P, N);
		// fitting
		double a0, b0, d0; linear_regression(P, N, a0, b0, d0);
		double at, bt, dt; fitLine_T(P, N, at, bt, dt);
		double al, bl, dl; fitLine_L(P, N, al, bl, dl);
		// visualization
		init();
		drawStraightLine(a0, b0, d0, 5, COLOR{ 232,232,232 });
		drawStraightLine(at, bt, dt, 5, COLOR{ 192,255,128 });
		drawStraightLine(al, bl, dl, 5, COLOR{ 160,232,255 });
		drawAxis(5, COLOR{ 0,0,255 });
		for (int i = 0; i < N; i++) drawDot(P[i], 5, COLOR{ 255,0,0 });
		// save image
		char s[] = "tests\\test00.png";
		s[10] = i / 10 + '0', s[11] = i % 10 + '0';
		save(s);
	}
}


int main() {
	randomTest_image();
	return 0;
}

