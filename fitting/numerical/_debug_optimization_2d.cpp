/* INCOMPLETE */
// A 2d function minimization experiment
// Comparing different optimization methods

// vec2
#include "geometry.h"



// CG stuff - the rendering looks nice ;)

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
double buffer0[W*H], buffer[W*H], bufferv[W*H], buffervx[W*H], buffervy[W*H];
#define Scale 50
#define Center vec2(0, 0)
const vec2 fromCoord(0.5*W - Scale * Center.x, Scale*Center.y - (0.5 - 0.5*H));
const vec2 fromScreen(-0.5*W / Scale + Center.x, (0.5*H - 0.5) / Scale + Center.y);

// painting functions
void drawAxis(double width, COLOR col, bool axis = true, bool grid = false) {
	width *= 0.5;
	// go through all pixels (may be slow)
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
		p = vec2(abs(p.x), abs(p.y));
		double d = 1e12;
		if (axis) {
			d = min(p.x, p.y) * Scale - width;
		}
		if (grid) {
			p.x = abs(p.x - floor(p.x) - .5), p.y = abs(p.y - floor(p.y) - .5);
			double g = Scale * (.5 - max(p.x, p.y));
			d = min(d, 1.0 - 0.5*clamp(1.0 - g, 0., 1.));
		}
		if (d < 0) canvas[j*W + i] = col;
		else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
	}
}
void drawDot(vec2 c, double r, COLOR col) {
	vec2 C = vec2(c.x, -c.y) * Scale + fromCoord;
	r -= 0.5;
	int i0 = max(0, (int)floor(C.x - r - 1)), i1 = min(W - 1, (int)ceil(C.x + r + 1));
	int j0 = max(0, (int)floor(C.y - r - 1)), j1 = min(H - 1, (int)ceil(C.y + r + 1));
	for (int j = j0; j <= j1; j++) for (int i = i0; i <= i1; i++) {
		vec2 p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale) + Center;
		double d = length(p - c) * Scale - r;
		canvas[j*W + i] = mix(canvas[j*W + i], col, clamp(1. - d, 0., 1.));  // shade by distance to anti-aliase
	}
}
void drawLine(vec2 p, vec2 q, COLOR col) {  // naive DDA, no anti-aliasing
	p = vec2(p.x, -p.y) * Scale + fromCoord;
	q = vec2(q.x, -q.y) * Scale + fromCoord;
	vec2 d = q - p;
	double slope = d.y / d.x;
	if (abs(slope) <= 1.0) {
		if (p.x > q.x) { vec2 t = p; p = q; q = t; }
		int x0 = max(0, int(p.x)), x1 = min(W - 1, int(q.x)), y;
		double yf = slope * x0 + (p.y - slope * p.x);
		for (int x = x0; x <= x1; x++) {
			y = (int)yf;
			if (y >= 0 && y < H) canvas[y*W + x] = col;
			yf += slope;
		}
	}
	else {
		slope = d.x / d.y;
		if (p.y > q.y) { vec2 t = p; p = q; q = t; }
		int y0 = max(0, int(p.y)), y1 = min(H - 1, int(q.y)), x;
		double xf = slope * y0 + (p.x - slope * p.y);
		for (int y = y0; y <= y1; y++) {
			x = (int)xf;
			if (x >= 0 && x < W) canvas[y*W + x] = col;
			xf += slope;
		}
	}
};
void drawContour(double F(vec2), double width, double contour = 0, bool log_scale = false) {
	double r = 0.5*width;
	// initialize a value buffer
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		vec2 p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
		double val = F(p);
		bufferv[j*W + i] = val;
		if (log_scale) val = (val > 0. ? .5 : -.5)*log10(val*val + 1.);
		buffer0[j*W + i] = val;
		if (contour != 0) val = sin((1. / contour)*PI*val);
		buffer[j*W + i] = val;
	}
	// calculate numerical gradient from neighbourhood values
	for (int j = 1; j < H - 1; j++) for (int i = 1; i < W - 1; i++) {
		buffervx[j*W + i] = bufferv[j*W + i + 1] - bufferv[j*W + i - 1];
		buffervy[j*W + i] = bufferv[j*W + i + W] - bufferv[j*W + i - W];
	}
	// graph the zero-isoline of gradient
	for (int j = 2; j < H - 2; j++) for (int i = 2; i < W - 2; i++) {
		// dFdx = 0
		double dx = buffervx[j*W + i + 1] - buffervx[j*W + i - 1];
		double dy = buffervx[j*W + i + W] - buffervx[j*W + i - W];
		double d = abs(buffervx[j*W + i] / (.5*sqrt(dx * dx + dy * dy))) - 2.*r;
		if ((d = 1. - d) > 0.) canvas[j*W + i] = mix(canvas[j*W + i], COLOR{ 192,192,192 }, 0.4 * clamp(d, 0., 1.));
		// dFdy = 0
		dx = buffervy[j*W + i + 1] - buffervy[j*W + i - 1];
		dy = buffervy[j*W + i + W] - buffervy[j*W + i - W];
		d = abs(buffervy[j*W + i] / (.5*sqrt(dx * dx + dy * dy))) - 2.*r;
		if ((d = 1. - d) > 0.) canvas[j*W + i] = mix(canvas[j*W + i], COLOR{ 192,192,192 }, 0.4 * clamp(d, 0., 1.));
	}
	// graph the contour
	for (int j = 1; j < H - 1; j++) for (int i = 1; i < W - 1; i++) {
		double dx = buffer[j*W + i + 1] - buffer[j*W + i - 1];
		double dy = buffer[j*W + i + W] - buffer[j*W + i - W];
		double m = .5*sqrt(dx * dx + dy * dy);  // magnitude of gradient
		double d = abs(buffer[j*W + i] / m) - r;  // divide by gradient to estimate distance
		if ((d = 1. - d) > 0.) {
			COLOR heatmap[5] = { COLOR{0,0,255},COLOR{75,215,180},COLOR{75,215,0},COLOR{255,215,0},COLOR{255,0,0} };  // colors
			double z = 10. / (exp(-.5*buffer0[j*W + i]) + 1.) - 5.; z = clamp(z, 0, 3.9999);  // interval mapping
			COLOR col = mix(heatmap[int(z)], heatmap[int(z) + 1], z - int(z));  // linear interpolation
			canvas[j*W + i] = mix(canvas[j*W + i], col, 0.5 * clamp(d, 0., 1.));
		}
	}
	// distance estimation has some artifacts at the points with high curvature
	// discontinuities may be stroked
	// zero-isoline stroke will be double-width when using logarithmic scale
}
#include "_debug_font_rendering.h"
void drawCharacter(char c, vec2 p, double sz, COLOR col) {
	int x0 = max((int)p.x, 0), y0 = max((int)p.y, 0);
	int x1 = min((int)(p.x + sz), W - 1), y1 = min((int)(p.y + sz), H - 1);
	for (int y = y0; y <= y1; y++) {
		for (int x = x0; x <= x1; x++) {
			double d = sdFont(c, (x - x0) / sz, (y - y0) / sz);
			d *= sz;
			if (d < 0.) canvas[y*W + x] = col;
			else if (d < 1.) canvas[y*W + x] = mix(col, canvas[y*W + x], d);
		}
	}
}
void drawString(const char s[], vec2 p, double h, COLOR col) {
	while (*s) {
		drawCharacter(*s, p, h, col);
		p.x += .5*h, s++;
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




// test functions for optimization
// all functions contain at least one global minimas

int Fi = 0;
int F_count = 0;
double F(vec2 p) {
	F_count++;
	double x = p.x, y = p.y;
#define sq(x) ((x)*(x))

	// original cases
	if (Fi < 10) {
		if (Fi == 0) return x * x + y * y;
		if (Fi == 1) return x * x + x * y + y * y - x - y;
		if (Fi == 2) return x * x*x*x - 4 * x*x*y + y * y*y*y;
		if (Fi == 3) return x * x - 1.99*x*y + y * y - 0.01*(x + y);
		if (Fi == 4) return x * x - abs(x)*y + y * y - 1.0;
		if (Fi == 5) return 100 * sq(sqrt(x*x + y * y) - 2.) - x;
		if (Fi == 6) return 100 * sq(sqrt(x*x + y * y) - 2.) - (x - y);
		if (Fi == 7) return sq(length(p - vec2(1, 1)) + length(p + vec2(1, 1)));
		if (Fi == 8) return sq(length(p - vec2(1, 1)) + length(p + vec2(1, 1)) + length(p + vec2(-1, 1)));
		if (Fi == 9) return sq(max(abs(x), abs(y)) - 1.) + x + y;
	}

	// https://en.wikipedia.org/wiki/Test_functions_for_optimization
	if (Fi < 20) {
		if (Fi == 10) return 100 * sq(y - x * x) + sq(1 - x);  // Rosenbrock function
		if (Fi == 11) return sq(x*x + y - 11) + sq(y*y + x - 7);  // Himmelblau's function
		if (Fi == 12) return sin(x + y) + sq(x - y) - 1.5*x + 2.5*y + 1 + 0.1*x*x;  // McCormick function (modified)
		if (Fi == 13) return sq(1.5 - x + x * y) + sq(2.25 - x + x * y*y) + sq(2.625 - x + x * y*y*y);  // Beale function
		if (Fi == 14) return (1 + sq(x + y + 1)*(19 - 14 * (x + y) + 3 * sq(x + y)))*(30 + sq(2 * x - 3 * y)*(18 - 32 * x + 48 * y + 3 * sq(2 * x - 3 * y)));  // Goldstein-Price function
		if (Fi == 15) return x * x * (2. + x * x* (-1.05 + 0.15*x * x)) + x * y + y * y;  // Three-hump camel function
		if (Fi == 16) return (x*x*x*x - 16 * x*x + 5 * x) + (y*y*y*y - 16 * y*y + 5 * y);  // Styblinski-Tang function
		if (Fi == 17) return sin(y)*exp(sq(1 - cos(x))) + cos(x)*exp(sq(1 - sin(y))) + sq(x - y) + sq(x + y);  // Mishra's Bird function (modified unconstrained)
		if (Fi == 18) return sq(sqrt(abs(y - 0.01*x*x)) + 0.1*abs(x + 10));  // modified
		if (Fi == 19) return sq(x + 2 * y - 7) + sq(2 * x + y - 5);  // Booth function
	}

	// https://www.sfu.ca/~ssurjano/optimization.html
	if (Fi < 30) {
		if (Fi == 20) return 2 * (sq(x - 1) + sq(x*x - 1)) + 3 * (sq(y - .25) + sq(y*y - .25));
		if (Fi == 21) return x * x + abs(y * y*y) + x + y;
		if (Fi == 22) return abs(x*x*x) + y * y*y*y;
		if (Fi == 23) return 0.26*(x*x + y * y) - 0.48*x*y;
		if (Fi == 24) return sq(x + y - 8) + sq(x*x + y * y - 18);
		if (Fi == 25) return sq(x + y - 8) + sq(x*x + y * y - 18) + sq(x*x*x + y * y*y - 44);
		if (Fi == 26) return sq(x + y - 8) + sq(x*x + y * y - 18) + sq(x*x*x + y * y*y - 44) + sq(x*x*x*x + y * y*y*y - 114);
		if (Fi == 27) return x * x + y * y + sq(0.5*x + y) + sq(sq(0.5*x + y));
		if (Fi == 28) return (4 - 2.1*x*x + x * x*x*x / 3)*x*x + x * y + (-4 + 4 * y*y)*y*y;
		if (Fi == 29) return sq(x - 1) + sq(2 * y*y - x);
	}

	return length(p) - 1.0;
};



// optimization header

#define _DEBUG_OPTIMIZATION
#include "optimization.h"

int main() {
	for (Fi = 0; Fi < 30; Fi++) {
		// iteration startpoints
		vec2 P0[4] = { vec2(4,3), vec2(3,-5), vec2(-8,0), vec2(-0.5,1) };

		for (int T = 0; T < 4; T++) {
			// initialize canvas
			init();
			drawContour(F, 1, .5, true);
			drawAxis(0.5, COLOR{ 128,128,128 }, true, true);

			drawDot(P0[T], 8, COLOR({ 232,0,0 }));
			char s[256];

			// Newton_Gradient_2d
			{
				vec2 p = P0[T];
				F_count = 0;
				p = Newton_Gradient_2d(F, p);
				drawDot(p, 8, COLOR({ 64,0,232 }));

				sprintf(s, "%d: (%.4lf,%.4lf,%.4lf)", F_count - 1, p.x, p.y, F(p));
				drawString(s, vec2(0, 4), 32, COLOR{ 128,0,128 });
			}

			// Newton_Iteration_2d
			{
				vec2 p = P0[T];
				F_count = 0;
				p = Newton_Iteration_2d(F, p);
				drawDot(p, 8, COLOR({ 160,64,0 }));

				sprintf(s, "%d: (%.4lf,%.4lf,%.4lf)", F_count - 1, p.x, p.y, F(p));
				drawString(s, vec2(0, 36), 32, COLOR{ 128,128,0 });
			}

			// save rendered image
			char file[] = "tests\\test00.0.png";
			file[13] = T + '1', file[10] = Fi / 10 + '0', file[11] = Fi % 10 + '0';
			save(file);
		}
		fflush(stdout);
	}
	return 0;
}

