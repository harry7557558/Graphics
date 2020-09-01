// physics: a 2d stretched string satisfying Hooke's law
// position along the x-axis is not equilibrium thus the equation is not wave equation
// I may get its equation wrong according to common sense
// (stiff equation?? Euler and Midpoint energy increase exponentially)

#include "numerical/ode.h"
#include "numerical/geometry.h"

// https://github.com/charlietangora/gif-h
#define GIF_FLIP_VERT
#include "libraries\gif.h"

#define W 800
#define H 250
unsigned img[H + 1][W];


void initImage() {
	for (int i = 0; i < W; i++) {
		for (int j = 0; j < H; j++) img[j][i] = 0xFFFFFF;
		img[H / 2][i] = 0x000000;
	}
}
int getHeight(double a) {
	int h = H * (0.25*a + 0.5);
	if (h < 0 || h >= H) return H;
	return h;
}
void plot(double x, double y, unsigned col) {
	int w = W * x;
	if (w >= 0 && w < W) img[getHeight(y)][w] = col;
}


typedef struct { vec2 p; vec2 v; } state;

const double k = 5.0 * W;  // spring constant
const double l0 = 1. / W;  // spring length

vec2 tension(vec2 a, vec2 b) {
	double l = length(b - a);
	return (b - a) * (k * (l - l0) / l);
}
void dxdt(double* x, double t, double* dx) {
	state *s = (state*)x;
	state *ds = (state*)dx;
	for (int i = 0; i <= W; i++) {
		ds[i].p = s[i].v;
	}
	ds[0].v = ds[W].v = vec2(0.0);
	for (int i = 1; i < W; i++) {
		ds[i].v = tension(s[i].p, s[i + 1].p) - tension(s[i - 1].p, s[i].p);
	}
}


int main() {
	if (sizeof(state) != 4 * sizeof(double)) {
		fprintf(stderr, "incorrect struct size\n");
		return -1;
	}
	GifWriter gif;
	GifBegin(&gif, "string1d.gif", W, H, 4);
	freopen("string1d.txt", "w", stdout);

	state S[W + 1];
	for (int i = 0; i <= W; i++) {
		double x = i / double(W);
		S[i].p = vec2(x, sin(PI*x) + sin(2 * PI*x));
		S[i].v = vec2(0.0);
	}

	const double dt = 0.01;
	double t = 0.0;
	double temp0[4 * W + 4], temp1[4 * W + 4], temp2[4 * W + 4];
	for (int f = 0; f < 200; f++) {
		//for (int u = 0; u < 40; u++) EulersMethod(dxdt, (double*)S, 4 * W + 4, t, .25*dt, temp0), t += .25*dt;
		//for (int u = 0; u < 20; u++) MidpointMethod(dxdt, (double*)S, 4 * W + 4, t, .5*dt, temp1, temp2), t += .5*dt;
		for (int u = 0; u < 10; u++) RungeKuttaMethod(dxdt, (double*)S, 4 * W + 4, t, dt, temp0, temp1, temp2), t += dt;

		// energy
		double E = 0;
		for (int i = 0; i <= W; i++) {
			E += 0.5*S[i].v.sqr();
		}
		for (int i = 0; i < W; i++) {
			E += 0.5*k*pow(length(S[i + 1].p - S[i].p) - l0, 2.);
		}
		printf("%.1lf\t%lf\n", t, E);

		// visualization
		initImage();
		for (int i = 0; i <= W; i++) {
			plot(S[i].p.x, S[i].p.y, 0x0000FF);
		}
		GifWriteFrame(&gif, (uint8_t*)img, W, H, 4);
	}

	GifEnd(&gif);
	return 0;
}

