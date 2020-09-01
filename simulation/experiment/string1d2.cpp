// physics: a 2d stretched string
// the string is a discretized as a long grid of springs satisfying Hooke's law
// gravity and air resistance (proper to v²) is added to the equation
// there are still vibrations even the string turns calm; I assume it can be reduced by adding spring damping

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
		img[H / 2][i] = 0x808080;
	}
}
int getHeight(double a) {
	int h = H * (0.25*a + .5);
	if (h < 0 || h >= H) return H;
	return h;
}
void plot(double x, double y, unsigned col) {
	int w = W * x;
	if (w >= 0 && w < W) img[getHeight(y)][w] = col;
}


#define SZ (2*W+2)
typedef struct { vec2 p; vec2 v; } state;

const double k = 10.0 * W;  // spring constant
const double l0 = 1. / W;  // spring length
const vec2 g(0, -0.05);  // gravity
const double inv_m = (.001 / l0);  // reciprocal of mass

vec2 tension(vec2 a, vec2 b) {
	double l = length(b - a);
	vec2 T = (b - a) * (k * (l - l0) / l);
	return T * inv_m;  // return acceleration
}
void dxdt(double* x, double t, double* dx) {
	state *s = (state*)x;
	state *ds = (state*)dx;
	state *s1 = s + (W + 1);
	state *ds1 = ds + (W + 1);

	// first derivative equals to velocity
	for (int i = 0; i < SZ; i++) {
		ds[i].p = s[i].v;
	}
	ds[0].v = ds[W].v = vec2(0.0);
	ds1[0].v = ds1[W].v = vec2(0.0);

	// spring force
	vec2 T_prev = tension(s[0].p, s[1].p), T1_prev = tension(s1[0].p, s1[1].p);  // tension of previous calculation
	for (int i = 1; i < W; i++) {
		vec2 T = tension(s[i].p, s[i + 1].p);
		ds[i].v = T - T_prev, T_prev = T;
		vec2 T1 = tension(s1[i].p, s1[i + 1].p);
		ds1[i].v = T1 - T1_prev, T1_prev = T1;
		vec2 f = tension(s[i].p, s1[i].p);
		ds[i].v += f, ds1[i].v -= f;
	}
	// damping force (air resistance)
	for (int i = 0; i < SZ; i++) {
		double a = -0.0008 / l0 * s[i].v.sqr();
		if (a != 0.0) ds[i].v += a * normalize(s[i].v);
	}
	// gravity
	for (int i = 1; i < W; i++) {
		ds[i].v += g, ds1[i].v += g;
	}
}


#include <chrono>

int main() {
	if (sizeof(state) != 4 * sizeof(double)) {
		fprintf(stderr, "incorrect struct size\n");
		return -1;
	}
	GifWriter gif;
	GifBegin(&gif, "string1d2.gif", W, H, 4);
	freopen("string1d2.txt", "w", stdout);

	// initialization
	state S[SZ];
	state *S1 = &S[W + 1];
	for (int i = 0; i <= W; i++) {
		double x = i / double(W);
		vec2 p = vec2(x, sin(PI*x) + sin(2 * PI*x));
		S[i].p = p + 0.5*vec2(0, l0);
		S1[i].p = p - 0.5*vec2(0, l0);
		S[i].v = S1[i].v = vec2(0.0);
	}

	double time_elapsed = 0;

	const double dt = 0.005;
	double t = 0.0;
	double temp0[4 * SZ], temp1[4 * SZ], temp2[4 * SZ];
	for (int f = 0; f < 200; f++) {
		// simulation with time recording
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int u = 0; u < 40; u++) RungeKuttaMethod(dxdt, (double*)S, 4 * SZ, t, dt, temp0, temp1, temp2), t += dt;
		time_elapsed += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

		// energy
		double E = 0;
		for (int i = 0; i < SZ; i++) {  // velocity
			E += 0.5*S[i].v.sqr();
		}
		for (int i = 0; i < W; i++) {  // "parallel" springs
			E += 0.5*k*pow(length(S[i + 1].p - S[i].p) - l0, 2.)*inv_m;
			E += 0.5*k*pow(length(S1[i + 1].p - S1[i].p) - l0, 2.)*inv_m;
		}
		for (int i = 1; i < W; i++) {  // "cross" springs
			E += 0.5*k*pow(length(S[i].p - S1[i].p) - l0, 2.)*inv_m;
		}
		for (int i = 0; i < SZ; i++) {  // gravity potential
			E -= dot(S[i].p, g);
		}
		printf("%.1lf\t%lf\n", t, E);

		// visualization
		initImage();
		for (int i = 0; i < SZ; i++) {
			plot(S[i].p.x, S[i].p.y, 0x0000FF);
		}
		GifWriteFrame(&gif, (uint8_t*)img, W, H, 4);
	}

	printf("\nsimulation time: %lf secs\n", time_elapsed);

	GifEnd(&gif);
	return 0;
}

