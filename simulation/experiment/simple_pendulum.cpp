// test ODE solver
// physics: a 2d simple pendulum with unit length and unit mass
// start with zero velosity, last 10 seconds

// Compare Euler's method, Midpoint method, and Runge-Kutta method
// To be fair, Euler's method is called 4 times per step, Midpoint 2 times, and Runge-Kutta one time

// Result:
// Euler's method almost a chaos; NAN at the end
// Midpoint method has significant energy drift that makes the pendulum flies "over" when the angle is large
// Runge-Kutta method: visually perfect! maximum energy error 0.005


#include "numerical/ode.h"
#include "numerical/geometry.h"

// https://github.com/charlietangora/gif-h
#define GIF_FLIP_VERT
#include "libraries\gif.h"

#define IMG_W 800
#define IMG_H 200
unsigned img[IMG_H + 1][IMG_W];
GifWriter gif;

void initImage() {
	for (int i = 0; i < IMG_W; i++) {
		for (int j = 0; j < IMG_H; j++) img[j][i] = 0xFFFFFF;
		img[IMG_H / 2][i] = 0x000000;
	}
}
int getHeight(double a) {  // -1<=a<=1
	int h = IMG_H * (0.4*a + 0.5);
	if (h < 0 || h >= IMG_H) return IMG_H;
	return h;
}


typedef struct { vec2 p; vec2 v; } state;

vec2 acceleration(vec2 p, vec2 v) {
	double l = length(p);  // should be 1
	vec2 t = p.rot() * (-9.8 / l) * p.x;  // -g/l sin(θ)
	vec2 n = -p * v.sqr() / l;  // v²/l
	return t + n;
}
void dxdt(double* x, double t, double* dx) {
	state s = *(state*)x;
	state ds; ds.p = s.v, ds.v = acceleration(s.p, s.v);
	*(state*)dx = ds;
}

void simulate(vec2 x0, double dt) {
	initImage();
	state s; s.p = x0, s.v = vec2(0.0);
	double t = 0.0;
	double temp0[4], temp1[4], temp2[4];
	for (int i = 0; i < IMG_W; i++) {
		//for (int u = 0; u < 4; u++) EulersMethod(dxdt, (double*)&s, 4, t, .25*dt, temp0), t += .25*dt;
		//for (int u = 0; u < 2; u++) MidpointMethod(dxdt, (double*)&s, 4, t, .5*dt, temp1, temp2), t += .5*dt;
		for (int u = 0; u < 1; u++) RungeKuttaMethod(dxdt, (double*)&s, 4, t, dt, temp0, temp1, temp2), t += dt;
		double a = atan2(s.p.x, -s.p.y);  // angle of the pendulum
		double L = length(s.p);  // should be 1
		double E = 0.5*s.v.sqr() - 9.8*(x0.y - s.p.y);  // should be 0
		img[getHeight(a / PI)][i] = 0xFF0000;  // angle, blue/yellow
		img[getHeight(L - 1)][i] = 0x008000;  // length, green/magenta
		img[getHeight(E)][i] = 0x0000FF;  // energy, red/cyan
		if (i % 10 == 0) printf("%d %.3lf  %lf %lf  %lf %lf  %lf  %lf\n", i, t, s.p.x, s.p.y, s.v.x, s.v.y, L, E);
	}
	GifWriteFrame(&gif, (uint8_t*)img, IMG_W, IMG_H, 4);
}

int main() {
	if (sizeof(state) != 4 * sizeof(double)) {
		fprintf(stderr, "incorrect struct size\n");
		return -1;
	}
	GifBegin(&gif, "simple_pendulum.gif", IMG_W, IMG_H, 4);
	freopen("simple_pendulum.txt", "w", stdout);

	const int Frames = 100;
	for (int i = 0; i < Frames; i++) {
		double a = i * PI / Frames;
		double dt = 10. / IMG_W;
		printf("%d %lf %lf\n", i, a, dt);
		simulate(vec2(sin(a), -cos(a)), dt);
		printf("\n");
	}

	GifEnd(&gif);
	return 0;
}

