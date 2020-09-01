// most of code are for rendering not simulating

#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

// https://github.com/charlietangora/gif-h
#include "libraries\gif.h"
#pragma warning(disable:4996)

typedef unsigned char byte;
typedef unsigned int abgr;

class vec2 {
public:
	double x, y;
	vec2() {}
	vec2(double a) :x(a), y(a) {}
	vec2(double x, double y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (vec2 v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (vec2 v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (double a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; }
	friend double length(vec2 v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(vec2 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(vec2 u, vec2 v) { return u.x*v.x + u.y*v.y; }
	friend double det(vec2 u, vec2 v) { return u.x*v.y - u.y*v.x; }
};

const double g = 9.8;
const double r = 0.025;
const double E = 0.7, e = 0.99; // coefficient of restitution
const vec2 box(2.0, 3.0);	// [-x, x], [0, y]

void render(abgr* img, int w, const vec2 *p, int N) {
	const double SC = 1.5 * length(box) / w;
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < w; i++) {
			vec2 P = vec2(i - 0.5*w, 0.5*w - j) * SC;
			if (abs(P.x) < box.x && P.y >= -0.5*box.y && P.y < 0.5*box.y) img[j*w + i] = 0xFFFFFFFF;
			else img[j*w + i] = 0xFF7F7F7F;
		}
	}
	double R = r / SC;
	for (int d = 0; d < N; d++) {
		vec2 c = (p[d] - vec2(0.0, 0.5*box.y)) * (1.0 / SC); c.x += 0.5*w, c.y = 0.5*w - c.y;
		int i0 = max(0, (int)floor(c.x - R - 1)), i1 = min(w - 1, (int)ceil(c.x + R + 1));
		int j0 = max(0, (int)floor(c.y - R - 1)), j1 = min(w - 1, (int)ceil(c.y + R + 1));
		for (int i = i0; i < i1; i++) {
			for (int j = j0; j < j1; j++) {
				if (length(vec2(i, j) - c) < R) img[j*w + i] = 0xFF7F4F2F;
			}
		}
	}
}

#include <chrono>

int main() {
	const double dt = 0.001;
	const int w = 1000;
	abgr* img = new abgr[w*w];
	GifWriter gif;
	GifBegin(&gif, "D:\\balls.gif", w, w, 4);

	double tot_t = 0.0, sim_t = 0.0, rnd_t = 0.0, enc_t = 0.0;
	auto t0 = chrono::high_resolution_clock::now();

	double t = 0;
	const int N = 400;
	vec2 *p = new vec2[N], *v = new vec2[N];
	/*const double o = 1.1;
	p[0] = vec2(-box.x + o * r, o*r), v[0] = vec2(0.0, 0.0);
	for (int i = 1; i < N; i++) {
		p[i] = p[i - 1] + vec2(0.0, 2.0*o*r);
		if (p[i].y > box.y - o * r) p[i].y = o * r, p[i].x += 2.0*o * r;
		v[i] = vec2(0.1*sin(378234.2*i), 0.1*cos(283234.23*i));
	}*/
	p[0] = vec2(-box.x + r, r), v[0] = vec2(5.0, 5.0);
	double d = r, h = r;
	for (int i = 1; i < N; i++) {
		if (i & 1) p[i] = vec2(-d, h);
		else {
			p[i] = vec2(d, h);
			h += 2.0*r;
			if (h > box.y - 2.0*r) h = r, d += 2.00001*r;
		}
		v[i] = vec2(0.0);
	}

	const vec2 a(0, -g);
	auto reflect = [](vec2 &v, vec2 &p, vec2 n, double d, double e) {		// n·p - d = 0
		//vec2 v0 = v;
		//double t = (dot(n, p) - d) / dot(n, v);
		v = v - n * (2.0*dot(v, n));
		v = v * e;
		//p = p - n * (dot(n, p) - d);	// most stable
		p = p - n * (2.0*(dot(n, p) - d));	// works best
		//p = p + (v - v0) * t;		// most reasonable but not best
	};
	auto collide = [](vec2 &p1, vec2 &p2, vec2 &v1, vec2 &v2, double e) {	// two spheres with same mass
		vec2 u1 = v1, u2 = v2;
		vec2 n = normalize(p1 - p2);
		v1 = n * dot(u2 - u1, n) + u1;
		v2 = u2 - v1 + u1;
		v1 = v1 * e, v2 = v2 * e;
		double d = 2.0 * r - length(p1 - p2);
		p1 = p1 + n * d, p2 = p2 - n * d;
	};
	for (int i = 0; i < 10000; i++) {
		auto s0 = chrono::high_resolution_clock::now();
		for (int d = 0; d < N; d++) {
			v[d] = v[d] + a * dt, p[d] = p[d] + v[d] * dt;
			if (p[d].y < r) reflect(v[d], p[d], vec2(0, 1), r, E);
			if (p[d].x > box.x - r) reflect(v[d], p[d], vec2(1, 0), box.x - r, E);
			if (p[d].y > box.y - r) reflect(v[d], p[d], vec2(0, 1), box.y - r, E);
			if (p[d].x < -box.x + r) reflect(v[d], p[d], vec2(1, 0), -box.x + r, E);
		}
		for (int c = 0; c < N; c++) {
			for (int d = c + 1; d < N; d++) {
				if (length(p[d] - p[c]) < 2.0*r) collide(p[c], p[d], v[c], v[d], e);
			}
		}
		t += dt;
		auto s1 = chrono::high_resolution_clock::now();
		sim_t += chrono::duration<double>(s1 - s0).count();
		if (i % 40 == 0) {
			s0 = chrono::high_resolution_clock::now();
			render(img, w, p, N);
			s1 = chrono::high_resolution_clock::now();
			rnd_t += chrono::duration<double>(s1 - s0).count();
			GifWriteFrame(&gif, (uint8_t*)img, w, w, 4);
			s0 = chrono::high_resolution_clock::now();
			enc_t += chrono::duration<double>(s0 - s1).count();
		}
	}

	auto t1 = chrono::high_resolution_clock::now();
	tot_t += chrono::duration<double>(t1 - t0).count();
	printf("tot_t: %lf\nsim_t: %lf\nrnd_t: %lf\nenc_t: %lf\n", tot_t, sim_t, rnd_t, enc_t);

	delete p; delete v;

	GifEnd(&gif);
	delete img;
	return 0;
}

