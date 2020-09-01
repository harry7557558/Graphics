// most of code are for rendering not simulating

#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626

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
const double r = 0.25;
const double E = 0.85, e = 0.95; // coefficient of restitution
const vec2 box(2.0, 3.0);	// [-x, x], [0, y]

void render(abgr* img, int w, const vec2 *p, int N) {
	const double SC = 1.5 * length(box) / w;
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < w; i++) {
			vec2 P = vec2(i - 0.5*w, w - 1 - j - 0.5*w) * SC;
			P.y += 0.5*box.y;
			if (abs(P.x) < box.x && P.y >= 0 && P.y < box.y) {
				img[j*w + i] = 0xFFFFFFFF;
				for (int d = 0; d < N; d++) {
					if (length(P - p[d]) < r) {
						img[j*w + i] = 0xFF7F4F2F;
						break;
					}
				}
			}
			else img[j*w + i] = 0xFF7F7F7F;
		}
	}
}

int main() {
	const double dt = 0.001;
	const int w = 400;
	abgr* img = new abgr[w*w];
	GifWriter gif;
	GifBegin(&gif, "D:\\balls.gif", w, w, 4);
	FILE *fs = fopen("D:\\s.txt", "w");

	double t = 0;
	const int N = 3;
	vec2 p[3] = { vec2(0, box.y - r), vec2(0, r), vec2(1, 2.0*r) }, v[3] = { vec2(10, -5), vec2(0, 5), vec2(-3, 3) };
	const vec2 a(0, -g);
	auto reflect = [](vec2 &v, vec2 &p, vec2 n, double d, double e) {		// n·p - d = 0
		//vec2 v0 = v;
		//double t = (dot(n, p) - d) / dot(n, v);
		v = v - n * (2.0*dot(v, n));
		v = v * e;
		//p = p - n * (dot(n, p) - d);	// works worst
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
		if (i % 40 == 0) {
			render(img, w, p, N);
			GifWriteFrame(&gif, (uint8_t*)img, w, w, 4);
			//fprintf(fs, "%lf\t(%lf,%lf)\t(%lf,%lf)\t\n", t, p.x, p.y, v.x, v.y);
		}
	}

	fclose(fs);
	GifEnd(&gif);
	delete img;
	return 0;
}

