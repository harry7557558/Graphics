#pragma GCC optimize "Ofast"

#include <cmath>
#include <stdio.h>
#include <chrono>
#include <thread>
using namespace std;

#pragma warning(disable:4996)

typedef unsigned char byte;
typedef struct { byte r, g, b; } COLOR;

COLOR toCOLOR(double c) {
	if (c > 1.0) c = 1.0;
	else if (c < 0.0) c = 0.0;
	unsigned char k = (unsigned char)(255.0*c);
	COLOR r; r.r = r.b = r.g = k; return r;
}

// https://github.com/miloyip/svpng.git
#include "svpng-master/svpng.inc"


#define PI 3.141592653589793

class vec2 {
public:
	double x, y;
	vec2() {}
	vec2(const double &x, const double &y) :x(x), y(y) {}
	vec2 operator - () const {
		return vec2(-x, -y);
	}
	vec2 operator + (const vec2 &v) const {
		return vec2(x + v.x, y + v.y);
	}
	vec2 operator - (const vec2 &v) const {
		return vec2(x - v.x, y - v.y);
	}
	vec2 operator * (const double &k) const {
		return vec2(k * x, k * y);
	}
};

double dot(const vec2 &u, const vec2 &v) {
	return u.x * v.x + u.y * v.y;
}
double length(const vec2 &v) {
	return sqrt(v.x * v.x + v.y * v.y);
}
vec2 normalize(const vec2 &v) {
	double m = 1.0 / sqrt(v.x * v.x + v.y * v.y);
	return vec2(m * v.x, m * v.y);
}

vec2 reflect(const vec2 &I, const vec2 &N) {
	return I - N * (2.0*dot(N, I));
}
vec2 refract(vec2 I, vec2 N, double n1, double n2, double &R) {
	double eta = n1 / n2;
	double ci = -dot(N, I);
	if (ci < 0) ci = -ci, N = -N;
	double ct = 1.0 - eta * eta * (1.0 - ci * ci);
	if (ct < 0) {
		R = -1; return vec2();
	}
	else ct = sqrt(ct);
	vec2 r = I * eta + N * (eta * ci - ct);
	double Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct); Rs *= Rs;
	double Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci); Rp *= Rp;
	R = 0.5 * (Rs + Rp);
	return r;
}

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))


#define IMG_W 600
#define IMG_H 400
#define SCALE 100.0
#define CENTER vec2(0.0, 0.0)
#define SAMPLE 256
#define EPSILON 1e-6
#define MAX_RECURSION 50
#define MIN_ERROR 0.01
#define BULB 2.0
#define INDEX 1.5

COLOR *img;

bool intBulb(vec2 p, vec2 d, double &t) {
	p = p - vec2(3.0, 3.0);
	double a = dot(d, d), b = -dot(p, d) / a, c = (dot(p, p) - 1) / a;
	double delta = b * b - c; if (delta < 0) return false;
	delta = sqrt(delta);
	double x1 = b + delta, x2 = b - delta;
	bool u = x1 < EPSILON, v = x2 < EPSILON;
	if (u && v) return false;
	t = u ? x2 : v ? x1 : x1 < x2 ? x1 : x2;
	return true;
}

bool intObj(vec2 p, vec2 d, double &t, vec2 &n) {
	double a = dot(d, d), b = -dot(p, d) / a, c = (dot(p, p) - 1) / a;
	double delta = b * b - c; if (delta < 0) return false;
	delta = sqrt(delta);
	double x1 = b + delta, x2 = b - delta;
	bool u = x1 < EPSILON, v = x2 < EPSILON;
	if (u && v) return false;
	t = u ? x2 : v ? x1 : x1 < x2 ? x1 : x2;
	n = normalize(p + d * t);
	return true;
}
bool inObj(const vec2 &p) {
	return length(p) < 1.0;
}

double traceRay(vec2 p, vec2 d, int N, double dm) {
	if (dm < MIN_ERROR) return 0.0f;
	if (N >= MAX_RECURSION) return 0.0;
	p = p + d * EPSILON;
	double t, tb; vec2 n;
	if (intObj(p, d, t, n)) {
		if (intBulb(p, d, tb) && tb <= t) return BULB;
		vec2 q = p + d * t;
		double R;
		if (inObj(p)) {
			vec2 r = refract(d, n, INDEX, 1.0, R);
			if (R == -1) return traceRay(q, reflect(d, n), N + 1, dm);
			double ot = traceRay(q, reflect(d, n), N + 1, R * dm);
			double it = traceRay(q, r, N + 1, (1.0 - R) * dm);
			return R * ot + (1.0 - R) * it;
		}
		else {
			vec2 r = refract(d, n, 1.0, INDEX, R);
			double ot = traceRay(q, reflect(d, n), N + 1, R * dm);
			double it = traceRay(q, r, N + 1, (1.0 - R) * dm);
			return R * ot + (1.0 - R) * it;
		}
	}
	if (intBulb(p, d, t)) return BULB;
	return 0.0;
}


int main() {
	img = new COLOR[IMG_W*IMG_H];

	auto t0 = chrono::high_resolution_clock::now();

	const unsigned L = IMG_W * IMG_H;
	const unsigned MAX_THREADS = thread::hardware_concurrency();
	const unsigned ppt = 0x1000;
	const unsigned N = L / ppt;

	auto task = [&](unsigned beg, unsigned end, bool* sig) {
		vec2 p;
		for (unsigned i = beg; i < end; i++) {
			p = vec2(i % IMG_W - 0.5*IMG_W, 0.5*IMG_H - (i / IMG_W + 1)) * (1.0 / SCALE) + CENTER * 0.5;
			double c = 0;
			double s = 1.0 / (SCALE * RAND_MAX), h = -0.5 / SCALE;
			for (int i = 0; i < SAMPLE; i++) {
				double a = 2.0 * PI * double(i + rand() / double(RAND_MAX)) / SAMPLE;
				vec2 d = vec2(cos(a), sin(a));
				c += traceRay(p + vec2(s * rand() + h, s * rand() + h), d, 0, 1.0);
			}
			img[i] = toCOLOR(c / SAMPLE);
		}
		*sig = true;
	};

	bool* fn = new bool[MAX_THREADS]; for (int i = 0; i < MAX_THREADS; i++) fn[i] = false;
	thread** T = new thread*[MAX_THREADS]; for (int i = 0; i < MAX_THREADS; i++) T[i] = NULL;

	unsigned released = 0, finished = 0;
	while (finished < N) {
		for (int i = 0; i < MAX_THREADS; i++) {
			if (fn[i]) {
				fn[i] = false;
				delete T[i]; T[i] = 0;
				if (++finished >= N) break;
				printf("\r%d / %d", finished, N);
			}
			if (!fn[i] && !T[i] && released < N) {
				T[i] = new thread(task, ppt * released, ppt * (released + 1), fn + i);
				T[i]->detach();
				released++;
			}
		}
		this_thread::sleep_for(0.001s);
	}
	task(N*ppt, L, fn);
	printf("\r%d / %d\n", N, N);

	delete fn;
	delete T;

	auto t1 = chrono::high_resolution_clock::now();
	double time_elapsed = chrono::duration<double>(t1 - t0).count();
	printf("%lfsecs elapsed. (%lffps)\n", time_elapsed, 1.0 / time_elapsed);

	FILE *fp = fopen("D:\\light2d\\test.png", "wb");
	svpng(fp, IMG_W, IMG_H, (unsigned char*)img, false);
	fclose(fp);

	delete img;
	return 0;
}
