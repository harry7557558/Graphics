#include <stdio.h>
#include <cmath>
#pragma warning(disable:4996)


#define PI 3.14159265358979
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))
#define fract(x) ((x)-floor(x))

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
	double sqr() const { return x * x + y * y; } 	// non-standard
	friend double length(vec2 v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(vec2 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(vec2 u, vec2 v) { return u.x*v.x + u.y*v.y; }
	friend double det(vec2 u, vec2 v) { return u.x*v.y - u.y*v.x; } 	// non-standard
};
class vec3 {
public:
	double x, y, z;
	vec3() {}
	vec3(double a) :x(a), y(a), z(a) {}
	vec3(double x, double y, double z) :x(x), y(y), z(z) {}
	vec3 operator - () const { return vec3(-x, -y, -z); }
	vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator * (const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); } 	// non-standard
	vec3 operator * (const double &k) const { return vec3(k * x, k * y, k * z); }
	double sqr() { return x * x + y * y + z * z; } 	// non-standard
	friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3 normalize(vec3 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y + v.z*v.z)); }
	friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
};

#define reflect(I, N) ((I)-(N)*(2.0*dot(N, I)))

double hash(double x) {
	return fract(sin(12.9898*x + 78.233)*43758.5453);
}
double hash(vec2 x) {
	return fract(sin(dot(x, vec2(12.9898, 78.233)))*43758.5453);
}
double hash(vec3 x) {
	return fract(sin(dot(x, vec3(12.9898, 78.233, 144.7272)))*43758.5453);
}

template<typename vec> vec Refract(vec I, vec N, double n1, double n2, double &R) {
	double eta = n1 / n2;
	double ci = -dot(N, I);
	if (ci < 0.) ci = -ci, N = -N;
	double ct = sqrt(1.0 - eta * eta * (1.0 - ci * ci));
	vec r = I * eta + N * (eta * ci - ct);
	double Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
	double Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
	R = 0.5 * (Rs * Rs + Rp * Rp);
	return r;
}


#define iTime 0.0
#define iFrame 0
#define iResolution vec2(640,360)

#define SAMPLE 64
#define EPSILON 1e-4
#define MAX_STEP 128
#define MAX_RECU 50

// Camera
vec3 POS = vec3(-10, -10, 2);  // camera ray origin
vec3 DIR = normalize(-POS) * 0.3;  // camera ray center direction, modulus represents the distance to the screen
double SCALE = 0.5;  // screen scaling, where 1 means the width and height of the screen forms a unit vector
vec3 SCR_O, SCR_A, SCR_B;  // P = O + uA + vB

#define BULB 10.0
#define INDEX 1.5
#define DENSITY 0.08  // fog, 1-exp(-d) chance of scattering per unit distance

bool intSphere(vec3 C, double R, vec3 p, vec3 d, double &t) {
	p = p - C;
	double a = dot(d, d), b = -dot(p, d) / a, c = (dot(p, p) - R * R) / a;
	double delta = b * b - c; if (delta < 0) return false;
	delta = sqrt(delta);
	double x1 = b + delta, x2 = b - delta;
	bool u = x1 < EPSILON, v = x2 < EPSILON;
	if (u && v) return false;
	t = u ? x2 : v ? x1 : x1 < x2 ? x1 : x2;
	//n = (p + d * t) * (1.0 / R);
	return true;
}

vec3 nSphere(vec3 C, vec3 p) {
	return normalize(p - C);
}

bool intBulb(vec3 p, vec3 d, double &t) {
	return intSphere(vec3(2.0), 1.0, p, d, t);
}

bool intObj(vec3 p, vec3 d, double &t, vec3 &n) {
	t = -p.z / d.z, n = vec3(0, 0, 1);
	return t > EPSILON;
}

bool inObj(vec3 p) {
	return p.z < 0.0;
}

vec3 gradient(double map(vec3), vec3 p) {
	const double eps = 1e-4;
	double a = map(vec3(p.x + eps, p.y + eps, p.z + eps));
	double b = map(vec3(p.x + eps, p.y - eps, p.z - eps));
	double c = map(vec3(p.x - eps, p.y + eps, p.z - eps));
	double d = map(vec3(p.x - eps, p.y - eps, p.z + eps));
	return normalize(vec3(a + b - c - d, a - b + c - d, a - b - c + d));
}

// rx and ry are uniform sampled random numbers between 0 and 1
vec3 randDir(vec3 n, double rx, double ry) {
	vec3 u = normalize(cross(n, vec3(1.0, 2.1, 3.2))), v = cross(u, n);
	return normalize((u * cos(2.0*PI*rx) + v * sin(2.0*PI*rx))*sqrt(ry) + n * sqrt(1.0 - ry));
}

// random direction
vec3 detour(vec3 p, vec3 d) {
	double a = 2.0*PI*hash(p.x + p.y), b = 2.0*hash(p.y + p.z) - 1.0;
	double c = sqrt(1.0 - b * b);
	return vec3(c*cos(a), c*sin(a), b);
}


double traceRay(vec3 p, vec3 d) {
	int N = 0;
	while (N++ < MAX_RECU) {
		double tb, to; vec3 n;
		bool ib = intBulb(p, d, tb);
		bool io = intObj(p, d, to, n);
		bool ins = inObj(p + d * EPSILON);
		double t = io ? (ib && tb < to ? tb : to) : (ib ? tb : 1e+12);
		double pb = exp(-DENSITY * t);
		if (!ins && hash(double(N) + t) > pb) {
			double rnd = hash(p.x * t + p.y);
			t = -log(1 - (1 - pb)*rnd) / DENSITY;
			p = p + d * t;
			d = detour(p, d);
		}
		else {
			if (io) {
				if (ib && tb < to) return BULB;
				else {
					vec3 q = p + d * to;
					vec3 r; double R;
					if (ins) r = Refract(d, n, INDEX, 1.0, R);
					else r = Refract(d, n, 1.0, INDEX, R);
					if (0.0*R != 0.0) R = 1.0;

					t = hash(q + d * R + iTime);
					if (t < R) p = q, d = reflect(d, n);
					else p = q, d = r;
					//d = randDir(n, hash(t), hash(p));
					//return 0.2;
				}
			}
			else return ib ? BULB : 0.0;
		}
	}
	return 0.0;
}


double Sample(vec2 coord) {
	double u = coord.x / iResolution.x, v = coord.y / iResolution.y;
	return traceRay(POS, normalize(SCR_O + SCR_A * u + SCR_B * v - POS));
}

double mainImage(vec2 coord) {
	double c = 0.0;
	for (int i = min(iFrame, 0); i < SAMPLE; i++) {
		double u = hash(double(i) + iTime - coord.x);
		double v = hash(double(i) - iTime + coord.y);
		c += Sample(coord + vec2(u, v));
	}
	return c / double(SAMPLE);
}




#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libraries\stb_image_write.h"

typedef unsigned char byte;
typedef struct { byte r, g, b; } rgb;

#include <chrono>
#include <thread>
#include <iostream>
#include <Windows.h>
#include <crtdbg.h>
#include <cstdlib>
#include <sstream>
#define dout(s) { std::wostringstream os_; os_ << s; OutputDebugStringW( os_.str().c_str() ); }	// log to Visual Studio output window
using namespace std;

void init() {
	vec3 cu = normalize(cross(DIR, vec3(0, 0, 1)));
	vec3 cv = normalize(cross(cu, DIR));
	vec3 cc = POS + DIR;
	vec2 R = normalize(iResolution) * SCALE;
	cu = cu * R.x, cv = cv * R.y;
	cc = cc - (cu + cv)*0.5;
	SCR_O = cc, SCR_A = cu, SCR_B = cv;
}

int main() {
	int w = (int)iResolution.x, h = (int)iResolution.y;
	byte *img = new byte[w*h];

	auto t0 = chrono::high_resolution_clock::now();

	init();

	const unsigned L = w * h;
	const unsigned MAX_THREADS = thread::hardware_concurrency();
	const unsigned ppt = 0x1000;
	const unsigned N = L / ppt;

	auto task = [&](unsigned beg, unsigned end, bool* sig) {
		double c; vec2 p;
		for (unsigned i = beg; i < end; i++) {
			c = 255.0*mainImage(vec2(i % w, h - 1 - i / w));
			if (c > 255.0) c = 255.0; if (c < 0.0) c = 0.0;
			img[i] = (byte)c;
		}
		*sig = true;
	};

	bool* fn = new bool[MAX_THREADS]; for (unsigned i = 0; i < MAX_THREADS; i++) fn[i] = false;
	thread** T = new thread*[MAX_THREADS]; for (unsigned i = 0; i < MAX_THREADS; i++) T[i] = NULL;

	unsigned released = 0, finished = 0;
	while (finished < N) {
		for (unsigned i = 0; i < MAX_THREADS; i++) {
			if (fn[i]) {
				fn[i] = false;
				delete T[i]; T[i] = 0;
				if (++finished >= N) break;
				cout << "\r" << finished << " / " << N;
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
	cout << "\r" << N << " / " << N << endl;

	delete fn;
	delete T;

	auto t1 = chrono::high_resolution_clock::now();
	double time_elapsed = chrono::duration<double>(t1 - t0).count();
	cout << time_elapsed << "secs elapsed. (" << (1.0 / time_elapsed) << "fps)\n";
	dout(endl << time_elapsed << "secs elapsed. (" << (1.0 / time_elapsed) << "fps)\n\n");

	stbi_write_bmp("D:\\SD.bmp", w, h, 1, img);
	delete img;
	return 0;
}

