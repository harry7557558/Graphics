#include <stdio.h>
#include <cmath>
#pragma warning(disable:4996)


#define PI 3.14159265358979
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

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
	double sqr() const { return x * x + y * y; } 	// not standard
	friend double length(vec2 v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(vec2 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(vec2 u, vec2 v) { return u.x*v.x + u.y*v.y; }
	friend double det(vec2 u, vec2 v) { return u.x*v.y - u.y*v.x; } 	// not standard
};

vec2 reflect(vec2 I, vec2 N) {
	return I - N * (2.0*dot(N, I));
}

#define fract(x) ((x)-floor(x))
#define iTime 5.0
#define iFrame 0
#define iResolution vec2(640,360)



#define SAMPLE 256
#define EPSILON 1e-4
#define MAX_STEP 64
#define MAX_DIST 10.0
#define MAX_RECU 50

vec2 CENTER = vec2(0.0, 0.0);
double SCALE = 100.0;
double BULB = 2.0;
double INDEX = 1.5;


double hash(vec2 x) {
	return fract(sin(dot(x, vec2(12.9898, 78.233)))*43758.5453);
}

vec2 Refract(vec2 I, vec2 N, double n1, double n2, double &R) {
	double eta = n1 / n2;
	double ci = -dot(N, I);
	if (ci < 0.) ci = -ci, N = -N;
	double ct = sqrt(1.0 - eta * eta * (1.0 - ci * ci));
	vec2 r = I * eta + N * (eta * ci - ct);
	double Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
	double Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
	R = 0.5 * (Rs * Rs + Rp * Rp);
	return r;
}

double sdBulb(vec2 p) {
	//return length(p - vec2(3.0)) - 1.0;
	//return length(vec2(abs(p.x) - 3.0, abs(p.y) - 3.0)) - 0.5;
	return length(vec2(p.x, p.y - 3.0)) - 1.0;
}

double sdObj(vec2 p) {
	//return length(p) - 1.0;	// circle
	//return (abs(p.x) > 0.8 ? length(vec2(abs(p.x) - 0.8, p.y)) : abs(p.y)) - 0.8;	// capsule
	//return max(abs(p.x) - 1.2, abs(p.y) - 0.75);	// rectangle
	//return max(length(vec2(p.x, p.y - 0.4)) - 1.0, p.y - 0.5);	// semi-circle
	//double k0 = length(vec2(0.618*p.x, p.y)); return k0 < 1.0 ? k0 - 1.0 : k0 * (k0 - 1.0) / length(vec2(0.382*p.x, p.y));	// ellipse (approximation)
	return min(max(abs(p.x), abs(p.y) - 1.2), max(abs(p.x) - 0.7, abs(p.y - 0.5))) - 0.2;	// cross
	//double d1 = (abs(p.y) > 1.0 ? length(vec2(abs(p.x) - 0.65, abs(p.y) - 1.0)) : abs(abs(p.x) - 0.65)) - 0.2, \
		d2 = max(abs(p.x) - 0.65, abs(p.y)) - 0.2; return min(d1, d2);	// letter H
}

vec2 gradient(vec2 p) {
	double k = 0.001;
	double u = sdObj(vec2(p.x + k, p.y)) - sdObj(vec2(p.x - k, p.y));
	double v = sdObj(vec2(p.x, p.y + k)) - sdObj(vec2(p.x, p.y - k));
	return vec2(u, v) * (0.5 / k);
}

double traceRay(vec2 p, vec2 d) {
	int N = 0;
	while (N++ < MAX_RECU) {
		double t = 10.0*EPSILON, dt, sdb, sdo, R;
		vec2 q, n, r;
		int i; for (i = 0; i < MAX_STEP; i++) {
			q = p + d * t;
			sdb = sdBulb(q);
			if (sdb <= EPSILON) return BULB;
			sdo = sdObj(q);
			dt = sdb > sdo ? sdo : sdb;
			if (abs(dt) <= EPSILON) {
				n = normalize(gradient(q)), r;
				if (dt >= 0.0) r = Refract(d, n, 1.0, INDEX, R);
				else r = Refract(d, n, INDEX, 1.0, R);
				if (0.0*R != 0.0) R = 1.0;
				break;
			}
			t += abs(dt);
			if (t > MAX_DIST) return 0.0;
		}
		if (i == MAX_STEP) return 0.0;

		t = hash(q + d * R + iTime);
		if (t < R) p = q, d = reflect(d, n);
		else p = q, d = r;
	}
	return 0.0;
}


double Sample(vec2 p) {
	double c = 0.0;
	double s = 1.0 / SCALE, h = -0.5 / SCALE;
	for (int i = 0; i < SAMPLE + min(iFrame, 0); i++) {
		double a = 2.0 * PI * (double(i) + hash(p + vec2(i) + iTime)) / double(SAMPLE);
		vec2 d = vec2(cos(a), sin(a));
		c += traceRay(p + vec2(hash(p + iTime - double(i))) * s, d);
	}
	return c / double(SAMPLE);
}

double mainImage(vec2 coord) {
	SCALE = 0.2*sqrt(iResolution.x*iResolution.y);
	vec2 p = (coord - iResolution * 0.5) * (1.0 / SCALE) + CENTER * 0.5;
	double c = Sample(p);
	return c;
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

int main() {
	int w = (int)iResolution.x, h = (int)iResolution.y;
	byte *img = new byte[w*h];

	auto t0 = chrono::high_resolution_clock::now();

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
