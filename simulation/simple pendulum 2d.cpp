#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626

// https://github.com/charlietangora/gif-h
#include "libraries\gif.h"
#pragma warning(disable:4996)


class vec2 {
public:
	double x, y;
	vec2() {}
	vec2(double x, double y) :x(x), y(y) {}
	~vec2() {}
	vec2 operator - () {
		return vec2(-x, -y);
	}
	vec2 operator + (vec2 v) {
		return vec2(x + v.x, y + v.y);
	}
	vec2 operator - (vec2 v) {
		return vec2(x - v.x, y - v.y);
	}
	vec2 operator * (double a) {
		return vec2(x*a, y*a);
	}
	double sqr() {
		return x * x + y * y;
	}
	friend double length(vec2 v) {
		return sqrt(v.x*v.x + v.y*v.y);
	}
	friend vec2 normalize(vec2 v) {
		return v * (1. / sqrt(v.x*v.x + v.y*v.y));
	}
	friend double dot(vec2 u, vec2 v) {
		return u.x*v.x + u.y*v.y;
	}
	friend double det(vec2 u, vec2 v) {
		return u.x*v.y - u.y*v.x;
	}
};


typedef unsigned int abgr;

const double g = 9.8;
const double r = 1.0;

void render(abgr* img, int w, vec2 p) {
	double l = 0.4*w, d; p = p * (l / r);
	vec2 u = normalize(p);
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < w; i++) {
			vec2 P(i - 0.5*w, w - 1 - j - 0.5*w);
			if (length(P) < 0.04*w) img[j*w + i] = 0xFF0000FF;
			else if (length(P - p) < 0.07*w) img[j*w + i] = 0xFF7F7F7F;
			else if ((d = dot(P, u)) > 0 && d < l && abs(det(P, u)) < 0.01*w) img[j*w + i] = 0xFF00FFFF;
			else img[j*w + i] = 0xFFFFFFFF;
		}
	}
}



int main() {
	const double dt = 0.001;
	const int w = 400;
	abgr* img = new abgr[w*w];
	GifWriter gif;
	GifBegin(&gif, "D:\\pendulum.gif", w, w, 4);
	FILE *of = fopen("D:\\s.txt", "w");

	auto test = [&](double theta0) {
		double t = 0;
		vec2 v(0.0, 0.0), p(r * sin(theta0), -r * cos(theta0));
		bool d = false;
		fprintf(of, "t\ta\t|a|\tv\t|v|\tp\t|p|=%lf\tθ in degrees\tE=%lf\t\n", r, p.y * g);
		for (int i = 0; i < 10000; i++) {
			double r1 = length(p);
			vec2 u = p * (1. / r1);
			vec2 ag = vec2(-u.y, u.x) * (-g * p.x / (r1*r1));
			vec2 ac = u * (-dot(v, v) / r1);
			vec2 a = ag + ac;
			t += dt, v = v + a * dt, p = p + v * dt;
			if (i != 0 && p.x >= 0 && v.y >= a.y * dt && v.y < 0) {
				if (!d) d = true;
				else { printf("%d\n", i); break; }
			}
			if (i % 40 == 0) {
				render(img, w, p);
				GifWriteFrame(&gif, (uint8_t*)img, w, w, 4);
			}
			fprintf(of, "%.4lf\t(%.4lf,%.4lf)\t%.5lf\t(%.4lf,%.4lf)\t%.5lf\t(%.4lf,%.4lf)\t%.8lf\t%.5lf°\t%.6lf\t\n", \
				t, a.x, a.y, length(a), v.x, v.y, length(v), p.x, p.y, length(p), 180.0 / PI * atan2(p.x, -p.y), 0.5*dot(v, v) + g * p.y);
		}
		fprintf(of, "\n");
	};

	test(0.1*PI);
	test(0.2*PI);
	test(0.35*PI);
	test(0.5*PI);
	test(0.75*PI);
	test(0.99*PI);


	fclose(of);
	GifEnd(&gif);
	delete img;
	return 0;
}

