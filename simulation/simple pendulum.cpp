#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626

// https://github.com/charlietangora/gif-h
#include "libraries\gif.h"

typedef unsigned int abgr;

const double g = 9.8;
const double r = 1.0;

void render(abgr* img, int w, double theta) {
	double l = 0.4*w, s = sin(theta), c = -cos(theta), bx = l * s, by = l * c;
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < w; i++) {
			double x = i - 0.5*w, y = w - 1 - j - 0.5*w, d;
			if (sqrt(x * x + y * y) < 0.04*w) img[j*w + i] = 0xFF0000FF;
			else if (sqrt((x - bx)*(x - bx) + (y - by)*(y - by)) < 0.07*w) img[j*w + i] = 0xFF7F7F7F;
			else if ((d = x * s + y * c) > 0 && d < l && abs(x * c - y * s) < 0.01*w) img[j*w + i] = 0xFF00FFFF;
			else img[j*w + i] = 0xFFFFFFFF;
		}
	}
}

int main() {
	const double dt = 0.01;
	const int w = 400;
	abgr* img = new abgr[w*w];
	GifWriter gif;
	GifBegin(&gif, "D:\\pendulum.gif", w, w, 4);

	auto test = [&](double a0) {
		double t = 0, v = 0, x = a0;
		bool d = false;
		for (int i = 0; i < 1000; i++) {
			double a = -g / r * sin(x);
			t += dt, v += a * dt, x += v * dt;
			if (i != 0 && v >= a * dt && v < 0) {
				if (!d) d = true;
				else { printf("%d\n", i); break; }
			}
			if (i % 4 == 0) {
				render(img, w, x);
				GifWriteFrame(&gif, (uint8_t*)img, w, w, 4);
			}
		}
	};

	test(0.1*PI);
	test(0.2*PI);
	test(0.35*PI);
	test(0.5*PI);
	test(0.75*PI);
	test(0.99*PI);

	GifEnd(&gif);
	delete img;
	return 0;
}

