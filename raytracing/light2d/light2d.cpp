#pragma GCC optimize "Ofast"

#include <cmath>
#include <fstream>
#include <chrono>
#include <iostream>
#include <thread>
using namespace std;

typedef unsigned int COLORREF;

COLORREF toCOLORREF(double c) {
	if (c > 1.0) c = 1.0;
	else if (c < 0.0) c = 0.0;
	unsigned char k = (unsigned char)(255.0*c);
	return (COLORREF)(unsigned(k) | unsigned(k << 8) | unsigned(k << 16));
}

bool saveBitmap(COLORREF *data, unsigned w, unsigned h, string FilePath) {
	int padding = 4 - (w * 24 / 8) % 4;
	if (w * 24 % 8 != 0) padding--;
	padding %= 4;
	ofstream _Image(FilePath, ios_base::out | ios_base::binary);
	if (_Image.fail()) return false;
	_Image << "BM";	// Signature, BM
	unsigned size = w * h * 24 / 8 + h * padding + 54;
	unsigned os;
	_Image.write((char*)&size, 4);	// File Size
	os = 0; _Image.write((char*)&os, 4);	// reserved
	os = 0x36; _Image.write((char*)&os, 4);	// Full header size, usually 54 byte
	os = 0x28; _Image.write((char*)&os, 4);	// DIB header size, 40 byte
	_Image.write((char*)&w, 4);	// width
	_Image.write((char*)&h, 4);	// height
	os = 1; _Image.write((char*)&os, 2);	// planes, always 1
	os = 24; _Image.write((char*)&os, 2);	// bits per pixel
	os = 0; _Image.write((char*)&os, 4);	// compression
	size -= 54; _Image.write((char*)&size, 4);	// Image Size
	_Image.write((char*)&os, 4);	// X pixels per meter
	_Image.write((char*)&os, 4);	// Y pixels per meter
	_Image.write((char*)&os, 4);
	_Image.write((char*)&os, 4);
	COLORREF *b = data;
	for (unsigned i = 0; i < h; i++) {
		for (unsigned j = 0; j < w; j++) {
			_Image.write((char*)b, 3), b++;
		}
		_Image.write((char*)&os, padding);	// Pad row size to a multiple of 4
	}
	_Image.close();
	return true;
}


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
	double ct = sqrt(1.0 - eta * eta * (1.0 - ci * ci));
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
#define EPSILON 1e-5
#define MAX_STEP 64
#define MAX_DIST 10.0
#define MAX_RECURSION 50
#define MIN_ERROR 0.01
#define BULB 2.0
#define INDEX 1.5

COLORREF *img;

double sdBulb(const vec2 &p) {
	//return length(p - vec2(3.0, 3.0)) - 1.0;
	//return length(vec2(abs(p.x) - 3.0, abs(p.y) - 3.0)) - 0.5;
	return length(vec2(p.x, p.y - 3.0)) - 1.0;
}

double sdObj(const vec2 &p) {
	//return length(p) - 1.0;	// circle
	//return (abs(p.x) > 0.8 ? length(vec2(abs(p.x) - 0.8, p.y)) : abs(p.y)) - 0.8;	// capsule
	//vec2 d(abs(p.x) - 1.2, abs(p.y) - 0.75); return length(vec2(max(d.x, 0.0), max(d.y, 0.0))) + min(max(d.x, d.y), 0.0);	// rectangle
	//return max(length(vec2(p.x, p.y - 0.4)) - 1.0, p.y - 0.5);	// semi-circle
	//double k0 = length(vec2(0.618*p.x, p.y)); return k0 < 1.0 ? k0 - 1.0 : k0 * (k0 - 1.0) / length(vec2(0.382*p.x, p.y));	// ellipse (approximation)
	double d1 = max(abs(p.x), abs(p.y) - 1.2) - 0.2, d2 = max(abs(p.x) - 0.7, abs(p.y - 0.5)) - 0.2; return min(d1, d2);	// cross
	//double d1 = (abs(p.y) > 1.0 ? length(vec2(abs(p.x) - 0.65, abs(p.y) - 1.0)) : abs(abs(p.x) - 0.65)) - 0.2, \
		d2 = max(abs(p.x) - 0.65, abs(p.y)) - 0.2; return min(d1, d2);	// letter H
}

vec2 gradient(const vec2 &p) {
	const double k = 0.001;
	double u = sdObj(vec2(p.x + k, p.y)) - sdObj(vec2(p.x - k, p.y));
	double v = sdObj(vec2(p.x, p.y + k)) - sdObj(vec2(p.x, p.y - k));
	return vec2(u, v) * (0.5 / k);
}

double traceRay(vec2 p, vec2 d, int N, double dm) {
	if (dm < MIN_ERROR) return 0.0f;
	double t = 10.0*EPSILON, dt, sdb, sdo;
	vec2 q;
	for (int i = 0; i < MAX_STEP; i++) {
		q = p + d * t;
		sdb = sdBulb(q);
		if (sdb <= EPSILON) return BULB;
		sdo = sdObj(q);
		dt = sdb > sdo ? sdo : sdb;
		if (abs(dt) <= EPSILON) {
			if (N >= MAX_RECURSION) return 0.0;
			vec2 n = normalize(gradient(q));
			double R;
			if (dt >= 0) {
				vec2 r = refract(d, n, 1.0, INDEX, R);
				double ot = traceRay(q, reflect(d, n), N + 1, R * dm);
				double it = traceRay(q, r, N + 1, (1.0 - R) * dm);
				return R * ot + (1.0 - R) * it;
			}
			else {
				vec2 r = refract(d, n, INDEX, 1.0, R);
				if (0.0*R != 0.0) return traceRay(q, reflect(d, n), N + 1, dm);
				double ot = traceRay(q, reflect(d, n), N + 1, R * dm);
				double it = traceRay(q, r, N + 1, (1.0 - R) * dm);
				return R * ot + (1.0 - R) * it;
			}
		}
		t += abs(dt);
		if (t > MAX_DIST) break;
	}
	return 0.0;
}

double sample(vec2 p) {
	double c = 0;
	double s = 1.0 / (SCALE * RAND_MAX), h = -0.5 / SCALE;
	for (int i = 0; i < SAMPLE; i++) {
		//double a = rand();    // uniform
		//double a = 2.0 * PI * double(i) / SAMPLE;    // stratified
		double a = 2.0 * PI * double(i + rand() / double(RAND_MAX)) / SAMPLE;    // jittered
		vec2 d = vec2(cos(a), sin(a));
		//c += traceRay(p, d, 0);    // without anti-aliasing
		c += traceRay(p + vec2(s * rand() + h, s * rand() + h), d, 0, 1.0);		// anti-aliasing may fail if the light is too bright
	}
	return c / SAMPLE;
}

int main() {
	img = new COLORREF[IMG_W*IMG_H];

	auto t0 = chrono::high_resolution_clock::now();

	const unsigned L = IMG_W * IMG_H;
	const unsigned MAX_THREADS = thread::hardware_concurrency();
	const unsigned ppt = 0x1000;
	const unsigned N = L / ppt;

	auto task = [&](unsigned beg, unsigned end, bool* sig) {
		double c; vec2 p;
		for (unsigned i = beg; i < end; i++) {
			p = vec2(i % IMG_W - 0.5*IMG_W, i / IMG_W - 0.5*IMG_H) * (1.0 / SCALE) + CENTER * 0.5;
			c = sample(p);
			img[i] = toCOLORREF(c);
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

	saveBitmap(img, IMG_W, IMG_H, "D:\\light2d\\test.bmp");
	delete img;
	return 0;
}
