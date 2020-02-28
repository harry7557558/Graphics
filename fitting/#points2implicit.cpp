#pragma GCC optimize "Ofast"

#include <cmath>
#include <stdio.h>
#include <vector>
using namespace std;

#pragma warning(disable:4996)

#define PI 3.1415926535897932384626
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

typedef unsigned char byte;
typedef struct { byte r, g, b; } COLOR;

class COLORf {
public:
	double r, g, b;
	COLORf() {}
	COLORf(const double &r, const double &g, const double &b) :r(r), g(g), b(b) {}
	COLORf(const COLOR &col) :r(col.r / 255.0), g(col.g / 255.0), b(col.b / 255.0) {}
	COLORf operator - () const { return COLORf(-r, -g, -b); }
	COLORf operator + (const COLORf &v) const { return COLORf(r + v.r, g + v.g, b + v.b); }
	COLORf operator - (const COLORf &v) const { return COLORf(r - v.r, g - v.g, b - v.b); }
	COLORf operator * (const COLORf &v) const { return COLORf(r * v.r, g * v.g, b * v.b); }
	COLORf operator * (const double &k) const { return COLORf(k * r, k * g, k * b); }
};
COLOR toCOLOR(double c) {
	if (c > 1.0) c = 1.0;
	else if (c < 0.0) c = 0.0;
	byte k = (byte)(255.0*c);
	COLOR r; r.r = r.b = r.g = k; return r;
}
COLOR toCOLOR(double r, double g, double b) {
	COLOR C;
	C.r = (byte)(255.0*(r > 1.0 ? 1.0 : r < 0.0 ? 0.0 : r));
	C.g = (byte)(255.0*(g > 1.0 ? 1.0 : g < 0.0 ? 0.0 : g));
	C.b = (byte)(255.0*(b > 1.0 ? 1.0 : b < 0.0 ? 0.0 : b));
	return C;
}
COLOR toCOLOR(const COLORf &col) {
	return toCOLOR(col.r, col.g, col.b);
}
COLOR mix(COLOR a, COLOR b, double d) {
	return toCOLOR(COLORf(a)*(1.0 - d) + COLORf(b)*d);
}


// https://github.com/miloyip/svpng.git
#include "svpng-master/svpng.inc"



//============================ Basic Definitions ============================//

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
vec2 cross(const vec2 &v) {
	return vec2(-v.y, v.x);
}
double det(const vec2 &u, const vec2 &v) {
	return u.x * v.y - u.y * v.x;
}
double length(const vec2 &v) {
	return sqrt(v.x * v.x + v.y * v.y);
}
double length2(const vec2 &v) {
	return v.x * v.x + v.y * v.y;
}
vec2 normalize(const vec2 &v) {
	double m = 1.0 / sqrt(v.x * v.x + v.y * v.y);
	return vec2(m * v.x, m * v.y);
}

double randval(double a) {
	return a * atanh(2.0 * (rand() + 0.5) / (RAND_MAX + 1.0) - 1);
}
vec2 randvec(double a) {
	double m = (rand() + 0.5) / (RAND_MAX + 1.0);
	m = a * atanh(2 * m - 1);
	double t = 2 * PI * rand() / double(RAND_MAX);
	return vec2(m*cos(t), m*sin(t));
}




//================================= Fitting =================================//


// https://people.cas.uab.edu/~mosya/cl/CM1nova.pdf
// https://www.emis.de/journals/BBMS/Bulletin/sup962/gander.pdf

#define EPSILON 1e-6

void fitLine(const vec2* p, int N, double &a, double &b, double &c) {	// ax+by+c=0
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0;
	for (int i = 0; i < N; i++) {
		double x = p[i].x, y = p[i].y;
		Sx += x, Sy += y, Sx2 += x * x, Sy2 += y * y, Sxy += x * y;
	}
#if 0
	// linear regression
	a = Sxy * N - Sx * Sy, b = Sx * Sx - Sx2 * N;
	if (abs(a) < abs(b)) c = Sx2 * Sy - Sx * Sxy;
	else b = a, a = Sy * Sy - Sy2 * N, c = Sy2 * Sx - Sy * Sxy;
	return;
#endif
	a = Sx * Sx - Sy * Sy + (Sy2 - Sx2) * N, b = 2.0*(Sxy * N - Sx * Sy);
	double t = -atan(b / a);
	if (a*cos(t) < b*sin(t)) t += PI;
	t *= 0.5;
	a = cos(t), b = sin(t);
	c = (Sx * a + Sy * b) / (-N);
}

bool fitCircle(const vec2* p, int N, vec2 &C, double &r) {
	double Sx = 0, Sy = 0, Sx2 = 0, Sy2 = 0, Sxy = 0, Sx3 = 0, Sy3 = 0, Sx2y = 0, Sxy2 = 0, Sx4 = 0, Sx2y2 = 0, Sy4 = 0;
	for (int i = 0; i < N; i++) {
		double x = p[i].x, y = p[i].y, x2 = x * x, y2 = y * y;
		Sx += x, Sy += y, Sx2 += x2, Sy2 += y2, Sxy += x * y,
			Sx3 += x2 * x, Sy3 += y2 * y, Sx2y += x2 * y, Sxy2 += x * y2,
			Sx4 += x2 * x2, Sx2y2 += x2 * y2, Sy4 += y2 * y2;
	}

	const double e = 1e-4;
	vector<vec2> R;
	unsigned Calls = 0;
	auto S = [&](double a, double b)->double {
		Calls++;
		double a2 = a * a, a3 = a2 * a, a4 = a2 * a2, b2 = b * b, b3 = b2 * b, b4 = b2 * b2;
		double r2 = ((a2*N - 2 * a*Sx + Sx2) + (b2*N - 2 * b*Sy + Sy2)) / N;
		return (a4 + 2 * a2*b2 + b4 - 2 * a2*r2 - 2 * b2*r2 + r2 * r2)*N \
			+ (-4 * a3 - 4 * a*b2 + 4 * a*r2)*Sx + (6 * a2 + 2 * b2 - 2 * r2)*Sx2 - 4 * a*Sx3 + Sx4 \
			+ (-4 * a2*b - 4 * b3 + 4 * b*r2)*Sy + (2 * a2 + 6 * b2 - 2 * r2)*Sy2 - 4 * b*Sy3 + Sy4 \
			+ 8 * a*b*Sxy - 4 * b*Sx2y - 4 * a*Sxy2 + 2 * Sx2y2;
	};
	// Newton's iteration minima finding
	unsigned M = 0; while (1) {
		vec2 X = randvec(1), dX;
		double da, db, da2, db2, dadb, det;
		unsigned n = 0; do {
			// samples
			double ab[3][3];
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
				ab[i][j] = S(X.x + (i - 1)*e, X.y + (j - 1)*e);
			}
			// first and second direction derivatives
			da = (ab[2][1] - ab[0][1]) / (2.0*e);
			db = (ab[1][2] - ab[1][0]) / (2.0*e);
			da2 = (ab[2][1] + ab[0][1] - 2 * ab[1][1]) / (e*e);
			db2 = (ab[1][2] + ab[1][0] - 2 * ab[1][1]) / (e*e);
			dadb = (ab[2][2] + ab[0][0] - ab[2][0] - ab[0][2]) / (4.0 * e*e);
			// iteration step
			det = da2 * db2 - dadb * dadb;
			if (0.0*det != 0.0) break;
			dX.x = (da * db2 - db * dadb) / det;
			dX.y = (da2 * db - dadb * da) / det;
			X = X - dX * 1.0;
			// check
			if (++n > 1000) break;
		} while (!(length(dX) < EPSILON));
		if (++n > 1000) break;
		if (++M > 1000 * (R.size() + 1)) break;
		if (0.0*det == 0.0) {
			unsigned i; for (i = 0; i < R.size(); i++) if (length(R[i] - X) < 100.0*EPSILON) break;
			if (i == R.size()) R.push_back(X);
		}
	}
	if (R.empty()) return false;
	int d = -1; double mind = INFINITY;
	for (unsigned i = 0; i < R.size(); i++) {
		double s = S(R[i].x, R[i].y);
		if (s < mind) mind = s, d = i;
	}
	C = R[d];
	r = sqrt(((C.x*C.x*N - 2 * C.x*Sx + Sx2) + (C.y*C.y*N - 2 * C.y*Sy + Sy2)) / N);
	return true;
}


// debug code
void minimaFinder() {
	const double e = 1e-4;
	vector<vec2> R;
	unsigned Calls = 0;
	auto F = [&](double x, double y)->double {
		return x * x - 4 * x*y + y * y + 2 * x + y - 1;
	};
	unsigned M = 0; while (1) {
		vec2 P = randvec(1), dP;
		double dx, dy, dx2, dy2, dxdy, det;
		unsigned n = 0; do {
			// samples
			double xy[3][3];
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
				xy[i][j] = F(P.x + (i - 1)*e, P.y + (j - 1)*e);
			}
			// first and second direction derivatives
			dx = (xy[2][1] - xy[0][1]) / (2.0*e);
			dy = (xy[1][2] - xy[1][0]) / (2.0*e);
			dx2 = (xy[2][1] + xy[0][1] - 2 * xy[1][1]) / (e*e);
			dy2 = (xy[1][2] + xy[1][0] - 2 * xy[1][1]) / (e*e);
			dxdy = (xy[2][2] + xy[0][0] - xy[2][0] - xy[0][2]) / (4.0 * e*e);
			// iteration step
			det = dx2 * dy2 - dxdy * dxdy;
			if (0.0*det != 0.0) break;
			dP.x = (dx * dy2 - dy * dxdy) / det;
			dP.y = (dx2 * dy - dxdy * dx) / det;
			P = P - dP * 1.0;
			// check
			if (++n > 1000) break;
		} while (!(length(dP) < EPSILON));
		if (++n > 1000) break;
		if (++M > 1000 * (R.size() + 1)) break;
		if (0.0*det == 0.0) {
			unsigned i; for (i = 0; i < R.size(); i++) if (length(R[i] - P) < 100.0*EPSILON) break;
			if (i == R.size()) R.push_back(P);
		}
	}
	for (unsigned i = 0; i < R.size(); i++) {
		double x = R[i].x, y = R[i].y;
		double z = F(x, y);
		double dx2 = (F(x + e, y) + F(x - e, y) - 2 * z) / (e*e);
		double dy2 = (F(x, y + e) + F(x, y - e) - 2 * z) / (e*e);
		if (dx2 < 0 && dy2 < 0) printf("maxima");
		else if (dx2 > 0 && dy2 > 0) printf("minima");
		else printf("saddle");
		printf(": (%lf, %lf, %lf)\n", x, y, z);
	}
}






//================================= Visualizing =================================//

COLOR* canvas;
#define W 900
#define H 600
#define Scale 50
#define Center vec2(0, 0)
const vec2 fromCoord(0.5*W - Scale * Center.x, Scale*Center.y - (0.5 - 0.5*H));
const vec2 fromScreen(-0.5*W / Scale + Center.x, (0.5*H - 0.5) / Scale + Center.y);


void drawAxis(double width, COLOR col) {	// low effeciency only for debug
	width *= 0.5;
	vec2 p;
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
			double d = min(abs(p.x), abs(p.y)) * Scale - width;
			if (d < 0) canvas[j*W + i] = col;
			else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
		}
	}
}
void drawDot(vec2 c, double r, COLOR col) {
	vec2 C = vec2(c.x, -c.y) * Scale + fromCoord;
	r -= 0.5;
	int i0 = max(0, (int)floor(C.x - r - 1)), i1 = min(W - 1, (int)ceil(C.x + r + 1));
	int j0 = max(0, (int)floor(C.y - r - 1)), j1 = min(H - 1, (int)ceil(C.y + r + 1));
	vec2 p;
	for (int j = j0; j <= j1; j++) {
		for (int i = i0; i <= i1; i++) {
			p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale) + Center;
			double d = length(p - c) * Scale - r;
			if (d < 0) canvas[j*W + i] = col;
			else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
		}
	}
}
void drawFittedLine(double a, double b, double c, double width, COLOR col) {
	double m = sqrt(a*a + b * b); a /= m, b /= m, c /= m;
	width *= 0.5;
	vec2 p;
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
		double d = abs(a*p.x + b * p.y + c) * Scale - width;
		if (d < 0) canvas[j*W + i] = col;
		else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
	}
}
void drawFittedCircle(vec2 c, double r, double width, COLOR col) {
	width *= 0.5;
	vec2 p;
	for (int j = 0; j < H; j++) for (int i = 0; i < W; i++) {
		p = vec2(i, -j) * (1.0 / Scale) + fromScreen;
		double d = abs(length(p - c) - r) * Scale - width;
		if (d < 0) canvas[j*W + i] = col;
		else if (d < 1) canvas[j*W + i] = mix(col, canvas[j*W + i], d);
	}
}


#include <time.h>
void init(int n) {
	canvas = new COLOR[W*H];
	for (int i = 0, l = W * H; i < l; i++) canvas[i].r = canvas[i].g = canvas[i].b = 255;
	printf("random number seed: %u\n", n);
	srand(n);
}
void save() {
	FILE *fp = fopen("D:\\Coding\\Graphics\\Test.png", "wb");
	if (fp == 0) printf("\aOpen File Failed!\n\n");
	else {
		svpng(fp, W, H, (unsigned char*)canvas, false);
		fclose(fp);
	}
	delete canvas;
	canvas = 0;
}


#include <chrono>
int main() {
	init((unsigned)time(0));
	//init(1575165484);

	const int N = 100;
	vec2 *p = new vec2[N];
	vec2 c = randvec(1.0);
	double r = rand()*3.0 / RAND_MAX + 1.5;
	double u = rand() / RAND_MAX + 0.5, v = rand() * 1.0 / RAND_MAX + 0.5;
	double a = rand() * 2 * PI / RAND_MAX;
	for (int i = 0; i < N; i++) {
		double t = a + randval(v);
		p[i] = c + vec2(cos(t), sin(t))*r + randvec(u);
	}
	drawFittedCircle(c, r, 8.0, toCOLOR(0.98));
	drawAxis(1.0, toCOLOR(0));

	{
		vec2 c; double r;
		auto t0 = chrono::high_resolution_clock::now();
		fitCircle(p, N, c, r);
		auto t1 = chrono::high_resolution_clock::now();
		double time_elapsed = chrono::duration<double>(t1 - t0).count();
		printf("Elapsed Time: %lfsecs\n", time_elapsed);
		drawFittedCircle(c, r, 2.0, toCOLOR(0, 0, 1));
	}
	for (int i = 0; i < N; i++) drawDot(p[i], 4.0, toCOLOR(1, 0, 0));

	save();
	delete p;
	return 0;
}

