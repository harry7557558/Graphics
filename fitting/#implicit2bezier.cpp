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
class vec3 {
public:
	double x, y, z;
	vec3() {}
	vec3(const double &x, const double &y, const double &z) :x(x), y(y), z(z) {}
	vec3 operator - () const { return vec3(-x, -y, -z); }
	vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator * (const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }
	vec3 operator * (const double &k) const {
		return vec3(k * x, k * y, k * z);
	}
};

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

vec2 randvec(double a) {
	double m = rand() / double(RAND_MAX);
	m = a * atanh(2 * m - 1);
	double t = 2 * PI * rand() / double(RAND_MAX);
	return vec2(m*cos(t), m*sin(t));
}


//===================== Cubic Spline Class and Operations =====================//

class spline3 {
public:
	vec2 A, B, C, D;	// A t^3 + 3 B t^2 + 3 C t + D
	vec2 Min, Max;
	vec2 eval(const double &t) const {
		return ((A*t + B * 3.0)*t + C * 3.0)*t + D;
	}
	vec2 tangent(const double &t) const {
		return ((A*t + B * 2.0)*t + C)*3.0;
	}

	spline3() {}
	spline3(vec2 A, vec2 B, vec2 C, vec2 D) :A(A), B(B*(1. / 3)), C(C*(1. / 3)), D(D) {}
	spline3 invert() const {
		return spline3(-A, (A + B)*3.0, (A + B * 2.0 + C)*-3.0, A + D + (B + C)*3.0);
	}
};

spline3 fromSegment(vec2 A, vec2 B) {
	return spline3(vec2(0, 0), vec2(0, 0), B - A, A);
}
spline3 fromBezier2(vec2 A, vec2 B, vec2 C) {
	return spline3(vec2(0, 0), A - B * 2.0 + C, (B - A) * 2.0, A);
}
spline3 fromBezier3(vec2 A, vec2 B, vec2 C, vec2 D) {
	return spline3(D - C * 3.0 + B * 3.0 - A, (C - B * 2.0 + A)*3.0, (B - A)*3.0, A);
}




//======================== Fitting Implicit Functions ========================//


// any single-connected continuous function
double F(const vec2 &p) {
	double x = p.x, y = p.y, x2 = x * x, y2 = y * y, x3 = x2 * x, y3 = y2 * y, x4 = x3 * x, y4 = y3 * y;
	//return x * (x3 - 2) + y * (y3 - 2);  // triangular
	//return x3 * (x - 2) + y3 * (y - 2);  // triangular teeth
	//return x4 + y4 - x2 - y2 - x - y;  // another triangular teeth
	return x3 * (x - 2) + y3 * (y - 2) + x;	// teeth, incisor
	//return ((x - 2)*(x - 2) + y2 - 1)*(x2 + (y - 2)*(y - 2) - 1) - 2;	 // peanut
	//return pow(x2 + y2 - y, 2) - (x2 + y2);  // Descartes's heart
	//return pow(x2 + y2, 3) - 20 * x2*y2 - 1e-5;  // quadrifolium
	//return (x2 + y2 - 4)*(x2 + y2 - 1) + 2 * (x + y);	// crescent
}

// numerical gradient and curvature of the function
vec2 gradient(const vec2 &v) {
	const double e = 0.001;
	double dx = F(vec2(v.x + e, v.y)) - F(vec2(v.x - e, v.y));
	double dy = F(vec2(v.x, v.y + e)) - F(vec2(v.x, v.y - e));
	return vec2(dx, dy) * (0.5 / e);
}
double curvature(const vec2 &v) {
	const double e = 0.001;
	double p = F(v),
		x_ = F(vec2(v.x + e, v.y)), _x = F(vec2(v.x - e, v.y)),
		y_ = F(vec2(v.x, v.y + e)), _y = F(vec2(v.x, v.y - e)),
		xpyp = F(vec2(v.x + e, v.y + e)), xpym = F(vec2(v.x + e, v.y - e)),
		xmyp = F(vec2(v.x - e, v.y + e)), xmym = F(vec2(v.x - e, v.y - e));
	double dx = 0.5*(x_ - _x) / e;
	double dy = 0.5*(y_ - _y) / e;
	double dxx = (x_ + _x - 2 * p) / (e*e);
	double dyy = (y_ + _y - 2 * p) / (e*e);
	double dxy = (xpyp - xpym - xmyp + xmym) / (4 * e*e);
	return pow(dx*dx + dy * dy, 1.5) / abs(dy*dy*dxx + dx * dx*dyy - 2 * dx*dy*dxy);
}

// find a point on implicit curve using Newton's iteration method
bool settle(vec2 &p) {
	double dt, m; vec2 grad;
	int N = 0;
	do {
		grad = gradient(p);
		m = dot(grad, grad);
		dt = F(p) / m;
		p = p - grad * dt;
		if (isnan(dt) || ++N > 100) return false;
	} while (abs(dt)*sqrt(m) > 1e-6);
	return true;
}




#define W 900
#define H 600
#define Scale 100
#define Center vec2(0, 0.5)


// draw a spline
void rasterize(const spline3 &l, COLOR *Canvas, int N, COLOR col) {
	auto transform = [](vec2 p) ->vec2 { return vec2(p.x, -p.y)*Scale; };
	vec2 A = transform(l.A), B = transform(l.B*3.0), C = transform(l.C*3.0),
		D = transform(l.D) + vec2(0.5*W - Scale * Center.x, Scale*Center.y - (1 - 0.5*H));
#define M(x,y) Canvas[(y)*W+(x)]

	auto drawLine = [&](vec2 p, vec2 q) {
		vec2 d = q - p;
		double slope = d.y / d.x;
		if (abs(slope) <= 1.0) {
			if (p.x > q.x) swap(p, q);
			int x0 = max(0, int(p.x)), x1 = min(W - 1, int(q.x)), y;
			double yf = slope * x0 + (p.y - slope * p.x);
			for (int x = x0; x <= x1; x++) {
				y = (int)yf;
				if (y >= 0 && y < H) M(x, y) = col;
				yf += slope;
			}
		}
		else {
			slope = d.x / d.y;
			if (p.y > q.y) swap(p, q);
			int y0 = max(0, int(p.y)), y1 = min(H - 1, int(q.y)), x;
			double xf = slope * y0 + (p.x - slope * p.y);
			for (int y = y0; y <= y1; y++) {
				x = (int)xf;
				if (x >= 0 && x < W) M(x, y) = col;
				xf += slope;
			}
		}
	};

	for (int i = 0; i < N; i++) {
		double t = i / double(N), s = (i + 1.0) / N;
		vec2 p = ((A*t + B)*t + C)*t + D, q = ((A*s + B)*s + C)*s + D;
		drawLine(p, q);
	}

#undef M
}

// visualize implicit function, with axis
bool visualize() {
	COLOR *img = new COLOR[W*H];
	vec2 p;
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale) + Center;
			double d = abs(F(p)) / length(gradient(p));
			if (abs(p.x) < d) d = abs(p.x); if (abs(p.y) < d) d = abs(p.y);
			img[j*W + i] = toCOLOR(Scale * d);
		}
	}
	FILE *fp = fopen("D:\\Coding\\Graphics\\Implicit.png", "wb");
	if (fp == 0) {
		printf("\aOpen File Failed!\n\n");
		return false;
	}
	svpng(fp, W, H, (unsigned char*)img, false);
	fclose(fp);
	delete img;
	return true;
}

// visualize list of splines
bool visualize(const vector<spline3> &v) {
	COLOR *img = new COLOR[W*H];
	vec2 p;
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			p = vec2(i - 0.5*W, 0.5*H - (j + 1)) * (1.0 / Scale) + Center;
			double d = abs(F(p)) / length(gradient(p));
			if (abs(p.x) < d) d = abs(p.x); if (abs(p.y) < d) d = abs(p.y);
			img[j*W + i] = toCOLOR(max(0.01*Scale * d, 0.0) + 0.95);
		}
	}
	for (unsigned i = 0; i < v.size(); i++) {
		rasterize(v[i], img, 30, i & 1 ? toCOLOR(1, 0, 0) : toCOLOR(0, 0, 1));
	}
	FILE *fp = fopen("D:\\Coding\\Graphics\\Bezier.png", "wb");
	if (fp == 0) {
		printf("\aOpen File Failed!\n\n");
		return false;
	}
	svpng(fp, W, H, (unsigned char*)img, false);
	fclose(fp);
	delete img;
	return true;
}




double Integral(double(*f)(double), double a, double b, int n) {
	// Simpson's method may not work well for cubic and higher degree functions
	n *= 2;
	double u = (b - a) / n;
	double s = 0;
	for (int i = 1; i < n; i += 2) s += f(a + u * i);
	s *= 2;
	for (int i = 2; i < n; i += 2) s += f(a + u * i);
	s = 2 * s + f(a) + f(b);
	return s * u / 3;
}
double calcError(const spline3 &p, int n) {
	n *= 2;
	double u = 1.0 / n;
	double s = 0;
	auto f = [&](double t) ->double {
		double e = F(p.eval(t));
		return e * e * length(p.tangent(t));
	};
	for (int i = 1; i < n; i += 2) s += f(u * i);
	s *= 2;
	for (int i = 2; i < n; i += 2) s += f(u * i);
	s = 2 * s + f(0) + f(1);
	return s * u / 3;
}


// assume P and Q are on the implicit surface and u and v are normalized
bool fitImplicit(const vec2 P, vec2 u, const vec2 Q, vec2 v, vec2 &A, vec2 &B, double e) {
	v = -v;
	double a, b, a0, b0; a0 = b0 = a = b = 0.4*length(P - Q);
	//A = P + u * a, B = Q + v * b; return true;
	auto f = [&](double _a, double _b)->double {
		return calcError(fromBezier3(P, P + u * _a, Q + v * _b, Q), 10);
	};
	auto grad = [&](double _a, double _b)->vec2 {
		const double ep = 0.001;
		double dx = f(_a + ep, _b) - f(_a - ep, _b);
		double dy = f(_a, _b + ep) - f(_a, _b - ep);
		return vec2(dx, dy) * (0.5 / ep);
	};
	vec2 g; int N = 0;
	double k = 1.0;
	do {
		g = grad(a, b);
		a -= k * g.x, b -= k * g.y;
		if (!(a > 0 && b > 0 && a < 1e+6 && b < 1e+6)) {
			a = a0, b = b0, k *= 0.2;
			if (a < 1e-6) return false;
		}
		if (++N > 10000) return false;
	} while (length(g) > e);
	printf("%lf, %lf (%d)\n", a, b, N);
	if (!(a > 0 && b > 0 && a < 1e+6 && b < 1e+6)) return false;
	A = P + u * a, B = Q + v * b;
	printf("Error = %lf\n", calcError(fromBezier3(P, A, B, Q), 10));
	return true;
}


// core function
bool toBezier(double e, vector<spline3> &v) {
	vec2 p; int N = 0;
	p = randvec(1);
	if (!settle(p)) return false;
	vec2 p0 = p, q;
	double s = 0;
	for (int d = 0; d < 18; d++) {
		p0 = p;
		for (int i = 0; i < 5; i++) {
			vec2 d = gradient(p);
			d = normalize(vec2(-d.y, d.x));
			double cvt = 0.2*curvature(p);
			if (cvt > 0.2) cvt = 0.2;
			q = p + d * (cvt);
			if (!settle(q)) return false;
			//v.push_back(fromSegment(p, q));
			s += length(p - q);
			p = q;
		}
		vec2 A, B;
		bool fs = fitImplicit(p0, normalize(cross(gradient(p0))), p, normalize(cross(gradient(p))), A, B, e*e);
		if (fs) v.push_back(fromBezier3(p0, A, B, p));
		else printf("fail\n");
	}
	return true;
}



int main() {
	srand(1);
	visualize();
	vector<spline3> v;
	toBezier(0.01, v);
	visualize(v);
	return 0;
}
