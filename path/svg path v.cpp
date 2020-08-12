#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

#pragma warning(disable:4996)

#define PI 3.1415926535897932384626
#define _13 0.333333333333333333333


//============================ Basic Definitions ============================//

class vec2 {
public:
	double x, y;
	vec2() {}
	vec2(double a) :x(a), y(a) {}
	vec2(double x, double y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (vec2 v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (vec2 v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (vec2 v) const { return vec2(x * v.x, y * v.y); } 	// non-standard
	vec2 operator * (double a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; } 	// non-standard
	friend double length(vec2 v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(vec2 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(vec2 u, vec2 v) { return u.x*v.x + u.y*v.y; }
	friend double det(vec2 u, vec2 v) { return u.x*v.y - u.y*v.x; } 	// non-standard
	vec2 rot() const { return vec2(-y, x); }  // rotate 90deg counterclockwise
};


// 2x3 matrix, linear transform
class mat2x3 {
public:
	double a, b, c, d, x, y;
	mat2x3() :a(1), b(0), c(0), d(1), x(0), y(0) {}
	mat2x3(double a, double b, double c, double d) :a(a), b(b), c(c), d(d), x(0), y(0) {}
	mat2x3(double s) :a(s), b(0), c(0), d(s), x(0), y(0) {}
	mat2x3(double x, double y) :a(1), b(0), c(0), d(1), x(x), y(y) {}
	mat2x3(double a, double b, double c, double d, double x, double y) :a(a), b(b), c(c), d(d), x(x), y(y) {}
	mat2x3 operator * (const mat2x3 &m) const {
		return mat2x3(a*m.a + b * m.c, a*m.b + b * m.d, c*m.a + d * m.c, c*m.b + d * m.d,
			a*m.x + b * m.y + x, c*m.x + d * m.y + y);
	}
	vec2 applyTo(const vec2 &v) const {
		return vec2(a*v.x + b * v.y + x, c*v.x + d * v.y + y);
	}
	vec2 operator * (const vec2 &v) const {
		return vec2(a*v.x + b * v.y, c*v.x + d * v.y);
	}
};
// operator*(vec2): no translation
// applyTo(vec2): have translation

mat2x3 RotationMatrix(double a) { return mat2x3(cos(a), -sin(a), sin(a), cos(a)); }
#define radians(deg) ((deg)*0.017453292519943295)


//===================== Cubic Spline Class and Operations =====================//

class spline3 {
public:
	vec2 A, B, C, D;	// A t^3 + 3 B t^2 + 3 C t + D
	vec2 Min, Max;
	void calcRange() {
		Min = D, Max = A + B + C + D;
		if (Min.x > Max.x) swap(Min.x, Max.x); if (Min.y > Max.y) swap(Min.y, Max.y);
		auto getMaxMin = [](double a, double b, double c, double d, double &Min, double &Max) {
			if (abs(a) < 1e-8) {
				if (abs(b) < 1e-8) return;
				double t = -c / (2.0 * b), x;
				if (t > 0 && t < 1) {
					x = ((a*t + b)*t + c)*t + d;
					if (x < Min) Min = x; if (x > Max) Max = x;
				}
				return;
			}
			double delta = b * b - 3.0 * a * c;
			if (delta >= 0) {
				delta = sqrt(delta);
				double t = (-b - delta) / (3.0*a), x;
				if (t > 0 && t < 1) {
					x = ((a*t + b)*t + c)*t + d;
					if (x < Min) Min = x; if (x > Max) Max = x;
				}
				t = (-b + delta) / (3.0*a);
				if (t > 0 && t < 1) {
					x = ((a*t + b)*t + c)*t + d;
					if (x < Min) Min = x; if (x > Max) Max = x;
				}
			}
		};
		getMaxMin(A.x, B.x, C.x, D.x, Min.x, Max.x);
		getMaxMin(A.y, B.y, C.y, D.y, Min.y, Max.y);
	}
	vec2 eval(const double &t) const {
		return ((A*t + B)*t + C)*t + D;
	}

	spline3() {}
	spline3(vec2 A, vec2 B, vec2 C, vec2 D) :A(A), B(B), C(C), D(D) {}
	void applyMatrix(const mat2x3 &M) {
		A = M * A, B = M * B, C = M * C, D = M.applyTo(D);
	}
	spline3 invert() const {
		return spline3(-A, A * 3.0 + B, A * -3.0 + B * -2.0 - C, A + B + C + D);
	}
};

spline3 fromSegment(vec2 A, vec2 B) {
	return spline3(0, 0, B - A, A);
}
spline3 fromBezier2(vec2 A, vec2 B, vec2 C) {
	return spline3(0, A - B * 2.0 + C, (B - A) * 2.0, A);
}
spline3 fromBezier3(vec2 A, vec2 B, vec2 C, vec2 D) {
	return spline3(D - C * 3.0 + B * 3.0 - A, (C - B * 2.0 + A)*3.0, (B - A)*3.0, A);
}
spline3 fromArc(double a, mat2x3 T) {  // fitting, unit arc from angle 0 to a
	double S = sin(a), C = cos(a);
	double s2 = S * S, c2 = C * C, sc2 = s2 + c2, sc22 = sc2 * sc2;
	a = 1. / (756.*(sc22 + 1.) + 810.*s2 - 1890.*(sc2 + 1.)*C + 2430.*c2);
	double c = (2520.*sc22 + 2736.*s2 + (-507.*sc2 - 6600.*C + 7215.)*C - 2628.) * a,
		b = (3996.*(sc2 + 1.) - 6750.*C)*S * a, d = (3439.*sc2 + 4276.*C - 7715.)*S * a;
	double p = (c - b * b * _13) * _13, q = -0.5 * ((b*b / 13.5 - c * _13) * b + d);
	a = q * q + p * p * p;
	double x = a > 0.0 ? cbrt(q + sqrt(a)) + cbrt(q - sqrt(a)) - b * _13
		: 2.0 * pow(q*q - a, 1. / 6) * cos(atan2(sqrt(-a), q) *_13) - b * _13;
	spline3 R = fromBezier3(vec2(1, 0), vec2(1, x), vec2(C + x * S, S - x * C), vec2(C, S));
	R.applyMatrix(T);
	return R;
}
void fromArc(vec2 c, vec2 r, double t0, double t1, double rot, vector<spline3> &v) {
	double dt = t1 - t0;
	int n = abs(dt) < 0.5 ? 1 : int((abs(dt) - 0.5) / (PI / 2)) + 1;
	dt /= n;
	mat2x3 B = RotationMatrix(t0);
	if (dt < 0) B = B * mat2x3(1, 0, 0, -1);
	spline3 s = fromArc(abs(dt), B), d;
	mat2x3 R = RotationMatrix(dt);
	mat2x3 T = mat2x3(c.x, c.y) * RotationMatrix(rot) * mat2x3(r.x, 0, 0, r.y);
	for (int i = 0; i < n; i++) {
		d = s; d.applyMatrix(T);
		v.push_back(d);
		s.applyMatrix(R);
	}
}

int toBezier(const spline3 &p, vec2 &A, vec2 &B, vec2 &C, vec2 &D) {
	int N = p.A.sqr() > 1e-12 ? 3 : p.B.sqr() > 1e-12 ? 2 : p.C.sqr() > 1e-12 ? 1 : 0;
	if (N == 0) A = p.D;
	else if (N == 1) A = p.D, B = A + p.C;
	else if (N == 2) A = p.D, B = p.C*0.5 + A, C = p.B - A + B * 2.0;
	else if (N == 3) A = p.D, B = p.C*_13 + A, C = p.B*_13 + B * 2.0 - A, D = p.A + (C - B)*3.0 + A;
	return N;
};

void getRange(vector<spline3> &v, vec2 &Min, vec2 &Max) {
	Min = vec2(INFINITY, INFINITY), Max = -Min;
	for (int i = 0, n = v.size(); i < n; i++) {
		v[i].calcRange();
		if (v[i].Min.x < Min.x) Min.x = v[i].Min.x;
		if (v[i].Min.y < Min.y) Min.y = v[i].Min.y;
		if (v[i].Max.x > Max.x) Max.x = v[i].Max.x;
		if (v[i].Max.y > Max.y) Max.y = v[i].Max.y;
	}
}
void getRange(vector<vector<spline3>> &v, vec2 &Min, vec2 &Max) {
	Min = vec2(INFINITY, INFINITY), Max = -Min;
	for (int i = 0, n = v.size(); i < n; i++) {
		vec2 _min, _max; getRange(v[i], _min, _max);
		if (_min.x < Min.x) Min.x = _min.x;
		if (_min.y < Min.y) Min.y = _min.y;
		if (_max.x > Max.x) Max.x = _max.x;
		if (_max.y > Max.y) Max.y = _max.y;
	}
}

void applyMatrix(vector<spline3> &v, mat2x3 M) {
	for (int i = 0, n = v.size(); i < n; i++) v[i].applyMatrix(M);
}
void applyMatrix(vector<vector<spline3>> &v, mat2x3 M) {
	for (int i = 0, n = v.size(); i < n; i++) applyMatrix(v[i], M);
}

// based on Green's formula, doesn't work for unclosed, self-intersecting, and some multiple-connected shapes
double calcArea(const vector<spline3> &v) {
	double S = 0;
	for (int i = 0, n = v.size(); i < n; i++) {
		vec2 a = v[i].A, b = v[i].B, c = v[i].C, d = v[i].D;
		S += a.x*(a.y*.5 + b.y*.4 + c.y*.25) + a.y*(b.x*.6 + c.x*.75 + d.x) + (b.y*(b.x*3. + c.x*4.) + c.y*(b.x*2. + c.x*3.)) / 6. + (b.y + c.y)*d.x;
	}
	return abs(S);
}
vec2 calcCOM(const vector<spline3> &v, double* A = 0) {
	double S = 0; vec2 P(0.0);
	for (int i = 0, n = v.size(); i < n; i++) {
		vec2 a = v[i].A, b = v[i].B, c = v[i].C, d = v[i].D;
		S += a.x*(a.y*.5 + b.y*.4 + c.y*.25) + a.y*(b.x*.6 + c.x*.75 + d.x) + (b.y*(b.x*3. + c.x*4.) + c.y*(b.x*2. + c.x*3.)) / 6. + (b.y + c.y)*d.x;
		// copied from an online integral calculator and it's pretty unneat
		P = P + vec2(((420 * c.y + 420 * b.y + 420 * a.y)*d.x*d.x + ((420 * c.x + 280 * b.x + 210 * a.x)*c.y + (560 * b.y + 630 * a.y)*c.x + (420 * b.x + 336 * a.x)*b.y + 504 * a.y*b.x + 420 * a.x*a.y)*d.x + (140 * c.x*c.x + (210 * b.x + 168 * a.x)*c.x + 84 * b.x*b.x + 140 * a.x*b.x + 60 * a.x*a.x)*c.y + (210 * b.y + 252 * a.y)*c.x*c.x + ((336 * b.x + 280 * a.x)*b.y + 420 * a.y*b.x + 360 * a.x*a.y)*c.x + (140 * b.x*b.x + 240 * a.x*b.x + 105 * a.x*a.x)*b.y + 180 * a.y*b.x*b.x + 315 * a.x*a.y*b.x + 140 * a.x*a.x*a.y) / 840, (((840 * c.y + 840 * b.y + 840 * a.y)*d.x + (420 * c.x + 280 * b.x + 210 * a.x)*c.y + (560 * b.y + 630 * a.y)*c.x + (420 * b.x + 336 * a.x)*b.y + 504 * a.y*b.x + 420 * a.x*a.y)*d.y + (420 * c.y*c.y + (840 * b.y + 840 * a.y)*c.y + 420 * b.y*b.y + 840 * a.y*b.y + 420 * a.y*a.y)*d.x + (280 * c.x + 210 * b.x + 168 * a.x)*c.y*c.y + ((630 * b.y + 672 * a.y)*c.x + (504 * b.x + 420 * a.x)*b.y + 560 * a.y*b.x + 480 * a.x*a.y)*c.y + (336 * b.y*b.y + 700 * a.y*b.y + 360 * a.y*a.y)*c.x + (280 * b.x + 240 * a.x)*b.y*b.y + (600 * a.y*b.x + 525 * a.x*a.y)*b.y + 315 * a.y*a.y*b.x + 280 * a.x*a.y*a.y) / 840);
		//P = P + vec2(-(((280 * c.x + 280 * b.x + 280 * a.x)*d.x + 140 * c.x*c.x + (280 * b.x + 280 * a.x)*c.x + 140 * b.x*b.x + 280 * a.x*b.x + 140 * a.x*a.x)*d.y + (-280 * c.y - 280 * b.y - 280 * a.y)*d.x*d.x + ((70 * a.x - 140 * c.x)*c.y + (-280 * b.y - 350 * a.y)*c.x + (-140 * b.x - 56 * a.x)*b.y - 224 * a.y*b.x - 140 * a.x*a.y)*d.x + ((70 * b.x + 112 * a.x)*c.x + 56 * b.x*b.x + 140 * a.x*b.x + 80 * a.x*a.x)*c.y + (-70 * b.y - 112 * a.y)*c.x*c.x + (-56 * b.x*b.y - 140 * a.y*b.x - 80 * a.x*a.y)*c.x + (40 * a.x*b.x + 35 * a.x*a.x)*b.y - 40 * a.y*b.x*b.x - 35 * a.x*a.y*b.x) / 840, -((280 * c.x + 280 * b.x + 280 * a.x)*d.y*d.y + ((-280 * c.y - 280 * b.y - 280 * a.y)*d.x + (140 * c.x + 280 * b.x + 350 * a.x)*c.y - 70 * a.y*c.x + (140 * b.x + 224 * a.x)*b.y + 56 * a.y*b.x + 140 * a.x*a.y)*d.y + (-140 * c.y*c.y + (-280 * b.y - 280 * a.y)*c.y - 140 * b.y*b.y - 280 * a.y*b.y - 140 * a.y*a.y)*d.x + (70 * b.x + 112 * a.x)*c.y*c.y + ((-70 * b.y - 112 * a.y)*c.x + (56 * b.x + 140 * a.x)*b.y + 80 * a.x*a.y)*c.y + (-56 * b.y*b.y - 140 * a.y*b.y - 80 * a.y*a.y)*c.x + 40 * a.x*b.y*b.y + (35 * a.x*a.y - 40 * a.y*b.x)*b.y - 35 * a.y*a.y*b.x) / 840);
	}
	if (A) *A = S;
	return P * (1. / S);
}

// line-based center of mass, works for unclosed and self-intersecting shapes
vec2 calcCenter(const vector<spline3> &v, double* L = 0) {
	const int dif = 20;
	double S = 0; vec2 P(0.0);
	for (int i = 0, n = v.size(); i < n; i++) {
		vec2 a = v[i].A, b = v[i].B, c = v[i].C, d = v[i].D;
		vec2 p = d;
		for (int ti = 1; ti <= dif; ti++) {
			double t = ti * (1.0 / dif);
			vec2 q = ((a*t + b)*t + c)*t + d;
			double dS = length(q - p);
			vec2 dP = (p + q)*(0.5*dS);
			S += dS, P = P + dP;
			p = q;
		}
	}
	if (L) *L = S;
	return P * (1.0 / S);
}
vec2 calcCenter(const vector<vector<spline3>> &v) {
	double S = 0; vec2 P(0.0);
	for (int i = 0, n = v.size(); i < n; i++) {
		double s;
		vec2 p = calcCenter(v[i], &s);
		S += s;
		P = P + p * s;
	}
	return P * (1.0 / S);
}



//===================== Conversion between SVG Path and Spline Lists =====================//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <streambuf>
#include <iomanip>

bool fromPath(const string &S, vector<spline3> &V) {
	if (S.empty()) return false;
#define isFloat(c) ((c >= '0' && c <= '9') || c == '-' || c == '.')
#define readFloat(r) { \
		while (d < n && (S[d] == ' ' || S[d] == ',')) d++; \
		if (d >= n || !isFloat(S[d])) return false; \
		unsigned sz; \
		(r) = stod(&S[d], &sz); \
		d += sz; \
	}
#define readPoint(v) { readFloat((v).x); readFloat((v).y); }
	char cmd = '\0'; vec2 P(0, 0), P0(0, 0), P1(NAN, NAN);
	for (int d = 0, n = S.size(); d < n;) {
		while (d < n && (S[d] == ' ' || S[d] == ',')) d++;

		if (string("MZLHVCSQTA").find(S[d] >= 'a' ? S[d] - 32 : S[d]) != -1) cmd = S[d], d++;
		else if (!isFloat(S[d])) return false;

		switch (cmd) {
		case 'M':; case 'm': {
			vec2 Q; readPoint(Q);
			if (cmd == 'm') P = P0 + Q;
			else P = Q;
			P0 = P, P1 = vec2(NAN, NAN);
			break;
		}
		case 'Z':; case 'z': {
			if ((P - P0).sqr() > 1e-12) V.push_back(fromSegment(P, P0));
			P1 = vec2(NAN, NAN);
			break;
		}
		case 'L':; case 'l': {
			vec2 Q; readPoint(Q);
			if (cmd == 'l') Q = P + Q;
			if ((Q - P).sqr() > 1e-12) V.push_back(fromSegment(P, Q));
			P1 = P, P = Q;
			break;
		}
		case 'H':; case 'h': {
			double c; readFloat(c);
			vec2 Q = P;
			if (cmd == 'H') Q.x = c;
			else Q.x += c;
			V.push_back(fromSegment(P, Q));
			P1 = P, P = Q;
			break;
		}
		case 'V':; case 'v': {
			double c; readFloat(c);
			vec2 Q = P;
			if (cmd == 'V') Q.y = c;
			else Q.y += c;
			V.push_back(fromSegment(P, Q));
			P1 = P, P = Q;
			break;
		}
		case 'C':; case 'c': {
			vec2 B, C, D;
			readPoint(B); readPoint(C); readPoint(D);
			if (cmd == 'c') B = B + P, C = C + P, D = D + P;
			V.push_back(fromBezier3(P, B, C, D));
			P1 = C, P = D;
			break;
		}
		case 'S':; case 's': {
			if (isnan(P1.x)) return 0;
			vec2 B = P * 2.0 - P1;
			vec2 C, D;
			readPoint(C); readPoint(D);
			if (cmd == 's') C = P + C, D = P + D;
			V.push_back(fromBezier3(P, B, C, D));
			P1 = C, P = D;
			break;
		}
		case 'Q':; case 'q': {
			vec2 B, C;
			readPoint(B); readPoint(C);
			if (cmd == 'q') B = B + P, C = C + P;
			V.push_back(fromBezier2(P, B, C));
			P1 = B, P = C;
			break;
		}
		case 'T':; case 't': {
			if (isnan(P1.x)) return 0;
			vec2 B = P * 2.0 - P1;
			vec2 C; readPoint(C);
			if (cmd == 't') C = P + C;
			V.push_back(fromBezier2(P, B, C));
			P1 = B, P = C;
			break;
		}
		case 'A':; case 'a': {		// possibly have bugs
			vec2 r; readPoint(r);
			double theta; readFloat(theta); theta = radians(theta);
			bool laf, sf; readFloat(laf); readFloat(sf);
			vec2 Q; readPoint(Q);
			if (cmd == 'a') Q = P + Q;

			mat2x3 T = mat2x3(1. / r.x, 0, 0, 1. / r.y) * RotationMatrix(-theta);
			vec2 p = T * P, q = T * Q, d = q - p;
			if (length(d) >= 2.0) {
				double s = (2.0 - 1e-12) / length(d);
				r = r * (1. / s), p = p * s, q = q * s, d = d * s;
			}
			double a = acos(0.5*length(d)), b;
			if (isnan(a)) {
				P = Q, P1 = P;
				break;
			}
			vec2 C = p + RotationMatrix(a) * normalize(d);
			T = mat2x3(r.x, 0, 0, r.y);
			C = T * C, p = T * p, q = T * q;
			if (!sf ^ laf) C = p + q - C;

			T = RotationMatrix(theta);
			a = atan2((p.y - C.y) / r.y, (p.x - C.x) / r.x), b = atan2((q.y - C.y) / r.y, (q.x - C.x) / r.x);
			if (sf && b < a) b += 2 * PI;
			if (!sf && a < b) a += 2 * PI;
			fromArc(T*C, r, a, b, theta, V);

			P = Q;
			break;
		}
		default: {
			return false;
		}
		}
	}
	return true;
#undef isFloat
#undef readFloat
#undef readPoint
}

string toPath(const vector<spline3> &v) {
	string s;
	if (v.empty()) return s;

	vec2 A, B, C, D, P = vec2(NAN, NAN);
	auto toString = [](vec2 p) ->string {
		stringstream ss;
		if (abs(p.x) < 1e-6) p.x = 0;
		if (abs(p.y) < 1e-6) p.y = 0;
		ss << setprecision(4) << p.x << "," << p.y;
		return string(ss.str());
	};
	for (int i = 0, l = v.size(); i < l; i++) {
		int n = toBezier(v[i], A, B, C, D);
		if (!((P - A).sqr() < 1e-4)) s += "M" + toString(A) + " ", P = A;
		s += n == 3 ? 'C' : n == 2 ? 'Q' : n == 1 ? 'L' : ' ';
		if (n >= 1) s += toString(B) + " ", P = B;
		if (n >= 2) s += toString(C) + " ", P = C;
		if (n >= 3) s += toString(D) + " ", P = D;
	}
	while (s[s.size() - 1] == ' ') s.erase(s.size() - 1, 1);
	return s;
}

string toMath(const vector<spline3> &v) {
	vec2 A, B, C, D;
	stringstream ss;
	ss << setprecision(3);
	for (int i = 0, n = v.size(); i < n; i++) {
		int N = toBezier(v[i], A, B, C, D);
		ss << "(";
		auto pos = noshowpos;
#define ssprint(v, s) if (abs(v) > 1e-4) ss << pos << (v) << s, pos = showpos;
		if (N == 0) ss << pos << A.x << "," << A.y;
		else if (N == 1) {
			ssprint(A.x, "(1-t)"); ssprint(B.x, "t");
			if (pos == noshowpos) ss << "0";
			ss << ","; pos = noshowpos;
			ssprint(A.y, "(1-t)"); ssprint(B.y, "t");
			if (pos == noshowpos) ss << "0";
		}
		else if (N == 2) {
			ssprint(A.x, "(1-t)^2"); ssprint(2.0*B.x, "t(1-t)"); ssprint(C.x, "t^2");
			if (pos == noshowpos) ss << "0";
			ss << ","; pos = noshowpos;
			ssprint(A.y, "(1-t)^2"); ssprint(2.0*B.y, "t(1-t)"); ssprint(C.y, "t^2");
			if (pos == noshowpos) ss << "0";
		}
		else {
			ssprint(A.x, "(1-t)^3"); ssprint(3.0*B.x, "t(1-t)^2"); ssprint(3.0*C.x, "t^2(1-t)"); ssprint(D.x, "t^3");
			if (pos == noshowpos) ss << "0";
			ss << ","; pos = noshowpos;
			ssprint(A.y, "(1-t)^3"); ssprint(3.0*B.y, "t(1-t)^2"); ssprint(3.0*C.y, "t^2(1-t)"); ssprint(D.y, "t^3");
			if (pos == noshowpos) ss << "0";
		}
		ss << ")\n";
	}
	return string(ss.str());
}



//================== Read SVG file - No error handling, valid svg please! ==================//

struct Element {
	// XML
	struct attribute { string name, value; };
	string tagName;
	vector<attribute> attributes;
	string value;
	vector<Element> childs;
	// Vector Graphics
	mat2x3 transform;
	vector<spline3> d;
};

#define isXMLName(c) (((c)>='a'&&(c)<='z')||((c)>='A'&&(c)<='Z')||((c)>='0'&&(c)<='9')||(c)=='-'||(c)=='_'||(c)=='.'||(c)==':')
#define skipSpace(s,d) while(s[d]>0&&s[d]<=' ')d++;
#define readName(r,s,d) while(isXMLName(s[d]))r.push_back(s[d++]);
#define readValue(r,s,d) while(s[d]&&s[d]!='\"'&&s[d]!='\''&&s[d]!='<'&&s[d]!='>')r.push_back(s[d++]);
bool readXML(const char *s, int &d, Element &r) {
	if (s[d++] != '<') return false;
	// read tagname
	r.tagName = "";
	skipSpace(s, d); readName(r.tagName, s, d); skipSpace(s, d);
	//cout << r.tagName << endl;
	// read attributes
	r.attributes.clear();
	while (s[d] != '/' && s[d] != '>') {
		r.attributes.push_back(Element::attribute());
		readName(r.attributes.back().name, s, d);
		skipSpace(s, d);
		if (s[d] == '=') {
			d++; skipSpace(s, d);  // xx="
			if (s[d] != '\"'&&s[d] != '\'') printf("[%d] No\": %d\n", __LINE__, d);
			d++; readValue(r.attributes.back().value, s, d);
			if (s[d] != '\"'&&s[d] != '\'') printf("[%d] No\": %d\n", __LINE__, d);
			d++;
			//cout << r.attributes.back().name << "      \t" << r.attributes.back().value << endl;
		}
		else printf("[%d] No=: %d\n", __LINE__, d);
		skipSpace(s, d);
	}
	if (s[d] == '/') {  // inline close element
		while (s[d++] != '>');
		return true;
	}
	else if (s[d] == '>') {
		d++;
		readValue(r.value, s, d);
		if (s[d] != '<') printf("[%d] No<: %d\n", __LINE__, d);
		while (s[d] == '<') {
			int d0 = d;
			d++; skipSpace(s, d);
			if (s[d] == '/') {  // close element
				while (s[d++] != '>');
				return true;
			}
			else d = d0;
			// read subtree
			r.childs.push_back(Element());
			//cout << endl;
			if (!readXML(s, d, r.childs.back())) return false;
			//cout << endl;
			readValue(r.value, s, d);
		}
		return true;
	}
	else return false;
}

void writeXML(const Element &E, FILE *fp) {
	if (E.tagName == "clipPath") return;  // remove it for some reason
	fprintf(fp, "<%s", &E.tagName[0]);
	for (int i = 0, n = E.attributes.size(); i < n; i++) {
		fprintf(fp, " %s=\"%s\"", &E.attributes[i].name[0], &E.attributes[i].value[0]);
	}
	fprintf(fp, ">");
	for (int i = 0, n = E.childs.size(); i < n; i++) {
		writeXML(E.childs[i], fp);
	}
	fprintf(fp, "</%s>", &E.tagName[0]);
}

#define readFloat(r,s,d) do{while(s[d]&&s[d]<'-')d++;unsigned sz;r=stod(&s[d],(size_t*)&sz);d+=sz;}while(0)
bool extractTransform(Element &E) {
	E.transform = mat2x3();
	bool ok = true;
	for (int i = 0, n = E.attributes.size(); i < n; i++) {
		if (E.attributes[i].name == "transform") {
			string t = E.attributes[i].value;
			int d = 0;
			while (t[d]) {
				while (t[d] < 'A' && t[d] != '\"') d++;
				string c; readName(c, t, d); skipSpace(t, d);
				if (c == "matrix") {
					double a, b, c, d_, x, y;
					readFloat(a, t, d); readFloat(c, t, d); readFloat(b, t, d); readFloat(d_, t, d); readFloat(x, t, d); readFloat(y, t, d);
					E.transform = E.transform * mat2x3(a, b, c, d_, x, y);
				}
				else if (c == "translate") {
					double x, y;
					readFloat(x, t, d); readFloat(y, t, d);
					E.transform = E.transform * mat2x3(x, y);
				}
				else {
					printf("[%d] %s\n", __LINE__, &c[0]);  // not supported
					ok = false; break;
				}
				break;
			}
		}
	}
	for (int i = 0, n = E.childs.size(); i < n; i++) {
		ok &= extractTransform(E.childs[i]);
	}
	return ok;
}

bool extractPath(Element &E) {
	E.d.clear();
	bool ok = true;
	for (int i = 0, n = E.attributes.size(); i < n; i++) {
		if (E.attributes[i].name == "d") {
			ok &= fromPath(E.attributes[i].value, E.d);
		}
	}
	for (int i = 0, n = E.childs.size(); i < n; i++) {
		ok &= extractPath(E.childs[i]);
	}
	return ok;
}

void flattenTransform(Element &E, mat2x3 T) {
	T = T * E.transform; E.transform = mat2x3();
	applyMatrix(E.d, T);
	for (int i = 0, n = E.childs.size(); i < n; i++) {
		flattenTransform(E.childs[i], T);
	}
}

void refreshTransform(Element &E) {
	stringstream ss;
	ss << "matrix(" << E.transform.a << "," << E.transform.c << "," << E.transform.b << "," << E.transform.d << "," << E.transform.x << "," << E.transform.y << ")";
	string s = ss.str();
	if (s == "matrix(1,0,0,1,0,0)") s = "";
	bool find = false;
	for (int i = 0, n = E.attributes.size(); i < n; i++) {
		if (E.attributes[i].name == "transform") {
			E.attributes[i].value = s;
			find = true; break;
		}
	}
	if (!find && !s.empty()) {
		E.attributes.push_back(Element::attribute{ "transform", s });
	}
	for (int i = 0, n = E.childs.size(); i < n; i++) {
		refreshTransform(E.childs[i]);
	}
}

void refreshPath(Element &E) {
	if (E.tagName == "path") {
		string s = toPath(E.d);
		for (int i = 0, n = E.attributes.size(); i < n; i++) {
			if (E.attributes[i].name == "d") {
				E.attributes[i].value = s; break;
			}
		}
	}
	for (int i = 0, n = E.childs.size(); i < n; i++) {
		refreshPath(E.childs[i]);
	}
}

void getPath(const Element &E, vector<vector<spline3>> &p) {
	if (!E.d.empty() && E.tagName != "clipPath") p.push_back(E.d);
	for (int i = 0, n = E.childs.size(); i < n; i++) getPath(E.childs[i], p);
}

// read a svg file - no "defs" please!
vector<vector<spline3>> readSVG(string filename) {
	vector<vector<spline3>> r;
	ifstream _if(filename);
	if (_if.fail()) return r;
	string s((istreambuf_iterator<char>(_if)), istreambuf_iterator<char>());
	_if.close();
	const int n = s.size();
	int d = s.find("<svg");
	if (d == -1) return r;
	Element SVG;
	if (!readXML(&s[0], d, SVG)) printf("Error! [%d]\n", __LINE__);
	extractTransform(SVG);
	extractPath(SVG);
	flattenTransform(SVG, mat2x3());
	getPath(SVG, r); return r;
	//refreshTransform(SVG);
	//refreshPath(SVG);
	//FILE *fp = fopen("D:\\ex+.svg", "w"); writeXML(SVG, fp); fclose(fp);
}

int main() {
#if 0
	string s;
	ifstream fin("D:\\temp.dat");
	getline(fin, s);
	fin.close();
	vector<spline3> path;
	if (!fromPath(s, path)) { cout << "Error!\n"; return 0; }
	vec2 Min, Max; getRange(path, Min, Max);
	double S = 5.0 / sqrt((Max.x - Min.x)*(Max.y - Min.y));
	vec2 C = calcCenter(path);
	applyMatrix(path, mat2x3(S, 0, 0, -S) * mat2x3(-C.x, -C.y));
	cout << toMath(path) << endl;
#else
	vector<vector<spline3>> paths = readSVG("D:\\ex.svg");
	if (paths.empty()) { cout << "Error!\n"; return 0; }
	vec2 Min, Max; getRange(paths, Min, Max);
	double S = 5.0 / sqrt((Max.x - Min.x)*(Max.y - Min.y));
	vec2 C = calcCenter(paths);
	applyMatrix(paths, mat2x3(S, 0, 0, -S) * mat2x3(-C.x, -C.y));
	for (int i = 0, n = paths.size(); i < n; i++) {
		cout << toMath(paths[i]) << endl;
	}
#endif
	return 0;
}

