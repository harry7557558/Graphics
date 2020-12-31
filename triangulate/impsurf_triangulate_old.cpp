// triangulation of implicit surface based on an old (and naive) source written more than one year ago
// change Win32 GUI display to STL output


// use headers that I currently feel comfortable
#include <stdio.h>
#include "numerical/geometry.h"
#include "numerical/random.h"
#include "ui/stl_encoder.h"

#include <iostream>
#define Dout(s) std::cout << s
#define dout(s) std::cout << s




// copied from a header file

#pragma region SDF_3D.h

double SD_Torus(double R, double r, vec3 P) {		// xOy
	double D = length(P.xy()) - R;
	return sqrt(D * D + P.z * P.z) - r;
}
double SD_Segment(vec3 V1, vec3 V2, vec3 P) {
	double t = dot(V2 - V1, P - V1);
	if (t < 0) return length(P - V1);
	if (t > (V2 - V1).sqr()) return length(P - V2);
	return length(cross(normalize(V2 - V1), P - V1));
}
double SD_Cylinder_z(double r, double min_z, double max_z, vec3 P) {	// perpendicular to xOy, min_z<max_z
	double d = length(P.xy()) - r;
	if (P.z < max_z && P.z > min_z) return max(d, max(min_z - P.z, P.z - max_z));
	if (d < 0) return P.z > max_z ? P.z - max_z : min_z - P.z;
	return P.z > max_z ? sqrt(d*d + (P.z - max_z)*(P.z - max_z)) : sqrt(d*d + (P.z - min_z)*(P.z - min_z));
}
double SD_OpExtrusion(double(*sd)(vec2), double h, vec3 P) {	// exact, extrude toward z-axis
	double d = sd(P.xy());
	if (P.z >= 0 && P.z <= h) return d > 0 ? d : max(d, max(-P.z, P.z - h));
	if (P.z > h) return d > 0 ? sqrt(d*d + (P.z - h)*(P.z - h)) : P.z - h;
	if (P.z < 0) return d > 0 ? sqrt(d*d + P.z*P.z) : -P.z;
}
double sd_Polygon(const vec2 *v, int N, vec2 p) {
	double d = dot(p - v[0], p - v[0]);
	bool sgn = 0;
	vec2 e, w, b; double c;
	for (int i = 0, j = N - 1; i < N; j = i++) {
		e = v[j] - v[i];
		w = p - v[i];
		c = dot(w, e) / e.sqr();
		b = c < 0 ? w : c > 1 ? w - e : w - e * c;
		d = min(d, b.sqr());
		if (e.y < 0) e.y = -e.y, w.y = -w.y;
		if (w.y > 0 && w.y < e.y && (w.y*e.x / e.y > w.x)) sgn ^= 1;
	}
	return sgn ? -sqrt(d) : sqrt(d);
}
inline double sd_Polygon(std::initializer_list<vec2> v, vec2 p) {
	return sd_Polygon(v.begin(), v.size(), p);
}
inline double sd_OpSubtract(double sd1, double sd2) { return max(sd1, -sd2); }
inline double sd_OpSmoothedSubtract(double a, double b, double k) {
	double h = 0.5 - 0.5*(b + a) / k;
	if (h < 0) return a; if (h > 1) return -b;
	return (1 - h)*a - h * b + k * h * (1 - h);
}
inline double sd_OpSmoothedUnion(double a, double b, double k) {
	double h = 0.5 + 0.5*(b - a) / k;
	if (h < 0) return b; if (h > 1) return a;
	return (1 - h)*b + h * a - k * h * (1 - h);
}

#pragma endregion  // SDF_3D.h




// test surfaces

double Imp0(vec3 p) {	// sphere
	return 1 - p.sqr();
	return p.x*p.x + p.y*p.y / 2.6 + p.z*p.z - 1;
	return exp(4 * (p.sqr() - 2)) + exp(4 * (0.5 - p.x*p.x - p.y*p.y - p.z*p.z / 4)) - 1;
}
double Imp1(vec3 p) {	// "rounded" cube
	return p.x*p.x*p.x*p.x + p.y*p.y*p.y*p.y + p.z*p.z*p.z*p.z - 1;
	return pow(p.x, 8) + pow(p.y, 8) + pow(p.z, 8) - 1;
	return p.x*p.x*p.x*p.x + p.y*p.y*p.y*p.y + p.z*p.z*p.z*p.z - p.x*p.x - p.y*p.y;
	return p.x*p.x*p.x*p.x + p.y*p.y*p.y*p.y + p.z*p.z*p.z*p.z - p.x*p.x - p.y*p.y + 0.1;
}
double Imp2(vec3 p) {	// torus
	return pow(p.sqr() + 0.8, 2) - 4 * p.xy().sqr();
	return pow(p.sqr() + 3.5, 2) - 16 * p.xy().sqr();
	return p.sqr()*p.sqr() - p.xy().sqr();
	return exp(10 * (4 * p.xy().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.xz().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.yz().sqr() - pow(p.sqr() + 0.96, 2))) - 1;
	return pow(pow(p.x, 6) + pow(p.y, 6) + pow(p.z, 6) + 3.5, 2) - 16 * (pow(p.x, 6) + pow(p.y, 6));
}
double Imp3(vec3 p) {	// peanut
	return exp(1 - (p.x - 1)*(p.x - 1) - p.y*p.y - p.z*p.z) + exp(1 - (p.x + 1)*(p.x + 1) - p.y*p.y - p.z*p.z) - 1;
	return exp(4 * (1.2 - (p.x - 1)*(p.x - 1) - (p.y - 1)*(p.y - 1) - (p.z - 1)*(p.z - 1))) + exp(4 * (1.2 - (p.x + 1)*(p.x + 1) - (p.y - 1)*(p.y - 1) - (p.z - 1)*(p.z - 1))) \
		+ exp(4 * (1.2 - (p.x - 1)*(p.x - 1) - (p.y + 1)*(p.y + 1) - (p.z - 1)*(p.z - 1))) + exp(4 * (1.2 - (p.x + 1)*(p.x + 1) - (p.y + 1)*(p.y + 1) - (p.z - 1)*(p.z - 1))) \
		+ exp(4 * (1.2 - (p.x - 1)*(p.x - 1) - (p.y - 1)*(p.y - 1) - (p.z + 1)*(p.z + 1))) + exp(4 * (1.2 - (p.x + 1)*(p.x + 1) - (p.y - 1)*(p.y - 1) - (p.z + 1)*(p.z + 1))) \
		+ exp(4 * (1.2 - (p.x - 1)*(p.x - 1) - (p.y + 1)*(p.y + 1) - (p.z + 1)*(p.z + 1))) + exp(4 * (1.2 - (p.x + 1)*(p.x + 1) - (p.y + 1)*(p.y + 1) - (p.z + 1)*(p.z + 1))) - 1;
	return exp(2 * (0.5 - p.x*p.x - p.y*p.y - 1.5*(p.z - 0.8)*(p.z - 0.8))) + exp(2 * (1.5 - p.x*p.x - p.y*p.y - 1.5*(p.z + 0.8)*(p.z + 0.8))) + 0.1*sin(20 * p.x)*sin(20 * p.x)*sin(20 * p.z) - 1;
}
double Imp4(vec3 p) {	// heart
	return pow(p.x*p.x + 2.25*p.y*p.y + p.z*p.z - 1, 3) - (p.x*p.x + 0.1125*p.y*p.y)*p.z*p.z*p.z;
}
double Imp5(vec3 p) {	// open surface that intersects itself
	return p.x*p.x + p.y*p.y + p.z*p.z*p.z - p.z*p.z;
}
double Imp6(vec3 p) {	// jax
	return ((p.x - 1)*(p.x - 1) + p.y*p.y + p.z*p.z) * ((p.x + 1)*(p.x + 1) + p.y*p.y + p.z*p.z) * (p.x*p.x + (p.y - 1)*(p.y - 1) + p.z*p.z) * (p.x*p.x + (p.y + 1)*(p.y + 1) + p.z*p.z) - 1.1;
	return ((p.x - 1)*(p.x - 1) + p.y*p.y + p.z*p.z) * ((p.x + 1)*(p.x + 1) + p.y*p.y + p.z*p.z) * (p.x*p.x + (p.y - 1)*(p.y - 1) + p.z*p.z) * (p.x*p.x + (p.y + 1)*(p.y + 1) + p.z*p.z) \
		* (p.x*p.x + p.y*p.y + (p.z - 1)*(p.z - 1)) * (p.x*p.x + p.y*p.y + (p.z + 1)*(p.z + 1)) - 1.5;
}
double Imp7(vec3 p) {	// test thin lines
	return sd_OpSmoothedUnion(min(length(p - vec3(-1, 0, 0)), length(p - vec3(1, 0, 0))) - 0.5, SD_Segment(vec3(-1, 0, 0), vec3(1, 0, 0), p) - 0.1, 0.1);
	return sd_OpSmoothedUnion(sd_OpSmoothedUnion(\
		sd_OpSmoothedUnion(length(vec3(abs(p.x), p.y, p.z) - vec3(1, 0, 0)) - 0.3, SD_Segment(vec3(-1, 0, 0), vec3(1, 0, 0), p), 0.1), \
		sd_OpSmoothedUnion(length(vec3(p.x, abs(p.y), p.z) - vec3(0, 1, 0)) - 0.3, SD_Segment(vec3(0, -1, 0), vec3(0, 1, 0), p), 0.1), 0.1), \
		sd_OpSmoothedUnion(length(vec3(p.x, p.y, abs(p.z)) - vec3(0, 0, 1)) - 0.3, SD_Segment(vec3(0, 0, -1), vec3(0, 0, 1), p), 0.1), 0.1) - 0.075;
}
double Imp8(vec3 p) {	// test genus
	return 2 * p.y*(p.y*p.y - 3 * p.x*p.x)*(1 - p.z*p.z) + (p.x*p.x + p.y*p.y)*(p.x*p.x + p.y*p.y) - (9 * p.z*p.z - 1)*(1 - p.z*p.z);
	return SD_Torus(1, 0, p - vec3(-1, -sqrt(3) / 3, 0)) * SD_Torus(1, 0, p - vec3(1, -sqrt(3) / 3, 0)) * SD_Torus(1, 0, p - vec3(0, sqrt(3) / 1.5, 0)) - 0.3;
}
double Imp9(vec3 p) {  // a star created for fun
	vec3 u = p * p;
	double d = u.x + 2.0*u.y + u.z - 1.0;
	return 4.0*d*d - p.z*(5.*u.x*u.x - 10.*u.x*u.z + u.z*u.z) - 1.0;
}

double ImpX(vec3 p) {		// cup constructed with csg, ultra complex piecewise implicit surface
	vec3 P = p + vec3(0, 0, 0.5);
	double cyl = SD_Cylinder_z(0.75, 0, 2, P);
	double sd = abs(cyl - 0.15) - 0.08;
	sd = sd_OpSmoothedSubtract(sd, 1.8 - P.z, 0.08);
	double handle = SD_OpExtrusion([](vec2 p) {
		return abs(sd_Polygon({ vec2(0,0.9), vec2(0.4,0.9), vec2(0.3,0.35), vec2(0,0.1) }, p) - 0.2) - 0.05;
	}, 0.3, vec3(P.x - 1.03, P.z - 0.38, P.y + 0.15)) - 0.1;
	handle = sd_OpSubtract(handle, cyl - 0.12);
	sd = sd_OpSmoothedUnion(sd, handle, 0.05);
	//return sd * sd;
	return sd;
}

#define Imp ImpX





#include <vector>
#include <stack>
struct f_triangle {
	vec3 A, B, C;
	vec3 col;
	f_triangle() {}
	f_triangle(vec3 A, vec3 B, vec3 C, vec3 col = vec3(0.0)) : A(A), B(B), C(C), col(col) {}
};
std::vector<f_triangle*> T;
std::vector<f_triangle*> Td;

struct gapNode {
	vec3 p;
	gapNode *m = 0, *n = 0;
	gapNode *nb = 0;
};



#define EPSILON 1e-6
#define UPSILON 1e-4

// tetrahedron method numerically calculate gradient
vec3 calcGradient(double(*f)(vec3), vec3 p) {
	double k_111 = f(vec3(p.x + EPSILON, p.y + EPSILON, p.z + EPSILON));
	double k_100 = f(vec3(p.x + EPSILON, p.y - EPSILON, p.z - EPSILON));
	double k_010 = f(vec3(p.x - EPSILON, p.y + EPSILON, p.z - EPSILON));
	double k_001 = f(vec3(p.x - EPSILON, p.y - EPSILON, p.z + EPSILON));
	vec3 n(k_111 + k_100 - k_010 - k_001, k_111 - k_100 + k_010 - k_001, k_111 - k_100 - k_010 + k_001);
	n /= 4 * EPSILON; return n;
}

// numerically calculate radius of curvature
double calcCurvatureR(double(*f)(vec3), vec3 p) {
	vec3 n = calcGradient(f, p);
#define RotationMatrix_xz(rx,rz) mat3(\
		cos(rz), -sin(rz), 0,\
		cos(rx)*sin(rz), cos(rx)*cos(rz), -sin(rx),\
		sin(rx)*sin(rz), sin(rx)*cos(rz), cos(rx)).transpose()
	mat3 R = RotationMatrix_xz(atan2(sqrt(n.x*n.x + n.y*n.y), n.z), atan2(n.x, -n.y));
	vec3 x = R * vec3(UPSILON, 0, 0), y = R * vec3(0, UPSILON, 0);
	double ax = ndot(calcGradient(f, p + x), calcGradient(f, p - x));
	double ay = ndot(calcGradient(f, p + y), calcGradient(f, p - y));
	ax = sqrt(1 - ax * ax), ay = sqrt(1 - ay * ay);
	double r = 2 * UPSILON / max(ax, ay);
	return clamp(r, 0.1, 1.0);
}

// Newton's iteration method finding point on surface
bool land(double(*f)(vec3), vec3 &p) {
	vec3 n; double d;
	int i = 0; do {
		n = calcGradient(f, p);
		d = f(p);
		p -= (d / n.sqr()) * n;
		if (++i > 50 || isnan(d)) return false;
	} while (abs(d) > EPSILON);
	return true;
}

// non-recursive growing phase, no depth limit, where a stack is needed
typedef struct {
	vec3 v1, v2, v3;
	int stage = 0;
} spread_step;
bool spread(double(*f)(vec3), vec3 v1, vec3 v2, double r, gapNode* &g) {
	const vec3 col = vec3(0.8, 0.8, 0.8);
	std::stack<spread_step> S;
	vec3 v3;
	S.push(spread_step());
	S.top().v1 = v1, S.top().v2 = v2;
	do {
		v1 = S.top().v1, v2 = S.top().v2, v3 = S.top().v3;
		if (S.top().stage == 0) {
			v3 = r * (axis_angle(calcGradient(f, v1), PI / 3) * normalize(v2 - v1)) + v1;
			land(f, v3);
			bool s = true;
			for (int i = 0; i < T.size(); i++) {	// O(n^2)
				if (length(v3 - T[i]->A) < 0.578*r || length(v3 - T[i]->B) < 0.578*r || length(v3 - T[i]->C) < 0.578*r) {
					s = false; S.pop(); break;
				}
			}
			if (s) {
				T.push_back(new f_triangle(v1, v2, v3, col));
				S.top().v3 = v3;
				S.top().stage = 1;
				S.push(spread_step());
				S.top().v1 = v1, S.top().v2 = v3;
			}
		}
		else if (S.top().stage == 1) {
			g->n = new gapNode;
			g->n->p = v3, g->n->m = g, g->n->n = 0;
			g = g->n;
			S.top().stage = 2;
			S.push(spread_step());
			S.top().v1 = v3, S.top().v2 = v2;
		}
		else S.pop();
	} while (!S.empty());
	return true;
}

bool spread_cd(double(*f)(vec3), vec3 v1, vec3 v2, double r, gapNode* &g) {
	const vec3 col = vec3(0.8, 0.8, 0.8);
	std::stack<spread_step> S;
	vec3 v3;
	S.push(spread_step());
	S.top().v1 = v1, S.top().v2 = v2;
	do {
		v1 = S.top().v1, v2 = S.top().v2, v3 = S.top().v3;
		if (S.top().stage == 0) {
			double R = r * max(max(calcCurvatureR(f, v1), calcCurvatureR(f, v2)), calcCurvatureR(f, 0.5*(v1 + v2)));
			v3 = R * (axis_angle(calcGradient(f, v1), PI / 3) * normalize(v2 - v1)) + v1;
			land(f, v3);
			bool s = true;
			R = 0.6 * r * max(max(calcCurvatureR(f, v1), calcCurvatureR(f, v2)), calcCurvatureR(f, v3));
			for (int i = 0; i < T.size(); i++) {	// O(n^2)
				if (length(v3 - T[i]->A) < R || length(v3 - T[i]->B) < R || length(v3 - T[i]->C) < R) {
					s = false; S.pop(); break;
				}
			}
			if (s) {
				T.push_back(new f_triangle(v1, v2, v3, col));
				S.top().v3 = v3;
				S.top().stage = 1;
				S.push(spread_step());
				S.top().v1 = v1, S.top().v2 = v3;
			}
		}
		else if (S.top().stage == 1) {
			g->n = new gapNode;
			g->n->p = v3, g->n->m = g, g->n->n = 0;
			g = g->n;
			S.top().stage = 2;
			S.push(spread_step());
			S.top().v1 = v3, S.top().v2 = v2;
		}
		else S.pop();
	} while (!S.empty());
	return true;
}


// debug
auto checkChain = [](gapNode *g) {
	int N = T.size();
	gapNode *g0 = g;
	while (1) {
		g = g->n;
		if (g->m->n != g) {
			Dout("g->m->n != g : " << g);
		}
		if (g->n->m != g) {
			Dout("g->n->m != g : " << g);
		}
		if (g == g0) {
			Dout("OK," << (T.size() - N));
			break;
		}
		if (--N <= 0) {
			Dout("--N == 0");
		}
	}
};
auto drawNB = [](gapNode *g, vec3 col) {
	if (g != 0) {
		checkChain(g);
		gapNode *g0 = g;
		do {
			if (g->nb != 0) {
				Td.push_back(new f_triangle(g->p, g->nb->p + vec3(0.003, 0, 0), g->nb->p - vec3(0.003, 0, 0), col));
				Td.push_back(new f_triangle(g->p, g->nb->p + vec3(0, 0.003, 0), g->nb->p - vec3(0, 0.003, 0), col));
				Td.push_back(new f_triangle(g->p, g->nb->p + vec3(0, 0, 0.003), g->nb->p - vec3(0, 0, 0.003), col));
			}
		} while ((g = g->n) != g0);
		g = 0;
	}
};

// fill a gap, g can be any node on the gap
auto calcNeighbour = [](double(*f)(vec3), gapNode *g, double r) {		// O(n^2)
	g->nb = 0;
	if (g->n->n == g || g->n->n->n == g) return;
	gapNode *m = g->m, *n = g->n;
	double d, md = INFINITY;
	mat3 R = axis_angle(calcGradient(f, g->p), PI / 2);
	auto crossTrig = [&](const gapNode *g, const gapNode *h)->bool {	// test if a f_triangle is between the two nodes (heuristics, not always work)
		if (length(g->p - h->p) > 2.5 * r) return true;
		for (int i = 0; i < T.size(); i++) {
			f_triangle *t = T[i];
			if ((g->p - t->A).sqr() < 16 * r*r) {
				if (g->p == t->A || g->p == t->B || g->p == t->C) {
					if (h->p == t->A || h->p == t->B || h->p == t->C) return true;

					if (g->p == t->B) std::swap(t->A, t->B); else if (g->p == t->C) std::swap(t->A, t->C);
					vec3 b = normalize(t->B - t->A), c = normalize(t->C - t->A), p = normalize(h->p - g->p); double m = dot(b, c);
					if (dot(p, b) > m && dot(p, c) > m) return true;
				}
				if (h->p == t->A || h->p == t->B || h->p == t->C) {
					if (h->p == t->B) std::swap(t->A, t->B); else if (h->p == t->C) std::swap(t->A, t->C);
					vec3 b = normalize(t->B - t->A), c = normalize(t->C - t->A), p = normalize(g->p - h->p); double m = dot(b, c);
					if (dot(p, b) > m && dot(p, c) > m) return true;
				}
			}
		}
		return false;
	};
	gapNode *h = n->n;
	do {
		d = length(h->p - g->p);
		if (d < md && !crossTrig(g, h)) md = d, g->nb = h;
	} while ((h = h->n) != m);

};
void fillGap(double(*f)(vec3), gapNode* &g, double r, int N) {
	const vec3 col = vec3(1, 0.6, 0);
	gapNode *g0 = g;
	if (N == 0) do {	// O(n^3)
		calcNeighbour(f, g, r);
	} while ((g = g->n) != g0);


	// fill angles less than 65° within the gap
	auto cutEars = [](double(*f)(vec3), gapNode* &g, double r) ->bool {
		const vec3 col = vec3(0.9, 0.8, 0.7);
		bool res = false;
		while (ndot(g->m->p - g->p, g->n->p - g->p) > 0.4 && det(g->m->p - g->p, g->n->p - g->p, calcGradient(f, g->p)) < 0) {
			T.push_back(new f_triangle(g->m->p, g->p, g->n->p, col));
			gapNode *h = g->m; g = g->n; delete g->m; g->m = h, h->n = g;
			calcNeighbour(f, h, r); calcNeighbour(f, g, r);
			res = true;
		}
		gapNode *g0 = g;
		do {
			if (ndot((g->m->p - g->p), (g->n->p - g->p)) > 0.4 && det(g->m->p - g->p, g->n->p - g->p, calcGradient(f, g->p)) < 0) {
				T.push_back(new f_triangle(g->m->p, g->p, g->n->p, col));
				gapNode *h = g->m; g = g->n; delete g->m; g->m = h, h->n = g;
				calcNeighbour(f, h, r); calcNeighbour(f, g, r);
				res = true;
			}
		} while ((g = g->n) != g0);
		return res;
	};

	auto fillX = [](double(*f)(vec3), gapNode* &g, double r)->bool {
		const vec3 col = vec3(0.8, 0.9, 0.7);
		gapNode *a, *b, *c, *d, *g0 = g;
		do {
			a = g, b = g->n, c = b->n, d = c->n;
			//if ((a->nb == c && b->nb == d && ((c->nb == a && d->nb == b) || (a->nb == d && d->nb == a))) || (a->nb == d && d->nb == a)) {
			if ((b->nb == d && c->nb == a) || (a->nb == d && d->nb == a)) {
				if (length(a->p - c->p) < length(b->p - d->p)) T.push_back(new f_triangle(a->p, b->p, c->p, col)), T.push_back(new f_triangle(a->p, d->p, c->p, col));
				else T.push_back(new f_triangle(a->p, b->p, d->p, col)), T.push_back(new f_triangle(c->p, b->p, d->p, col));
				delete b, c; b = c = 0;
				a->n = d, d->m = a;
				calcNeighbour(f, a, r), calcNeighbour(f, d, r);
				return true;
			}
		} while ((g = g->n) != g0);
		return false;
	};

	auto fillT = [](double(*f)(vec3), gapNode* &g, double r)->bool {
		const vec3 col = vec3(0.7, 0.9, 0.8);
		gapNode *a, *b, *c, *g0 = g;
		do {
			a = g, b = g->n, c = b->n;
			if (a->nb == c && c->nb == a) {
				T.push_back(new f_triangle(a->p, b->p, c->p, col));
				delete b; b = 0;
				a->n = c, c->m = a;
				calcNeighbour(f, a, r), calcNeighbour(f, c, r);
				return true;
			}
		} while ((g = g->n) != g0);
		return false;
	};

	auto fillW = [](double(*f)(vec3), gapNode* &g, double r)->bool {
		const vec3 col = vec3(0.7, 0.9, 0.8);
		gapNode *a, *b, *c, *d, *e, *g0 = g;
		do {
			a = g, b = g->n, c = b->n, d = c->n, e = d->n;
			if (a == e) return false;
			if (a->nb == e && e->nb == a || a == e->n) {
				gapNode *_m = a->m, *_n = e->n; a->m = e, e->n = a;
				gapNode *t = a, *mt = 0; double s, ms = 1;
				for (int i = 0; i < 5; i++) {
					s = ndot(t->n->p - t->p, t->m->p - t->p);
					if (s < ms) ms = s, mt = t;
					t = t->n;
				}
				T.push_back(new f_triangle(mt->p, mt->m->p, mt->m->m->p, col));
				T.push_back(new f_triangle(mt->p, mt->n->p, mt->n->n->p, col));
				T.push_back(new f_triangle(mt->p, mt->m->m->p, mt->n->n->p, col));
				delete b, c, d; b = c = d = 0;
				a->m = _m, e->n = _n; a->n = e, e->m = a;
				calcNeighbour(f, a, r), calcNeighbour(f, e, r);
				return true;
			}
		} while ((g = g->n) != g0);
		return false;
	};

	auto fillH = [](double(*F)(vec3), gapNode* &g, double r)->bool {
		const vec3 col = vec3(0.6, 0.8, 0.8);
		gapNode *a, *b, *c, *d, *e, *f, *g0 = g;
		do {
			a = g, b = g->n, c = b->n, d = c->n, e = d->n, f = e->n;
			if (a == e || a == f) return false;
			if ((a->nb == f && f->nb == a) || a == f->n) {
				gapNode *_m = a->m, *_n = f->n; a->m = f, f->n = a;
				gapNode *t = a, *mt = 0; double s, ms = 1;
				for (int i = 0; i < 5; i++) {
					s = ndot(t->n->p - t->p, t->m->p - t->p);
					if (s < ms) ms = s, mt = t;
					t = t->n;
				}
				t = mt;
				if (ndot((t->m->m->m->p - t->m->m->p), (t->n->n->n->p - t->n->n->p)) < -0.8) {
					gapNode *a = t, *b = t->m, *c = t->m->m, *d = t->m->m->m;
					if (length(a->p - c->p) < length(b->p - d->p)) T.push_back(new f_triangle(a->p, b->p, c->p, col)), T.push_back(new f_triangle(a->p, d->p, c->p, col));
					else T.push_back(new f_triangle(a->p, d->p, b->p, col)), T.push_back(new f_triangle(c->p, d->p, b->p, col));
					b = t->n, c = t->n->n;
					if (length(a->p - c->p) < length(b->p - d->p)) T.push_back(new f_triangle(a->p, b->p, c->p, col)), T.push_back(new f_triangle(a->p, d->p, c->p, col));
					else T.push_back(new f_triangle(a->p, d->p, b->p, col)), T.push_back(new f_triangle(c->p, d->p, b->p, col));
				}
				else {
					T.push_back(new f_triangle(t->p, t->m->p, t->m->m->p, col));
					T.push_back(new f_triangle(t->p, t->n->p, t->n->n->p, col));
					T.push_back(new f_triangle(t->p, t->m->m->p, t->n->n->p, col));
					T.push_back(new f_triangle(t->m->m->m->p, t->m->m->p, t->n->n->p, col));
				}
				delete b, c, d, e; b = c = d = e = 0;
				a->m = _m, f->n = _n; a->n = f, f->m = a;
				calcNeighbour(F, a, r), calcNeighbour(F, f, r);
				return true;
			}
		} while ((g = g->n) != g0);
		return false;
	};

	auto fillTg = [](double(*f)(vec3), gapNode* &g, double r)->bool {
		const vec3 col = vec3(0.7, 0.9, 0.8);
		gapNode *a, *b, *c, *g0 = g;
		do {
			a = g, b = g->n, c = b->n;
			if (a->nb == c || c->nb == a) {
				T.push_back(new f_triangle(a->p, b->p, c->p, col));
				delete b; b = 0;
				a->n = c, c->m = a;
				calcNeighbour(f, a, r), calcNeighbour(f, c, r);
				return true;
			}
		} while ((g = g->n) != g0);
		return false;
	};

	bool s = false, t = false;
	do {
		s = true, t = false;
		do {
			s = (cutEars(f, g, r) || fillX(f, g, r) || fillT(f, g, r) || fillW(f, g, r) || fillH(f, g, r));
			if (s) t = true;
		} while (s);
		if (fillTg(f, g, r)) t = true;
	} while (t);


	if (g->n->n == g) {
		Dout("Succeed");
		delete g->n; //delete g;
		g = 0; return;
	}

	g0 = g;
	do {
		if (g->nb != 0 && g->nb->nb != 0 && g->nb->nb == g && g->n->nb != 0 && g->n->nb->nb != 0 && g->n->nb->nb->m == g && g->nb->m == g->n->nb) {

			// construct bridge, genus n construct 2n bridges
			gapNode *h = g->n;
			T.push_back(new f_triangle(g->p, g->nb->p, h->nb->p, col));
			T.push_back(new f_triangle(g->p, h->p, h->nb->p, col));
			g->nb->m = g, g->n = g->nb;
			h->nb->n = h, h->m = h->nb;
			calcNeighbour(f, g, r); calcNeighbour(f, g->n, r);
			calcNeighbour(f, h, r); calcNeighbour(f, h->m, r);
			Dout("Bridge Constructed");

			fillGap(f, g, r, N + 1);
			if (g != 0) dout("Fail\n");

			return;
		}
	} while ((g = g->n) != g0);

	Dout("Bridge Construction Failed");
}

// subdivide f_triangles to fit the surface
void subdivide(double(*f)(vec3)) {
	int N = T.size();
	for (int i = 0; i < N; i++) {
		f_triangle t = *T[i];
		vec3 AB = 0.5*(t.A + t.B), BC = 0.5*(t.B + t.C), CA = 0.5*(t.C + t.A);
		land(f, AB), land(f, BC), land(f, CA);
		T.push_back(new f_triangle(AB, t.B, BC, t.col));
		T.push_back(new f_triangle(BC, t.C, CA, t.col));
		T.push_back(new f_triangle(CA, t.A, AB, t.col));
		T.push_back(new f_triangle(AB, BC, CA, t.col));
	}
	T.erase(T.begin(), T.begin() + N);
}


bool triangulate(double(*f)(vec3), double r) {
	if (!T.empty()) return false;

	// settle first point
	vec3 p; int tp = 0;
	uint32_t seed = 0;
	do {
		p = rand3(seed) * (lcg_next(seed)*(2. / 4294967296.));
		if (++tp > 1000) return false;
	} while (!land(f, p));
	// once the first point is settled, failing in landing points is not easy to happen

	// settle first f_triangle
	vec3 n = normalize(calcGradient(f, p));
	vec3 i = ncross(n, vec3(0, 0, 1)), j = axis_angle(n, PI / 3) * i;
	vec3 a = p + r * i, b = p + r * j;
	land(f, a); land(f, b);
	T.push_back(new f_triangle(p, a, b, vec3(0.8, 0, 0)));

	// growing phase, gap defined by a list of vertexes
	gapNode *g0 = new gapNode; g0->p = p, g0->m = 0, g0->n = new gapNode;
	gapNode *g = g0->n; g->m = g0, g->p = b, g->n = 0;
	if (!spread(f, b, a, r, g)) return false;
	g->n = new gapNode; g->n->m = g, g->n->p = a, g->n->n = g0, g0->m = g->n;
	g0 = g;

	fillGap(f, g, r, 0);

	//subdivide(f); subdivide(f); subdivide(f);


	for (int i = 0; i < Td.size(); i++) T.push_back(Td[i]);
	//for (int i = 0; i < T.size(); i++) T[i]->col = RGBf(0.87, 0.8, 0.7);

	return false;
}





int main(int argc, char* argv[]) {

	triangulate(Imp, 0.1);
	std::vector<stl_triangle> Trigs;
	for (int i = 0, l = T.size(); i < l; i++) {
		Trigs.push_back(stl_triangle(T[i]->A, T[i]->B, T[i]->C, T[i]->col));
	}
	writeSTL(argv[1], &Trigs[0], Trigs.size());  // normals are incorrect

	return 0;
}
