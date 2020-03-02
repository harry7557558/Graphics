#include <cmath>
#include <stdio.h>
#include <iostream>



#pragma region Computational Geometry

#define PI 3.1415926535897932384626
#define mix(x,y,a) ((x)*(1.0-(a))+(y)*(a))
#define clamp(x,a,b) ((x)<(a)?(a):(x)>(b)?(b):(x))

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
};

// square of distance to a line segment
double sdSqLine(vec2 p, vec2 a, vec2 b) {
	vec2 pa = p - a, ba = b - a;
	double h = dot(pa, ba) / dot(ba, ba);
	return (pa - ba * clamp(h, 0.0, 1.0)).sqr();
}

// test if segment a and b intersect each other
bool intersect(vec2 ap, vec2 aq, vec2 bp, vec2 bq) {
	if (det(aq - ap, bp - ap) * det(aq - ap, bq - ap) >= 0) return false;
	if (det(bq - bp, ap - bp) * det(bq - bp, aq - bp) >= 0) return false;
	return true;
}
// return the point of intersection
vec2 Intersect(vec2 P0, vec2 P1, vec2 d0, vec2 d1) {
	double t = det(d1, P1 - P0) / det(d1, d0);
	return P0 + d0 * t;
}


#pragma endregion



#include <vector>
typedef std::vector<vec2> polygon;

// output svg path to console or file (usually for debug)
void printPolygon(std::ostream& os, const polygon &p) {
	for (int i = 0; i < p.size(); i++) {
		os << (i == 0 ? "M " : "L ") << p[i].x << " " << p[i].y << " ";
	}
	os << "Z\n";
}




// It should be guarenteed that the number of vertexes of the polygon is at least 3.

// # require no self-intersection
// @ require convexity
// $ general case (may not work in special case)


// $ return true if the polygon is self-intersecting
bool isSelfIntersecting(const polygon &p) {  // brute force approach
	for (int i = 0, n = p.size(); i < n - 2; i++) for (int j = i + 2; j < n; j++)
		if (intersect(p[i], p[i + 1], p[j], p[(j + 1) % n])) return true;
	return false;
}

// $ return true if the polygon is convex
bool isConvex(const polygon &p) {  // change all >0 to >=0 if this polygon is clockwise
	bool k = det(p[0] - p[1], p[2] - p[1]) > 0;
	for (int i = 1, n = p.size(); i < n; i++) {
		if (k != (det(p[i] - p[(i + 1) % n], p[(i + 2) % n] - p[(i + 1) % n]) > 0)) return false;
	}
	return true;
}


// return the perimeter of a polygon
double calcPerimeter(const polygon &p) {
	double P = 0;
	for (int i = 0, n = p.size(); i < n; i++) {
		P += length(p[(i + 1) % n] - p[i]);
	}
	return P;
}

// # return the (signed) area of a polygon
double calcArea(const polygon &p) {
	double A = 0;
	for (int i = 0, n = p.size(); i < n; i++) {
		A += det(p[i], p[(i + 1) % n]);
	}
	return 0.5*A;
}


// # return the average of all vertexes
vec2 calcCenter(const polygon &p) {
	vec2 C(0.0);
	for (int i = 0; i < p.size(); i++) {
		C = C + p[i];
	}
	return C * (1.0 / p.size());
}

// ## return the center of mass
vec2 calcCOM(const polygon &p) {
	vec2 C(0.0); double A = 0, dA;
	for (int i = 0, n = p.size(); i < n; i++) {
		dA = det(p[i], p[(i + 1) % n]), A += dA;
		C = C + (p[i] + p[(i + 1) % n]) * dA;
	}
	return C * (1. / (3.*A));
}
