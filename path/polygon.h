#pragma region Computational Geometry

#include <cmath>
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

// output polygon as svg path (usually for debug)
#include <sstream>
std::wstring sprintPolygon(const polygon &p) {
	std::wostringstream os;
	for (int i = 0, n = p.size(); i < n; i++) {
		os << (i == 0 ? L"M " : L"L ") << p[i].x << L" " << p[i].y << L" ";
	}
	os << L"Z";
	return os.str();
}
#define printPolygon(p) wprintf(L"%s\n", &sprintPolygon(p)[0])


// polygon vertex order
void shiftForward(polygon &P) {
	P.insert(P.begin(), P[P.size() - 1]), P.pop_back();
}
void shiftBackward(polygon &P) {
	P.push_back(P[0]), P.erase(P.begin());
}
void reverseDirection(polygon &P) {
	for (int i = 1, n = P.size(); i <= (n - 1) / 2; i++) std::swap(P[i], P[n - i]);
}

// return false if already (counter)clockwise
// for only testing cw and ccw, call calcArea()
bool forceClockwise(polygon &P) {
	double A = 0;
	for (int i = 0, n = P.size(); i < n; i++) A += det(P[i], P[(i + 1) % n]);
	if (A <= 0) return false;
	reverseDirection(P);
	return true;
}
bool forceCounterClockwise(polygon &P) {
	double A = 0;
	for (int i = 0, n = P.size(); i < n; i++) A += det(P[i], P[(i + 1) % n]);
	if (A >= 0) return false;
	reverseDirection(P);
	return true;
}

// force cw/ccw for polygons with self-intersection
bool expandSelfIntersection(polygon &P);
bool forceClockwiseEI(polygon &P) {
	expandSelfIntersection(P);
	return forceClockwise(P);
}
bool forceCounterClockwiseEI(polygon &P) {
	expandSelfIntersection(P);
	return forceCounterClockwise(P);
}
auto& forceAntiClockwise = forceCounterClockwise;
auto& forceAntiClockwiseEI = forceCounterClockwiseEI;



// geometric transformations

void translatePolygon(polygon &S, vec2 p) {
	for (unsigned i = 0; i < S.size(); i++) S[i] = S[i] + p;
}

void rotatePolygon(polygon &S, vec2 C, double a) {
	vec2 R = vec2(sin(a), cos(a)), q;
	for (unsigned i = 0; i < S.size(); i++)
		q = S[i] - C, S[i] = C + vec2(det(q, R), dot(R, q));
}

void scalePolygon(polygon &S, vec2 C, double s) {
	C = C * (1.0 - s);
	for (unsigned i = 0; i < S.size(); i++) S[i] = S[i] * s + C;
}




// It should be guaranteed that the number of vertices of a polygon is at least 3.

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



// $ test if a point is inside a polygon
bool isInside(const polygon &S, vec2 p) {
	bool r = false; vec2 d;
	for (int i = 0, n = S.size(); i < n; i++) {
		if (p.x > S[i].x == p.x <= S[(i + 1) % n].x &&
			det(d = S[(i + 1) % n] - S[i], p - S[i]) * d.x > 0.0) r = !r;
	}
	return r;
}



// return the perimeter of a polygon
double calcPerimeter(const polygon &p) {
	double P = 0;
	for (int i = 0, n = p.size(); i < n; i++) {
		P += length(p[(i + 1) % n] - p[i]);
	}
	return P;
}

// # return the area of a polygon, negative when vertices ordered clockwise
double calcArea(const polygon &p) {
	double A = 0;
	for (int i = 0, n = p.size(); i < n; i++) {
		A += det(p[i], p[(i + 1) % n]);
	}
	return 0.5*A;
}



// return the average of all vertices
vec2 calcCenter(const polygon &p) {
	vec2 C(0.0);
	for (int i = 0, n = p.size(); i < n; i++) C = C + p[i];
	return C * (1.0 / p.size());
}

// # return the center of mass
vec2 calcCOM(const polygon &p) {
	vec2 C(0.0); double A = 0, dA;
	for (int i = 0, n = p.size(); i < n; i++) {
		dA = det(p[i], p[(i + 1) % n]), A += dA;
		C = C + (p[i] + p[(i + 1) % n]) * dA;
	}
	return C * (1. / (3.*A));
}

// return the average of all edges
vec2 calcCenterE(const polygon &p) {
	vec2 C(0.0); double S = 0, dS;
	for (int i = 0, n = p.size(); i < n; i++) {
		dS = length(p[(i + 1) % n] - p[i]), S += dS;
		C = C + (p[i] + p[(i + 1) % n]) * dS;
	}
	return C * (0.5 / S);
}





// calculate the Axis-Aligned Bounding Box
void calcAABB(const polygon &p, vec2 &Min, vec2 &Max) {
	Min = Max = p[0];
	for (int i = 1, n = p.size(); i < n; i++) {
		if (p[i].x < Min.x) Min.x = p[i].x;
		if (p[i].y < Min.y) Min.y = p[i].y;
		if (p[i].x > Max.x) Max.x = p[i].x;
		if (p[i].y > Max.y) Max.y = p[i].y;
	}
}


// calculate the Convex Hull
#include <algorithm>
polygon calcConvexHull(polygon P) {
	int N = P.size();
	std::sort(P.begin(), P.begin() + N, [](const vec2 &p, const vec2 &q)->bool {
		return p.x == q.x ? p.y < q.y : p.x < q.x;  // sort by x; if x equals, sort by y
	});
	polygon u;
	for (int d = 0, n = 0; d < N;) {
		if (n <= 1) u.push_back(P[d]), n++;
		else {
			if (det(P[d] - u[n - 2], u[n - 1] - u[n - 2]) >= 0) u[n - 1] = P[d];
			else u.push_back(P[d]), n++;
		}
		while (n > 2 && det(u[n - 1] - u[n - 3], u[n - 2] - u[n - 3]) >= 0) u.erase(u.begin() + n - 2), n--;
		int _d = d;
		while (d < N && P[d].x == P[_d].x) d++;
	}
	polygon v;
	for (int d = N - 1, n = 0; d >= 0;) {
		if (n <= 1) v.push_back(P[d]), n++;
		else {
			if (det(P[d] - v[n - 2], v[n - 1] - v[n - 2]) >= 0) v[n - 1] = P[d];
			else v.push_back(P[d]), n++;
		}
		while (n > 2 && det(v[n - 1] - v[n - 3], v[n - 2] - v[n - 3]) >= 0) v.erase(v.begin() + n - 2), n--;
		int _d = d;
		while (d >= 0 && P[d].x == P[_d].x) d--;
	}
	u.insert(u.end(), u.back().y == v[0].y ? v.begin() + 1 : v.begin(), v.back().y == u[0].y ? v.end() - 1 : v.end());
	return u;
}


// https://www.nayuki.io/page/smallest-enclosing-circle
//void calcBoundingCircle(polygon P, vec2 &C, vec2 &R);





#define _isEqual(p,q) (((p)-(q)).sqr()<1e-16)
#define _isCollinear(a,b,c) (abs(det((b)-(a),(c)-(a)))<1e-8)

auto _polygon_pushback = [](polygon &fig, vec2 p) {
	if (0.0*p.x*p.y != 0.0) return;   // NAN, necessary for pathfinder functions
	if (fig.empty()) fig.push_back(p);
	else if (!_isEqual(p, fig.back())) {
		if (fig.size() >= 2 && _isCollinear(p, fig.back(), fig[fig.size() - 2])) fig.back() = p;
		else fig.push_back(p);
	}
};


// $ return false if no self-intersection found - runs in O(N²)
bool expandSelfIntersection(polygon &P) {
	// debug
	return false;
}


// $ cut a polygon, where dot(p-p0,n)>0 part is cut off
polygon cutPolygonFromPlane(const polygon &fig, vec2 p0, vec2 n) {
	const double c = dot(p0, n);
	int l = fig.size();
	polygon res;

	// find a point that will not be cut off
	int d0;
	for (d0 = 0; d0 < l; d0++) {
		if (dot(fig[d0], n) < c) break;
	}
	if (d0 >= l) return res;  // the whole shape is cut off

	// trace segment
	auto intersect = [](vec2 p, vec2 q, const double &d, const vec2 &n)->vec2 {
		q = q - p;
		double t = (d - dot(n, p)) / dot(n, q);   // sometimes NAN
		return p + q * t;
	};
	for (int i = 0, d = d0, e = (d + 1) % l; i < l; i++, d = e, e = (e + 1) % l) {
		if (dot(fig[d], n) < c) {
			if (dot(fig[e], n) <= c) _polygon_pushback(res, fig[e]);
			else _polygon_pushback(res, intersect(fig[d], fig[e], c, n));
		}
		else {
			if (dot(fig[e], n) > c);
			else {
				_polygon_pushback(res, intersect(fig[d], fig[e], c, n));
				_polygon_pushback(res, fig[e]);
			}
		}
	}

	// occurs when the line goes through a vertex
	if (_isEqual(res[0], res.back())) res.pop_back();

	// occurs when an edge lies on the line
	bool op = false; do {
		if ((l = res.size()) < 3) return polygon();
		if (op = _isCollinear(res[0], res.back(), res[l - 2])) res.pop_back();
		else if (op = _isCollinear(res.back(), res[0], res[1])) res.erase(res.begin());
	} while (op);

	return res;
}


// 

