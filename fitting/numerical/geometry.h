

#ifndef __INC_GEOMETRY_H

#define __INC_GEOMETRY_H

#include <cmath>
#include <math.h>
#include <stdlib.h>

#ifndef PI
#define PI 3.1415926535897932384626
#endif

// for macros, do not use rand() because it will be called multiple times
#ifndef max
#define max(x,y) ((x)>(y)?(x):(y))
#endif
#ifndef min
#define min(x,y) ((x)<(y)?(x):(y))
#endif
#define clamp(x,a,b) ((x)<(a)?(a):(x)>(b)?(b):(x))
#define mix(x,y,a) ((x)*(1-a)+(y)*(a))

#define invsqrt(x) (1.0/sqrt(x))


// a sketchy planar vector template

class vec2 {
public:
	double x, y;
	explicit vec2() {}
	explicit vec2(const double &a) :x(a), y(a) {}
	explicit vec2(const double &x, const double &y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (const vec2 &v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (const vec2 &v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (const vec2 &v) const { return vec2(x * v.x, y * v.y); }
	vec2 operator * (const double &a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; }
	friend double length(const vec2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(const vec2 &v) { return v * invsqrt(v.x*v.x + v.y*v.y); }
	friend double dot(const vec2 &u, const vec2 &v) { return u.x*v.x + u.y*v.y; }
	friend double det(const vec2 &u, const vec2 &v) { return u.x*v.y - u.y*v.x; }

	void operator += (const vec2 &v) { x += v.x, y += v.y; }
	void operator -= (const vec2 &v) { x -= v.x, y -= v.y; }
	void operator *= (const vec2 &v) { x *= v.x, y *= v.y; }
	friend vec2 operator * (const double &a, const vec2 &v) { return vec2(a*v.x, a*v.y); }
	void operator *= (const double &a) { x *= a, y *= a; }
	vec2 operator / (const double &a) const { return vec2(x / a, y / a); }
	void operator /= (const double &a) { x /= a, y /= a; }

	vec2 yx() const { return vec2(y, x); }
	vec2 rot() const { return vec2(-y, x); }
	vec2 rotr() const { return vec2(y, -x); }

	bool operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	bool operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	void operator /= (const vec2 &v) { x /= v.x, y /= v.x; }
	friend vec2 pMax(const vec2 &a, const vec2 &b) { return vec2(max(a.x, b.x), max(a.y, b.y)); }
	friend vec2 pMin(const vec2 &a, const vec2 &b) { return vec2(min(a.x, b.x), min(a.y, b.y)); }
	friend vec2 abs(const vec2 &a) { return vec2(abs(a.x), abs(a.y)); }
	friend vec2 floor(const vec2 &a) { return vec2(floor(a.x), floor(a.y)); }
	friend vec2 ceil(const vec2 &a) { return vec2(ceil(a.x), ceil(a.y)); }
	friend vec2 sqrt(const vec2 &a) { return vec2(sqrt(a.x), sqrt(a.y)); }
	friend vec2 sin(const vec2 &a) { return vec2(sin(a.x), sin(a.y)); }
	friend vec2 cos(const vec2 &a) { return vec2(cos(a.x), cos(a.y)); }
	friend vec2 atan(const vec2 &a) { return vec2(atan(a.x), atan(a.y)); }
};


// a more sketchy 3d vector template

class vec3 {
public:
	double x, y, z;
	vec3() {}
	explicit vec3(double a) :x(a), y(a), z(a) {}
	explicit vec3(double x, double y, double z) :x(x), y(y), z(z) {}
	explicit vec3(vec2 p, double z = 0) :x(p.x), y(p.y), z(z) {}
	vec3 operator - () const { return vec3(-x, -y, -z); }
	vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator * (const vec3 &v) const { return vec3(x*v.x, y*v.y, z*v.z); }  // element wise
	vec3 operator * (const double &k) const { return vec3(k * x, k * y, k * z); }
	double sqr() { return x * x + y * y + z * z; }
	friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3 normalize(vec3 v) { return v * invsqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
	friend double det(vec3 a, vec3 b, vec3 c) { return dot(a, cross(b, c)); }

	void operator += (const vec3 &v) { x += v.x, y += v.y, z += v.z; }
	void operator -= (const vec3 &v) { x -= v.x, y -= v.y, z -= v.z; }
	void operator *= (const vec3 &v) { x *= v.x, y *= v.y, z *= v.z; }
	friend vec3 operator * (const double &a, const vec3 &v) { return vec3(a*v.x, a*v.y, a*v.z); }
	void operator *= (const double &a) { x *= a, y *= a, z *= a; }
	vec3 operator / (const double &a) const { return vec3(x / a, y / a, z / a); }
	void operator /= (const double &a) { x /= a, y /= a, z /= a; }

	vec2 xy() { return vec2(x, y); }
	bool operator == (const vec3 &v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator != (const vec3 &v) const { return x != v.x || y != v.y || z != v.z; }
	friend vec3 pMax(const vec3 &a, const vec3 &b) { return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
	friend vec3 pMin(const vec3 &a, const vec3 &b) { return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
};

class mat3 {
public:
	double v[3][3];
	mat3() {}
	explicit mat3(double k) {  // diagonal matrix
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			v[i][j] = k * (i == j);
	}
	explicit mat3(vec3 lambda) {  // diagonal matrix
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			v[i][j] = i == j ? ((double*)&lambda)[i] : 0;
	}
	explicit mat3(vec3 lambda, double xy, double xz, double yz) {  // symmetric matrix
		v[0][0] = lambda.x, v[1][1] = lambda.y, v[2][2] = lambda.z;
		v[0][1] = v[1][0] = xy, v[0][2] = v[2][0] = xz, v[1][2] = v[2][1] = yz;
	}
	explicit mat3(vec3 i, vec3 j, vec3 k) {  // matrix by column vectors
		for (int u = 0; u < 3; u++) v[u][0] = ((double*)&i)[u], v[u][1] = ((double*)&j)[u], v[u][2] = ((double*)&k)[u];
	}
	explicit mat3(double _00, double _01, double _02, double _10, double _11, double _12, double _20, double _21, double _22) {  // ordered in row-wise
		v[0][0] = _00, v[0][1] = _01, v[0][2] = _02, v[1][0] = _10, v[1][1] = _11, v[1][2] = _12, v[2][0] = _20, v[2][1] = _21, v[2][2] = _22;
	}
	vec3 row(int i) const { return vec3(v[i][0], v[i][1], v[i][2]); }
	vec3 column(int i) const { return vec3(v[0][i], v[1][i], v[2][i]); }
	vec3 diag() const { return vec3(v[0][0], v[1][1], v[2][2]); }
	void operator += (const mat3 &m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] += (&m.v[0][0])[i]; }
	void operator -= (const mat3 &m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] -= (&m.v[0][0])[i]; }
	void operator *= (double m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] *= m; }
	mat3 operator + (const mat3 &m) const { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] + (&m.v[0][0])[i]; return r; }
	mat3 operator - (const mat3 &m) const { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] - (&m.v[0][0])[i]; return r; }
	mat3 operator * (double m) const { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] * m; return r; }
	friend mat3 operator * (double a, const mat3 &m) { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = a * (&m.v[0][0])[i]; return r; }
	friend double determinant(const mat3 &m) { return m.v[0][0] * (m.v[1][1] * m.v[2][2] - m.v[1][2] * m.v[2][1]) - m.v[0][1] * (m.v[1][0] * m.v[2][2] - m.v[1][2] * m.v[2][0]) + m.v[0][2] * (m.v[1][0] * m.v[2][1] - m.v[1][1] * m.v[2][0]); }
	friend double trace(const mat3 &m) { return m.v[0][0] + m.v[1][1] + m.v[2][2]; }
	friend double sumsqr(const mat3 &m) { double r = 0; for (int i = 0; i < 9; i++) r += (&m.v[0][0])[i] * (&m.v[0][0])[i]; return r; }  // sum of square of elements

	vec3 operator * (const vec3 &a) const { return vec3(v[0][0] * a.x + v[0][1] * a.y + v[0][2] * a.z, v[1][0] * a.x + v[1][1] * a.y + v[1][2] * a.z, v[2][0] * a.x + v[2][1] * a.y + v[2][2] * a.z); }
};

mat3 tensor(vec3 u, vec3 v) { return mat3(u*v.x, u*v.y, u*v.z); }
mat3 axis_angle(vec3 n, double a) {
	n = normalize(n); double ct = cos(a), st = sin(a);
	return mat3(
		ct + n.x*n.x*(1 - ct), n.x*n.y*(1 - ct) - n.z*st, n.x*n.z*(1 - ct) + n.y*st,
		n.y*n.x*(1 - ct) + n.z*st, ct + n.y*n.y*(1 - ct), n.y*n.z*(1 - ct) - n.x*st,
		n.z*n.x*(1 - ct) - n.y*st, n.z*n.y*(1 - ct) + n.x*st, ct + n.z*n.z*(1 - ct)
	);
}



double degree(double rad) {
	return rad * (180. / PI);
}


#endif // __INC_GEOMETRY_H

