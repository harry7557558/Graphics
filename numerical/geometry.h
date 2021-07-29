// vector, matrix, small useful functions


#ifndef __INC_GEOMETRY_H

#define __INC_GEOMETRY_H

#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef PI
#define PI 3.1415926535897932384626
#endif

// defining max and min as macros may cause problems in standard libraries
// NO `using namespace std;`
#undef max
#undef min
template<typename T> inline T max(T x, T y) { return (x > y ? x : y); }
template<typename T> inline T min(T x, T y) { return (x < y ? x : y); }
template<typename T, typename t> inline T clamp(T x, t a, t b) { return (x<a ? a : x>b ? b : x); }
template<typename T, typename f> inline T mix(T x, T y, f a) { return (x * (f(1) - a) + y * a); }  // lerp
template<typename T> inline T mod(T x, T y) { return x - y * floor(x / y); }
template<typename T> inline T fract(T x) { return x - floor(x); }
template<typename T> inline T sign(T x) { return (T)(x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0); }
template<typename T> inline T smoothstep(T edge0, T edge1, T x) {
	T t = clamp((x - edge0) / (edge1 - edge0), T(0.0), T(1.0));
	return t * t * (T(3.0) - T(2.0) * t);
}

// for double precision
#define invsqrt(x) (1.0/sqrt(x))

// fast inverse square root - u sure this is faster ??!!
// https://en.wikipedia.org/wiki/Fast_inverse_square_root
inline float invsqrt_fast_f(float number) {
	int32_t i;  // or uint32_t
	float x2, y;
	x2 = number * 0.5f;
	y = number;
	i = *(int32_t*)&y;                       // evil floating point bit level hacking
	i = 0x5f375a86 - (i >> 1);               // what the ****?
	y = *(float*)&i;
	y = y * (1.5f - (x2 * y * y));   // 1st iteration
	// y  = y * ( 1.5f - ( x2 * y * y ) );   // 2nd iteration, this can be removed
	return y;
}


// a sketchy planar vector template

struct vec2;
struct vec2f;

struct ivec2 {
	int x, y;
	ivec2() {}
	ivec2(int a) :x(a), y(a) {}
	ivec2(int x, int y) :x(x), y(y) {}
	ivec2(vec2 p);
	bool operator == (const ivec2 &v) const { return x == v.x && y == v.y; }
	bool operator != (const ivec2 &v) const { return x != v.x || y != v.y; }
	ivec2 operator - () const { return ivec2(-x, -y); }
	ivec2 operator + (const ivec2 &v) const { return ivec2(x + v.x, y + v.y); }
	ivec2 operator - (const ivec2 &v) const { return ivec2(x - v.x, y - v.y); }
	ivec2 operator * (const ivec2 &v) const { return ivec2(x * v.x, y * v.y); }
	ivec2 operator * (const int &a) const { return ivec2(x*a, y*a); }
	friend ivec2 operator * (const int &a, const ivec2 &v) { return ivec2(a*v.x, a*v.y); }
	ivec2 operator / (const int &a) const { return ivec2(x / a, y / a); }
	ivec2 operator % (const int &a) const { return ivec2(x % a, y % a); }
	ivec2 yx() const { return ivec2(y, x); }
	friend int dot(const ivec2 &u, const ivec2 &v) { return u.x*v.x + u.y*v.y; }
	friend int det(const ivec2 &u, const ivec2 &v) { return u.x*v.y - u.y*v.x; }
};

struct vec2 {
	double x, y;
	explicit vec2() {}
	explicit vec2(const double &a) :x(a), y(a) {}
	explicit vec2(const double &x, const double &y) :x(x), y(y) {}
	explicit vec2(const ivec2 &p) :x(p.x), y(p.y) {}
	explicit vec2(const vec2f &p);

	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (const vec2 &v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (const vec2 &v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (const vec2 &v) const { return vec2(x * v.x, y * v.y); }
	vec2 operator * (const double &a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; }
	friend double length(const vec2 &v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(const vec2 &v) { double m = invsqrt(v.x*v.x + v.y*v.y); return vec2(v.x*m, v.y*m); }
	friend double dot(const vec2 &u, const vec2 &v) { return u.x*v.x + u.y*v.y; }
	friend double det(const vec2 &u, const vec2 &v) { return u.x*v.y - u.y*v.x; }
	friend double ndot(const vec2 &u, const vec2 &v) { return dot(u, v) * invsqrt(u.sqr()*v.sqr()); }
	friend double ndet(const vec2 &u, const vec2 &v) { return det(u, v) * invsqrt(u.sqr()*v.sqr()); }

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
	vec2 rotate(double a) const { double ca = cos(a), sa = sin(a); return vec2(ca*x - sa * y, sa*x + ca * y); }

	bool operator == (const vec2 &v) const { return x == v.x && y == v.y; }
	bool operator != (const vec2 &v) const { return x != v.x || y != v.y; }
	vec2 operator / (const vec2 &v) const { return vec2(x / v.x, y / v.y); }
	void operator /= (const vec2 &v) { x /= v.x, y /= v.x; }
	friend vec2 pMax(const vec2 &a, const vec2 &b) { return vec2(max(a.x, b.x), max(a.y, b.y)); }
	friend vec2 pMin(const vec2 &a, const vec2 &b) { return vec2(min(a.x, b.x), min(a.y, b.y)); }
	friend vec2 abs(const vec2 &a) { return vec2(abs(a.x), abs(a.y)); }
	friend vec2 floor(const vec2 &a) { return vec2(floor(a.x), floor(a.y)); }
	friend vec2 ceil(const vec2 &a) { return vec2(ceil(a.x), ceil(a.y)); }
	friend vec2 fract(const vec2 &a) { return vec2(fract(a.x), fract(a.y)); }
	friend vec2 sqrt(const vec2 &a) { return vec2(sqrt(a.x), sqrt(a.y)); }
	friend vec2 sin(const vec2 &a) { return vec2(sin(a.x), sin(a.y)); }
	friend vec2 cos(const vec2 &a) { return vec2(cos(a.x), cos(a.y)); }
	friend vec2 atan(const vec2 &a) { return vec2(atan(a.x), atan(a.y)); }
	friend vec2 exp(const vec2 &a) { return vec2(exp(a.x), exp(a.y)); }
	friend vec2 log(const vec2 &a) { return vec2(log(a.x), log(a.y)); }
};

struct vec2f {
	float x, y;
	explicit vec2f() {}
	explicit vec2f(const float &a) :x(a), y(a) {}
	explicit vec2f(const float &x, const float &y) :x(x), y(y) {}
	//explicit vec2f(const int &x, const int &y) :x((float)x), y((float)y) {}
	explicit vec2f(vec2 p) :x((float)p.x), y((float)p.y) {};
	explicit vec2f(ivec2 p) :x((float)p.x), y((float)p.y) {};

	vec2f operator - () const { return vec2f(-x, -y); }
	vec2f operator + (const vec2f &v) const { return vec2f(x + v.x, y + v.y); }
	vec2f operator - (const vec2f &v) const { return vec2f(x - v.x, y - v.y); }
	vec2f operator * (const vec2f &v) const { return vec2f(x * v.x, y * v.y); }
	vec2f operator / (const vec2f &v) const { return vec2f(x / v.x, y / v.y); }
	vec2f operator * (const float &a) const { return vec2f(x*a, y*a); }
	float sqr() const { return x * x + y * y; }
	friend float length(const vec2f &v) { return sqrtf(v.x*v.x + v.y*v.y); }
	friend vec2f normalize(const vec2f &v) { float m = 1.f / sqrtf(v.x*v.x + v.y*v.y); return vec2f(v.x*m, v.y*m); }
	friend float dot(const vec2f &u, const vec2f &v) { return u.x*v.x + u.y*v.y; }
	friend float det(const vec2f &u, const vec2f &v) { return u.x*v.y - u.y*v.x; }

	void operator += (const vec2f &v) { x += v.x, y += v.y; }
	void operator -= (const vec2f &v) { x -= v.x, y -= v.y; }
	void operator *= (const vec2f &v) { x *= v.x, y *= v.y; }
	friend vec2f operator * (const float &a, const vec2f &v) { return vec2f(a*v.x, a*v.y); }
	void operator *= (const float &a) { x *= a, y *= a; }
	vec2f operator / (const float &a) const { return vec2f(x / a, y / a); }
	void operator /= (const float &a) { x /= a, y /= a; }

	vec2f yx() const { return vec2f(y, x); }
	vec2f rot() const { return vec2f(-y, x); }
	vec2f rotr() const { return vec2f(y, -x); }

	friend vec2f abs(const vec2f &a) { return vec2f(abs(a.x), abs(a.y)); }
	friend vec2f floor(const vec2f &a) { return vec2f(floor(a.x), floor(a.y)); }
	friend vec2f ceil(const vec2f &a) { return vec2f(ceil(a.x), ceil(a.y)); }
	friend vec2f fract(const vec2f &a) { return vec2f(fract(a.x), fract(a.y)); }
	friend vec2f sin(const vec2f &a) { return vec2f(sin(a.x), sin(a.y)); }
	friend vec2f cos(const vec2f &a) { return vec2f(cos(a.x), cos(a.y)); }
	friend vec2f sqrt(const vec2f &a) { return vec2f(sqrt(a.x), sqrt(a.y)); }
	friend vec2f exp(const vec2f &a) { return vec2f(exp(a.x), exp(a.y)); }
	friend vec2f log(const vec2f &a) { return vec2f(log(a.x), log(a.y)); }
};

ivec2::ivec2(vec2 p) :x((int)p.x), y((int)p.y) {}
vec2::vec2(const vec2f &p) : x((double)p.x), y((double)p.y) {}

vec2 sincos(double a) { return vec2(sin(a), cos(a)); }
vec2 cossin(double a) { return vec2(cos(a), sin(a)); }



struct mat2 {
public:
	double v[2][2];
	mat2() {}
	explicit mat2(double k) {
		v[0][0] = v[1][1] = k, v[1][0] = v[0][1] = 0;
	}
	explicit mat2(vec2 lambda) {
		v[0][0] = lambda.x, v[1][1] = lambda.y;
		v[1][0] = v[0][1] = 0;
	}
	explicit mat2(vec2 col1, vec2 col2) {
		v[0][0] = col1.x, v[1][0] = col1.y;
		v[0][1] = col2.x, v[1][1] = col2.y;
	}
	explicit mat2(double _00, double _10, double _01, double _11) {  // by columns
		v[0][0] = _00, v[0][1] = _01, v[1][0] = _10, v[1][1] = _11;
	}
	vec2 row(int i) const { return vec2(v[i][0], v[i][1]); }
	vec2 column(int i) const { return vec2(v[0][i], v[1][i]); }
	vec2 diag() const { return vec2(v[0][0], v[1][1]); }
	mat2 operator - () const { return mat2(-v[0][0], -v[1][0], -v[0][1], -v[1][1]); }
	void operator += (const mat2 &m) { for (int i = 0; i < 4; i++) (&v[0][0])[i] += (&m.v[0][0])[i]; }
	void operator -= (const mat2 &m) { for (int i = 0; i < 4; i++) (&v[0][0])[i] -= (&m.v[0][0])[i]; }
	void operator *= (double m) { for (int i = 0; i < 4; i++) (&v[0][0])[i] *= m; }
	mat2 operator + (const mat2 &m) const { mat2 r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = (&v[0][0])[i] + (&m.v[0][0])[i]; return r; }
	mat2 operator - (const mat2 &m) const { mat2 r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = (&v[0][0])[i] - (&m.v[0][0])[i]; return r; }
	mat2 operator * (double m) const { mat2 r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = (&v[0][0])[i] * m; return r; }
	friend mat2 operator * (double a, const mat2 &m) { mat2 r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = a * (&m.v[0][0])[i]; return r; }
	mat2 operator * (const mat2 &m) const { return mat2(v[0][0] * m.v[0][0] + v[0][1] * m.v[1][0], v[1][0] * m.v[0][0] + v[1][1] * m.v[1][0], v[0][0] * m.v[0][1] + v[0][1] * m.v[1][1], v[1][0] * m.v[0][1] + v[1][1] * m.v[1][1]); }
	friend double determinant(const mat2 &m) { return m.v[0][0] * m.v[1][1] - m.v[0][1] * m.v[1][0]; }
	mat2 transpose() const { mat2 r; for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) r.v[i][j] = v[j][i]; return r; }
	mat2 inverse() const { double d = 1.0 / (v[0][0] * v[1][1] - v[0][1] * v[1][0]); return mat2(d*v[1][1], -d * v[1][0], -d * v[0][1], d*v[0][0]); }
	friend double trace(const mat2 &m) { return m.v[0][0] + m.v[1][1]; }
	vec2 operator * (const vec2 &a) const { return vec2(v[0][0] * a.x + v[0][1] * a.y, v[1][0] * a.x + v[1][1] * a.y); }
};

mat2 rotationMatrix2d(double a) {
	double ca = cos(a), sa = sin(a);
	return mat2(ca, -sa, sa, ca);
}


struct mat2f {
public:
	float v[2][2];
	mat2f() {}
	explicit mat2f(float k) {
		v[0][0] = v[1][1] = k, v[1][0] = v[0][1] = 0;
	}
	explicit mat2f(vec2f lambda) {
		v[0][0] = lambda.x, v[1][1] = lambda.y;
		v[1][0] = v[0][1] = 0;
	}
	explicit mat2f(vec2f col1, vec2f col2) {
		v[0][0] = col1.x, v[1][0] = col1.y;
		v[0][1] = col2.x, v[1][1] = col2.y;
	}
	explicit mat2f(float _00, float _10, float _01, float _11) {  // by columns
		v[0][0] = _00, v[0][1] = _01, v[1][0] = _10, v[1][1] = _11;
	}
	vec2f row(int i) const { return vec2f(v[i][0], v[i][1]); }
	vec2f column(int i) const { return vec2f(v[0][i], v[1][i]); }
	vec2f diag() const { return vec2f(v[0][0], v[1][1]); }
	mat2f operator - () const { return mat2f(-v[0][0], -v[1][0], -v[0][1], -v[1][1]); }
	void operator += (const mat2f &m) { for (int i = 0; i < 4; i++) (&v[0][0])[i] += (&m.v[0][0])[i]; }
	void operator -= (const mat2f &m) { for (int i = 0; i < 4; i++) (&v[0][0])[i] -= (&m.v[0][0])[i]; }
	void operator *= (float m) { for (int i = 0; i < 4; i++) (&v[0][0])[i] *= m; }
	mat2f operator + (const mat2f &m) const { mat2f r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = (&v[0][0])[i] + (&m.v[0][0])[i]; return r; }
	mat2f operator - (const mat2f &m) const { mat2f r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = (&v[0][0])[i] - (&m.v[0][0])[i]; return r; }
	mat2f operator * (float m) const { mat2f r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = (&v[0][0])[i] * m; return r; }
	friend mat2f operator * (float a, const mat2f &m) { mat2f r; for (int i = 0; i < 4; i++) (&r.v[0][0])[i] = a * (&m.v[0][0])[i]; return r; }
	mat2f operator * (const mat2f &m) const { return mat2f(v[0][0] * m.v[0][0] + v[0][1] * m.v[1][0], v[1][0] * m.v[0][0] + v[1][1] * m.v[1][0], v[0][0] * m.v[0][1] + v[0][1] * m.v[1][1], v[1][0] * m.v[0][1] + v[1][1] * m.v[1][1]); }
	friend float determinant(const mat2f &m) { return m.v[0][0] * m.v[1][1] - m.v[0][1] * m.v[1][0]; }
	mat2f transpose() const { mat2f r; for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++) r.v[i][j] = v[j][i]; return r; }
	mat2f inverse() const { float d = 1.0f / (v[0][0] * v[1][1] - v[0][1] * v[1][0]); return mat2f(d*v[1][1], -d * v[1][0], -d * v[0][1], d*v[0][0]); }
	friend float trace(const mat2f &m) { return m.v[0][0] + m.v[1][1]; }
	vec2f operator * (const vec2f &a) const { return vec2f(v[0][0] * a.x + v[0][1] * a.y, v[1][0] * a.x + v[1][1] * a.y); }
};




// a more sketchy 3d vector template

struct vec3;
struct vec3f;

struct ivec3 {
	int x, y, z;
	ivec3() {}
	explicit ivec3(int a) :x(a), y(a), z(a) {}
	explicit ivec3(int x, int y, int z) : x(x), y(y), z(z) {}
	explicit ivec3(vec3 p);
	explicit ivec3(vec3f p);
	bool operator == (const ivec3 &v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator != (const ivec3 &v) const { return x != v.x || y != v.y || z != v.z; }
	ivec3 operator - () const { return ivec3(-x, -y, -z); }
	ivec3 operator + (const ivec3 &v) const { return ivec3(x + v.x, y + v.y, z + v.z); }
	ivec3 operator - (const ivec3 &v) const { return ivec3(x - v.x, y - v.y, z - v.z); }
	ivec3 operator * (const ivec3 &v) const { return ivec3(x * v.x, y * v.y, z * v.z); }
	ivec3 operator * (const int &a) const { return ivec3(x*a, y*a, z*a); }
	friend ivec3 operator * (const int &a, const ivec3 &v) { return ivec3(a*v.x, a*v.y, a*v.z); }
	ivec3 operator / (const int &a) const { return ivec3(x / a, y / a, z / a); }
	ivec3 operator % (const int &a) const { return ivec3(x % a, y % a, z % a); }
};

struct vec3 {
	double x, y, z;
	vec3() {}
	explicit vec3(double a) :x(a), y(a), z(a) {}
	explicit vec3(double x, double y, double z) :x(x), y(y), z(z) {}
	explicit vec3(vec2 p, double z = 0) :x(p.x), y(p.y), z(z) {}
	explicit vec3(ivec3 p) :x(p.x), y(p.y), z(p.z) {}
	explicit vec3(vec3f p);
	vec3 operator - () const { return vec3(-x, -y, -z); }
	vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator * (const vec3 &v) const { return vec3(x * v.x, y * v.y, z * v.z); }  // element wise
	vec3 operator / (const vec3 &v) const { return vec3(x / v.x, y / v.y, z / v.z); }
	vec3 operator + (const double &k) const { return vec3(x + k, y + k, z + k); }
	vec3 operator - (const double &k) const { return vec3(x - k, y - k, z - k); }
	vec3 operator * (const double &k) const { return vec3(x * k, y * k, z * k); }
	vec3 operator / (const double &a) const { return vec3(x / a, y / a, z / a); }
	double sqr() const { return x * x + y * y + z * z; }
	friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3 normalize(vec3 v) { double m = invsqrt(v.x*v.x + v.y*v.y + v.z*v.z); return vec3(v.x*m, v.y*m, v.z*m); }
	friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
	friend double det(vec3 a, vec3 b, vec3 c) { return dot(a, cross(b, c)); }
	friend double ndot(const vec3 &u, const vec3 &v) { return dot(u, v) * invsqrt(u.sqr()*v.sqr()); }
	friend vec3 ncross(vec3 u, vec3 v) { return normalize(cross(u, v)); }

	void operator += (const vec3 &v) { x += v.x, y += v.y, z += v.z; }
	void operator -= (const vec3 &v) { x -= v.x, y -= v.y, z -= v.z; }
	void operator *= (const vec3 &v) { x *= v.x, y *= v.y, z *= v.z; }
	void operator /= (const vec3 &v) { x /= v.x, y /= v.y, z /= v.z; }
	friend vec3 operator * (const double &a, const vec3 &v) { return vec3(a*v.x, a*v.y, a*v.z); }
	void operator += (const double &a) { x += a, y += a, z += a; }
	void operator -= (const double &a) { x -= a, y -= a, z -= a; }
	void operator *= (const double &a) { x *= a, y *= a, z *= a; }
	void operator /= (const double &a) { x /= a, y /= a, z /= a; }

	bool operator == (const vec3 &v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator != (const vec3 &v) const { return x != v.x || y != v.y || z != v.z; }
	friend vec3 pMax(const vec3 &a, const vec3 &b) { return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
	friend vec3 pMin(const vec3 &a, const vec3 &b) { return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
	friend vec3 abs(const vec3 &a) { return vec3(abs(a.x), abs(a.y), abs(a.z)); }
	friend vec3 floor(const vec3 &a) { return vec3(floor(a.x), floor(a.y), floor(a.z)); }
	friend vec3 ceil(const vec3 &a) { return vec3(ceil(a.x), ceil(a.y), ceil(a.z)); }
	friend vec3 fract(const vec3 &a) { return vec3(fract(a.x), fract(a.y), fract(a.z)); }
	friend vec3 sin(const vec3 &a) { return vec3(sin(a.x), sin(a.y), sin(a.z)); }
	friend vec3 cos(const vec3 &a) { return vec3(cos(a.x), cos(a.y), cos(a.z)); }
	friend vec3 sqrt(const vec3 &a) { return vec3(sqrt(a.x), sqrt(a.y), sqrt(a.z)); }
	friend vec3 exp(const vec3 &a) { return vec3(exp(a.x), exp(a.y), exp(a.z)); }
	friend vec3 log(const vec3 &a) { return vec3(log(a.x), log(a.y), log(a.z)); }
	friend vec3 pow(const vec3 &a, const double &e) { return vec3(pow(a.x, e), pow(a.y, e), pow(a.z, e)); }

	vec2 xy() const { return vec2(x, y); }
	vec2 xz() const { return vec2(x, z); }
	vec2 yx() const { return vec2(y, x); }
	vec2 yz() const { return vec2(y, z); }
	vec2 zx() const { return vec2(z, x); }
	vec2 zy() const { return vec2(z, y); }
	vec3 xyz() const { return vec3(x, y, z); }
	vec3 xzy() const { return vec3(x, z, y); }
	vec3 yzx() const { return vec3(y, z, x); }
	vec3 yxz() const { return vec3(y, x, z); }
	vec3 zxy() const { return vec3(z, x, y); }
	vec3 zyx() const { return vec3(z, y, x); }
	vec3 rz90() const { return vec3(-y, x, z); }
	vec3 rz180() const { return vec3(-x, -y, z); }
	vec3 rz270() const { return vec3(y, -x, z); }
	vec3 ry90() const { return vec3(z, y, -x); }
	vec3 ry180() const { return vec3(-x, y, -z); }
	vec3 ry270() const { return vec3(-z, y, x); }
	vec3 rx90() const { return vec3(x, -z, y); }
	vec3 rx180() const { return vec3(x, -y, -z); }
	vec3 rx270() const { return vec3(x, z, -y); }
};

struct vec3f {
	float x, y, z;
	vec3f() {}
	explicit vec3f(float a) :x(a), y(a), z(a) {}
	explicit vec3f(int a) :x((float)a), y((float)a), z((float)a) {}
	explicit vec3f(double a) :x((float)a), y((float)a), z((float)a) {}
	explicit vec3f(float x, float y, float z) :x(x), y(y), z(z) {}
	//explicit vec3f(int x, int y, int z) :x((float)x), y((float)y), z((float)z) {}
	explicit vec3f(vec2f p, float z = 0.0f) :x(p.x), y(p.y), z(z) {}
	explicit vec3f(vec3 p) : x((float)p.x), y((float)p.y), z((float)p.z) {}
	explicit vec3f(ivec3 p) : x((float)p.x), y((float)p.y), z((float)p.z) {}

	vec3f operator - () const { return vec3f(-x, -y, -z); }
	vec3f operator + (const vec3f &v) const { return vec3f(x + v.x, y + v.y, z + v.z); }
	vec3f operator - (const vec3f &v) const { return vec3f(x - v.x, y - v.y, z - v.z); }
	vec3f operator * (const vec3f &v) const { return vec3f(x * v.x, y * v.y, z * v.z); }  // element wise
	vec3f operator / (const vec3f &v) const { return vec3f(x / v.x, y / v.y, z / v.z); }
	vec3f operator + (const float &k) const { return vec3f(x + k, y + k, z + k); }
	vec3f operator - (const float &k) const { return vec3f(x - k, y - k, z - k); }
	vec3f operator * (const float &k) const { return vec3f(x * k, y * k, z * k); }
	vec3f operator / (const float &a) const { return vec3f(x / a, y / a, z / a); }
	float sqr() const { return x * x + y * y + z * z; }
	friend float length(vec3f v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3f normalize(vec3f v) { float m = 1.0f / sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); return vec3f(v.x*m, v.y*m, v.z*m); }
	friend float dot(vec3f u, vec3f v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3f cross(vec3f u, vec3f v) { return vec3f(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
	friend float det(vec3f a, vec3f b, vec3f c) { return dot(a, cross(b, c)); }

	void operator += (const vec3f &v) { x += v.x, y += v.y, z += v.z; }
	void operator -= (const vec3f &v) { x -= v.x, y -= v.y, z -= v.z; }
	void operator *= (const vec3f &v) { x *= v.x, y *= v.y, z *= v.z; }
	void operator /= (const vec3f &v) { x /= v.x, y /= v.y, z /= v.z; }
	friend vec3f operator * (const float &a, const vec3f &v) { return vec3f(a*v.x, a*v.y, a*v.z); }
	void operator += (const float &a) { x += a, y += a, z += a; }
	void operator -= (const float &a) { x -= a, y -= a, z -= a; }
	void operator *= (const float &a) { x *= a, y *= a, z *= a; }
	void operator /= (const float &a) { x /= a, y /= a, z /= a; }

	bool operator == (const vec3f &v) const { return x == v.x && y == v.y && z == v.z; }
	bool operator != (const vec3f &v) const { return x != v.x || y != v.y || z != v.z; }
	friend vec3f pMax(const vec3f &a, const vec3f &b) { return vec3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
	friend vec3f pMin(const vec3f &a, const vec3f &b) { return vec3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
	friend vec3f floor(const vec3f &a) { return vec3f(floor(a.x), floor(a.y), floor(a.z)); }
	friend vec3f ceil(const vec3f &a) { return vec3f(ceil(a.x), ceil(a.y), ceil(a.z)); }
	friend vec3f fract(const vec3f &a) { return vec3f(fract(a.x), fract(a.y), fract(a.z)); }
	friend vec3f sin(const vec3f &a) { return vec3f(sin(a.x), sin(a.y), sin(a.z)); }
	friend vec3f cos(const vec3f &a) { return vec3f(cos(a.x), cos(a.y), cos(a.z)); }
	friend vec3f sqrt(const vec3f &a) { return vec3f(sqrt(a.x), sqrt(a.y), sqrt(a.z)); }
	friend vec3f exp(const vec3f &a) { return vec3f(exp(a.x), exp(a.y), exp(a.z)); }
	friend vec3f log(const vec3f &a) { return vec3f(log(a.x), log(a.y), log(a.z)); }
	friend vec3f pow(const vec3f &a, const float &e) { return vec3f(pow(a.x, e), pow(a.y, e), pow(a.z, e)); }

	vec2f xx() const { return vec2f(x, x); }
	vec2f xy() const { return vec2f(x, y); }
	vec2f xz() const { return vec2f(x, z); }
	vec2f yx() const { return vec2f(y, x); }
	vec2f yy() const { return vec2f(y, y); }
	vec2f yz() const { return vec2f(y, z); }
	vec2f zx() const { return vec2f(z, x); }
	vec2f zy() const { return vec2f(z, y); }
	vec2f zz() const { return vec2f(z, z); }
	vec3f xyz() const { return vec3f(x, y, z); }
	vec3f xzy() const { return vec3f(x, z, y); }
	vec3f yzx() const { return vec3f(y, z, x); }
	vec3f yxz() const { return vec3f(y, x, z); }
	vec3f zxy() const { return vec3f(z, x, y); }
	vec3f zyx() const { return vec3f(z, y, x); }
	vec3f rz90() const { return vec3f(-y, x, z); }
	vec3f rz180() const { return vec3f(-x, -y, z); }
	vec3f rz270() const { return vec3f(y, -x, z); }
	vec3f ry90() const { return vec3f(z, y, -x); }
	vec3f ry180() const { return vec3f(-x, y, -z); }
	vec3f ry270() const { return vec3f(-z, y, x); }
	vec3f rx90() const { return vec3f(x, -z, y); }
	vec3f rx180() const { return vec3f(x, -y, -z); }
	vec3f rx270() const { return vec3f(x, z, -y); }
};

ivec3::ivec3(vec3 p) : x((int)p.x), y((int)p.y), z((int)p.z) {}
ivec3::ivec3(vec3f p) : x((int)p.x), y((int)p.y), z((int)p.z) {}
vec3::vec3(vec3f p) : x((double)p.x), y((double)p.y), z((double)p.z) {}


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
	explicit mat3(double _00, double _10, double _20, double _01, double _11, double _21, double _02, double _12, double _22) {  // ordered in column-wise
		v[0][0] = _00, v[0][1] = _01, v[0][2] = _02, v[1][0] = _10, v[1][1] = _11, v[1][2] = _12, v[2][0] = _20, v[2][1] = _21, v[2][2] = _22;
	}
	double* operator[] (int d) { return &v[d][0]; }
	const double* operator[] (int d) const { return &v[d][0]; }
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
	mat3 transpose() const { mat3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.v[i][j] = v[j][i]; return r; }
	friend mat3 transpose(const mat3 &m) { return mat3(m.v[0][0], m.v[0][1], m.v[0][2], m.v[1][0], m.v[1][1], m.v[1][2], m.v[2][0], m.v[2][1], m.v[2][2]); }
	friend double trace(const mat3 &m) { return m.v[0][0] + m.v[1][1] + m.v[2][2]; }
	friend double sumsqr(const mat3 &m) { double r = 0; for (int i = 0; i < 9; i++) r += (&m.v[0][0])[i] * (&m.v[0][0])[i]; return r; }  // sum of square of elements

	mat3 operator * (const mat3 &A) const {
		mat3 R;
		for (int m = 0; m < 3; m++) for (int n = 0; n < 3; n++) {
			R.v[m][n] = 0;
			for (int i = 0; i < 3; i++) R.v[m][n] += v[m][i] * A.v[i][n];
		}
		return R;
	}
	vec3 operator * (const vec3 &a) const {
		return vec3(
			v[0][0] * a.x + v[0][1] * a.y + v[0][2] * a.z,
			v[1][0] * a.x + v[1][1] * a.y + v[1][2] * a.z,
			v[2][0] * a.x + v[2][1] * a.y + v[2][2] * a.z);
	}
	mat3 inverse() const {
		mat3 R;
		R.v[0][0] = v[1][1] * v[2][2] - v[1][2] * v[2][1], R.v[0][1] = v[0][2] * v[2][1] - v[0][1] * v[2][2], R.v[0][2] = v[0][1] * v[1][2] - v[0][2] * v[1][1];
		R.v[1][0] = v[1][2] * v[2][0] - v[1][0] * v[2][2], R.v[1][1] = v[0][0] * v[2][2] - v[0][2] * v[2][0], R.v[1][2] = v[0][2] * v[1][0] - v[0][0] * v[1][2];
		R.v[2][0] = v[1][0] * v[2][1] - v[1][1] * v[2][0], R.v[2][1] = v[0][1] * v[2][0] - v[0][0] * v[2][1], R.v[2][2] = v[0][0] * v[1][1] - v[0][1] * v[1][0];
		double m = 1.0 / (v[0][0] * R.v[0][0] + v[0][1] * R.v[1][0] + v[0][2] * R.v[2][0]);
		R.v[0][0] *= m, R.v[0][1] *= m, R.v[0][2] *= m, R.v[1][0] *= m, R.v[1][1] *= m, R.v[1][2] *= m, R.v[2][0] *= m, R.v[2][1] *= m, R.v[2][2] *= m;
		return R;
	}
	friend mat3 inverse(const mat3 &A) { return A.inverse(); }

};

mat3 tensor(vec3 u, vec3 v) { return mat3(u*v.x, u*v.y, u*v.z); }
mat3 axis_angle(vec3 n, double a) {
	n = normalize(n); double ct = cos(a), st = sin(a);
	return mat3(
		ct + n.x*n.x*(1 - ct), n.y*n.x*(1 - ct) + n.z*st, n.z*n.x*(1 - ct) - n.y*st,
		n.x*n.y*(1 - ct) - n.z*st, ct + n.y*n.y*(1 - ct), n.z*n.y*(1 - ct) + n.x*st,
		n.x*n.z*(1 - ct) + n.y*st, n.y*n.z*(1 - ct) - n.x*st, ct + n.z*n.z*(1 - ct)
	);
}
mat3 rotationMatrix_x(double a) {
	double ca = cos(a), sa = sin(a);
	return mat3(
		1, 0, 0,
		0, ca, sa,
		0, -sa, ca
	);
}
mat3 rotationMatrix_y(double a) {
	double ca = cos(a), sa = sin(a);
	return mat3(
		ca, 0, -sa,
		0, 1, 0,
		sa, 0, ca
	);
}
mat3 rotationMatrix_z(double a) {
	double ca = cos(a), sa = sin(a);
	return mat3(
		ca, sa, 0,
		-sa, ca, 0,
		0, 0, 1
	);
}

// mostly used for debug
double degree(double rad) {
	return rad * (180. / PI);
}
double radians(double deg) {
	return deg * (PI / 180.);
}



class mat3f {
public:
	float v[3][3];
	mat3f() {}
	explicit mat3f(float k) {  // diagonal matrix
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			v[i][j] = k * (i == j);
	}
	explicit mat3f(vec3f lambda) {  // diagonal matrix
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			v[i][j] = i == j ? ((float*)&lambda)[i] : 0;
	}
	explicit mat3f(vec3f i, vec3f j, vec3f k) {  // matrix by column vectors
		for (int u = 0; u < 3; u++) v[u][0] = ((float*)&i)[u], v[u][1] = ((float*)&j)[u], v[u][2] = ((float*)&k)[u];
	}
	explicit mat3f(float _00, float _10, float _20, float _01, float _11, float _21, float _02, float _12, float _22) {  // ordered in column-wise
		v[0][0] = _00, v[0][1] = _01, v[0][2] = _02, v[1][0] = _10, v[1][1] = _11, v[1][2] = _12, v[2][0] = _20, v[2][1] = _21, v[2][2] = _22;
	}
	float* operator[] (int d) { return &v[d][0]; }
	const float* operator[] (int d) const { return &v[d][0]; }
	void operator += (const mat3f &m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] += (&m.v[0][0])[i]; }
	void operator -= (const mat3f &m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] -= (&m.v[0][0])[i]; }
	void operator *= (float m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] *= m; }
	mat3f operator + (const mat3f &m) const { mat3f r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] + (&m.v[0][0])[i]; return r; }
	mat3f operator - (const mat3f &m) const { mat3f r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] - (&m.v[0][0])[i]; return r; }
	mat3f operator * (float m) const { mat3f r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] * m; return r; }
	friend mat3f operator * (float a, const mat3f &m) { mat3f r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = a * (&m.v[0][0])[i]; return r; }
	friend float determinant(const mat3f &m) { return m.v[0][0] * (m.v[1][1] * m.v[2][2] - m.v[1][2] * m.v[2][1]) - m.v[0][1] * (m.v[1][0] * m.v[2][2] - m.v[1][2] * m.v[2][0]) + m.v[0][2] * (m.v[1][0] * m.v[2][1] - m.v[1][1] * m.v[2][0]); }
	mat3f transpose() const { mat3f r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.v[i][j] = v[j][i]; return r; }
	friend mat3f transpose(const mat3f &m) { return mat3f(m.v[0][0], m.v[0][1], m.v[0][2], m.v[1][0], m.v[1][1], m.v[1][2], m.v[2][0], m.v[2][1], m.v[2][2]); }
	friend double trace(const mat3f &m) { return m.v[0][0] + m.v[1][1] + m.v[2][2]; }

	mat3f operator * (const mat3f &A) const {
		mat3f R;
		for (int m = 0; m < 3; m++) for (int n = 0; n < 3; n++) {
			R.v[m][n] = 0;
			for (int i = 0; i < 3; i++) R.v[m][n] += v[m][i] * A.v[i][n];
		}
		return R;
	}
	vec3f operator * (const vec3f &a) const {
		return vec3f(
			v[0][0] * a.x + v[0][1] * a.y + v[0][2] * a.z,
			v[1][0] * a.x + v[1][1] * a.y + v[1][2] * a.z,
			v[2][0] * a.x + v[2][1] * a.y + v[2][2] * a.z);
	}
};



struct vec4f {
	float x, y, z, w;
	vec4f() {}
	explicit vec4f(float a) :x(a), y(a), z(a), w(a) {}
	explicit vec4f(float x, float y, float z, float w) :x(x), y(y), z(z), w(w) {}
	explicit vec4f(vec3f p, float w = 0.0f) : x(p.x), y(p.y), z(p.z), w(w) {}
};



// recommended over pMax and pMin
vec2 max(vec2 a, vec2 b) { return vec2(max(a.x, b.x), max(a.y, b.y)); }
vec2f max(vec2f a, vec2f b) { return vec2f(max(a.x, b.x), max(a.y, b.y)); }
ivec2 max(ivec2 a, ivec2 b) { return ivec2(max(a.x, b.x), max(a.y, b.y)); }
vec2 min(vec2 a, vec2 b) { return vec2(min(a.x, b.x), min(a.y, b.y)); }
vec2f min(vec2f a, vec2f b) { return vec2f(min(a.x, b.x), min(a.y, b.y)); }
ivec2 min(ivec2 a, ivec2 b) { return ivec2(min(a.x, b.x), min(a.y, b.y)); }
vec3 max(vec3 a, vec3 b) { return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
vec3f max(vec3f a, vec3f b) { return vec3f(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
ivec3 max(ivec3 a, ivec3 b) { return ivec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
vec3 min(vec3 a, vec3 b) { return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
vec3f min(vec3f a, vec3f b) { return vec3f(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
ivec3 min(ivec3 a, ivec3 b) { return ivec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
vec2 clamp(vec2 p, double a, double b) { return vec2(clamp(p.x, a, b), clamp(p.y, a, b)); }
vec2f clamp(vec2f p, float a, float b) { return vec2f(clamp(p.x, a, b), clamp(p.y, a, b)); }
vec3 clamp(vec3 p, double a, double b) { return vec3(clamp(p.x, a, b), clamp(p.y, a, b), clamp(p.z, a, b)); }
vec3f clamp(vec3f p, float a, float b) { return vec3f(clamp(p.x, a, b), clamp(p.y, a, b), clamp(p.z, a, b)); }




// triangle classes
#include <initializer_list>
struct triangle_3d {
	vec3 V[3];
	vec3& operator[](int d) { return V[d]; }
	const vec3& operator[](int d) const { return V[d]; }
	triangle_3d() {}
	triangle_3d(vec3 v0, vec3 v1, vec3 v2) { V[0] = v0, V[1] = v1, V[2] = v2; }
	triangle_3d(std::initializer_list<vec3> s) { V[0] = *s.begin(), V[1] = *(s.begin() + 1), V[2] = *(s.begin() + 2); }
	vec3 center() const { return (V[0] + V[1] + V[2])*(1. / 3.); }
	vec3 normal() const { return cross(V[1] - V[0], V[2] - V[0]); }  // ccw
	vec3 unit_normal() const { return ncross(V[1] - V[0], V[2] - V[0]); }
	void translate(vec3 d) { V[0] += d, V[1] += d, V[2] += d; }
	void scale(double s) { V[0] *= s, V[1] *= s, V[2] *= s; }
	void applyMatrix(mat3 M) { V[0] = M * V[0], V[1] = M * V[1], V[2] = M * V[2]; }
	double area() const { return 0.5*length(cross(V[1] - V[0], V[2] - V[0])); }
};
struct triangle_3d_f {
	vec3f v[3];
	vec3f& operator[](int d) { return v[d]; }
	const vec3f& operator[](int d) const { return v[d]; }
	triangle_3d_f() {}
	triangle_3d_f(vec3f v0, vec3f v1, vec3f v2) { v[0] = v0, v[1] = v1, v[2] = v2; }
	triangle_3d_f(std::initializer_list<vec3f> s) { v[0] = *s.begin(), v[1] = *(s.begin() + 1), v[2] = *(s.begin() + 2); }
	vec3f min_p() const { return min(min(v[0], v[1]), v[2]); }
	vec3f max_p() const { return max(max(v[0], v[1]), v[2]); }
	float min_z() const { return min(min(v[0].z, v[1].z), v[2].z); }
	float max_z() const { return max(max(v[0].z, v[1].z), v[2].z); }
};
struct triangle_2d {
	vec2 v[3];
	vec2& operator[](int d) { return v[d]; }
	const vec2& operator[](int d) const { return v[d]; }
	triangle_2d() {}
	triangle_2d(vec2 v0, vec2 v1, vec2 v2) { v[0] = v0, v[1] = v1, v[2] = v2; }
	triangle_2d(std::initializer_list<vec2> s) { v[0] = *s.begin(), v[1] = *(s.begin() + 1), v[2] = *(s.begin() + 2); }
	vec2 center() const { return (v[0] + v[1] + v[2])*(1. / 3.); }
	void translate(vec2 d) { v[0] += d, v[1] += d, v[2] += d; }
	void scale(double s) { v[0] *= s, v[1] *= s, v[2] *= s; }
	double area() const { return 0.5*det(v[1] - v[0], v[2] - v[0]); }  // signed
};

struct segment_2d_f {
	vec2f v[2];
	vec2f& operator[](int d) { return v[d]; }
	const vec2f& operator[](int d) const { return v[d]; }
	segment_2d_f() {}
	segment_2d_f(vec2f v0, vec2f v1) { v[0] = v0, v[1] = v1; }
	segment_2d_f(std::initializer_list<vec2f> s) { v[0] = *s.begin(), v[1] = *(s.begin() + 1); }

	float len() const { return length(v[1] - v[0]); }

	// distance to the line segment
	float dist2(vec2f p) const {
		vec2f q = p - v[0], d = v[1] - v[0];
		float h = dot(q, d) / dot(d, d);
		return (q - d * clamp(h, 0.0f, 1.0f)).sqr();
	}
	float dist(vec2f p) const {
		return sqrt(dist2(p));
	}
};


#endif // __INC_GEOMETRY_H

