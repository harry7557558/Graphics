// define _RANDOM_H_BETTER_QUALITY before including this file for better random number quality


#ifndef __INC_RANDOM_H

#define __INC_RANDOM_H


#ifndef __INC_GEOMETRY_H
#include "geometry.h"
#endif



// hash function
unsigned hashu(unsigned x) {
	x = ((x >> 16) ^ x) * 0x45d9f3bu;
	x = ((x >> 16) ^ x) * 0x45d9f3bu;
	return x = (x >> 16) ^ x;
}
double hashf(double x, double y) {
	return fmod(sin(12.9898*x + 78.233*y + 1.) * 43758.5453, 1.);  // a hash function for GLSL; [-1,1] in C++
};



// random number generators
// use to generate floatpoint random number

#ifndef _RANDOM_H_BETTER_QUALITY

// from Numerical Recipes
unsigned int _IDUM = 0;
unsigned randu() { return _IDUM = _IDUM * 1664525 + 1013904223; }
void _SRAND(unsigned i) { _IDUM = hashu(i); }

#else

// a better random number generator
static unsigned _SEED_X = 123456789, _SEED_Y = 362436069, _SEED_Z = 521288629;
unsigned randu() {
	_SEED_X ^= _SEED_X << 16; _SEED_X ^= _SEED_X >> 5; _SEED_X ^= _SEED_X << 1;
	unsigned t = _SEED_X; _SEED_X = _SEED_Y; _SEED_Y = _SEED_Z; _SEED_Z = t ^ _SEED_X ^ _SEED_Y;
	return _SEED_Z;
}
void _SRAND(unsigned i) { _SEED_X = hashu(i); _SEED_Y = hashu(i + 1); _SEED_Z = hashu(i + 2); }

#endif



// float-point random number generators

#ifndef PI
#define PI 3.1415926535897932384626
#endif

// approximation of inverse error function, use to generate normal distribution
double erfinv(double x) {
	double n = log(1 - x * x), t = 0.5 * n + 2 / (PI*0.147);
	return (x < 0 ? -1 : 1) * sqrt(-t + sqrt(t*t - n / 0.147));
}

double randf(double a, double b) { return a + (randu() / 4294967296.)*(b - a); }  // uniform distribution in [a,b)
double randf_n(double a) { return sqrt(2.) * a * erfinv(2. * randf(0., 1.) - 1.); }  // normal distribution by standard deviation
int randi(int a, int b) { return int(randf(a, b)); }  // uniform pseudorandom integer in [a,b)
vec2 rand2() { double a = randf(0, 2.*PI); return vec2(cos(a), sin(a)); }  // uniform distributed unit vector
vec2 rand2(double r) { double m = randf(0, r), a = randf(0, 2.*PI); return vec2(m*cos(a), m*sin(a)); }  // default distribution in |v|<r
vec2 rand2_u(double r) { double m = sqrt(randf(0, r*r)), a = randf(0, 2.*PI); return vec2(m*cos(a), m*sin(a)); }  // uniform distribution in |v|<r
vec2 rand2_n(double a) { return vec2(randf_n(a), randf_n(a)); }  // normal distribution by standard deviation
vec3 rand3() { double u = randf(0, 2.*PI), v = randf(-1, 1); return vec3(vec2(cos(u), sin(u))*sqrt(1 - v * v), v); }  // uniform distributed unit vector
vec3 rand3(double r) { return rand3()*randf(0, r); }  // default distribution in |v|<r
vec3 rand3_u(double r) { double m = r * pow(randf(0, 1), 1. / 3.), u = randf(0, 2.*PI), v = randf(-1, 1); return m * vec3(vec2(cos(u), sin(u))*sqrt(1 - v * v), v); }  // uniform distribution in |v|<r
vec3 rand3_n(double a) { return vec3(randf_n(a), randf_n(a), randf_n(a)); }  // normal distribution by standard deviation
vec3 rand3_c() { double u = randf(0, 2 * PI), v = randf(0, 1); return vec3(sqrt(v)*vec2(cos(u), sin(u)), sqrt(1 - v)); }  // cosine-weighted random hemisphere

vec2 rand2_f(double a, double b) { return vec2(randf(a, b), randf(a, b)); }
vec3 rand3_f(double a, double b) { return vec3(randf(a, b), randf(a, b), randf(a, b)); }

mat3 randRotation() {
	return axis_angle(rand3(), randf(0, 2 * PI));
}

// from Numerical Recipes, not accurate for small xm
double poisson(double xm) {
	while (1) {
		double em, y;
		double sq = sqrt(2.*xm), alxm = log(xm), g = xm * alxm - lgamma(xm + 1.);
		do {
			y = tan(randf(0, PI));
			em = sq * y + xm;
		} while (em < 0.);
		double t = 0.9*(1. + y * y)*exp(em*alxm - lgamma(em + 1.) - g);
		if (randf(0, 1) < t) return em;
	}
}


#endif // __INC_RANDOM_H

