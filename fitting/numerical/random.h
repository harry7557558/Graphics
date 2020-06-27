
#ifndef PI
#define PI 3.1415926535897932384626
#endif

// approximation of inverse error function
double erfinv(double x) {
	double n = log(1 - x * x), t = 0.5 * n + 2 / (PI*0.147);
	return (x < 0 ? -1 : 1) * sqrt(-t + sqrt(t*t - n / 0.147));
}

// fast random number from Numerical Recipes
// use to generate floatpoint random number
unsigned int _IDUM = 0;
unsigned randu() { return _IDUM = _IDUM * 1664525 + 1013904223; }

double randf(double a, double b) { return a + (randu() / 4294967296.)*(b - a); }  // uniform distribution in [a,b)
vec2 randv(double r) { double m = randf(0, r), a = randf(0, 2.*PI); return vec2(m*cos(a), m*sin(a)); }  // default distribution in |v|<r
vec2 randv_u(double r) { double m = sqrt(randf(0, r*r)), a = randf(0, 2.*PI); return vec2(m*cos(a), m*sin(a)); }  // uniform distribution in |v|<r
double randf_n(double a) { return sqrt(2.) * a * erfinv(2. * randf(0., 1.) - 1.); }  // normal distribution by standard deviation
vec2 randv_n(double a) { return vec2(randf_n(a), randf_n(a)); }  // normal distribution by standard deviation


