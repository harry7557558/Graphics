
#include <cmath>

#ifndef PI
#define PI 3.1415926535897932384626
#endif

// Simpson's method
template<typename T, typename Fun>
T NIntegral(Fun f, double a, double b, int n) {
	n *= 2;
	double u = (b - a) / n;
	T s(0);
	for (int i = 1; i < n; i += 2) s += f(a + u * i);
	s *= 2;
	for (int i = 2; i < n; i += 2) s += f(a + u * i);
	s = 2 * s + f(a) + f(b);
	return s * (u / 3);
}

