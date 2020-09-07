// Try to find the rules of the moments of inertia of elliptical rings with uniform line density
// An elliptical ring has a major radius a and a minor radius b

// See if the ratio between the moment of inertia and the mass of elliptical rings can be expressed analytically in terms of a and b
// Seems like the answer isn't in elementary functions...

#include "numerical/integration.h"
#include "numerical/geometry.h"
#include <stdio.h>

// calculate the perimeter of an ellipse
double ellipse_perimeter(double a, double b) {
	double a2 = a * a, b2 = b * b;
	return NIntegral<double>([&](double t) {
		double ct = cos(t), st = sin(t);
		return sqrt(a2 * st*st + b2 * ct*ct);
	}, 0, 2 * PI, 1000);
}

// calculate the ratio of the moment of inertia of an elliptical ring to its mass
// Interactive graph: https://www.desmos.com/calculator/raeuahskxm
double ellipse_inertia(double a, double b) {
	double a2 = a * a, b2 = b * b;
	vec2 mi = NIntegral<vec2>([&](double t) {
		double ct = cos(t), st = sin(t);
		ct *= ct, st *= st;
		double dl = sqrt(a2 * st + b2 * ct);
		double dI = a2 * ct + b2 * st;
		return vec2(dl, dl * dI);
	}, 0, 2 * PI, 1000);
	return mi.y / mi.x;
}

int main() {
	for (double a = 1; a <= 10; a += 1) {
		for (double b = 1; b <= a; b += 1) {
			printf("%lg %lg \t", a, b);
			//printf("%.8lf\n", ellipse_perimeter(a, b) / PI);
			printf("%.8lf\n", ellipse_inertia(a, b));
		}
	}
	return 0;
}
