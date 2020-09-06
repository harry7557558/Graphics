// try to find the rules about the moment of inertia of elliptical rings
// I may visualize using an interactive 3d plot if possible

#include "numerical/integration.h"
#include "numerical/geometry.h"
#include <stdio.h>
using namespace std;

// calculate the perimeter of an ellipse
double ellipse_perimeter(double a, double b) {
	double a2 = a * a, b2 = b * b;
	return NIntegral<double>([&](double t) {
		double ct = cos(t), st = sin(t);
		return sqrt(a2 * st*st + b2 * ct*ct);
	}, 0, 2 * PI, 1000);
}

// calculate the ratio of the moment of inertia of an elliptical ring to its mass
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
