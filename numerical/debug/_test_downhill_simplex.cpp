#include "ui/stl_encoder.h"
std::vector<stl_triangle> STL;
#include "numerical/optimization.h"

void test2d() {
	double PS[3][2] = {
		{-10, -10},
		{-10, 10},
		{10, -10}
	};
	double *P[3] = { PS[0], PS[1], PS[2] };
	double val[3];
	int d = downhillSimplex(2, [](double p[]) {
		double x = p[0], y = p[1];
		return sin(x) + sin(y) + 0.1*(x*x + y * y);
		//return pow(100.*pow(y - x * x, 2.) + pow(1 - y, 2.), 0.2);
	}, P, val, false, 1e-6);

	vec2 u = downhillSimplex_2d([](vec2 p) {
		double x = p.x, y = p.y;
		return sin(x) + sin(y) + 0.1*(x*x + y * y);
	}, (vec2*)&PS[0][0], 1e-8);
	printf("(%lf,%lf,%lf)\n", u.x, u.y, sin(u.x) + sin(u.y));

	printf("\n");
	for (int i = 0; i < 2; i++) printf("%lf ", P[d][i]);
	printf(" %lf\n", val[d]);
}

void test3d() {
	double PS[4][3] = { {-10, -10, -10} };
	double *P[4] = { PS[0], PS[1], PS[2], PS[3] };
	//setupInitialSimplex_axesAligned(3, P[0], P, 20);
	setupInitialSimplex_regular(3, P[0], P, 5);
	double val[4];

	int d = downhillSimplex(3, [](double p[]) {
		double x = p[0], y = p[1], z = p[2];
		//return pow(x * x + y * y + z * z, 0.6);
		//return sin(x)*sin(y)*sin(z) + 0.1*(x*x + y * y + z * z);
		//return 100.*pow(x*x - y * y - z, 2.) + vec3(x - 2, y - 1, z - 3).sqr();
		//return log(100.*pow(x*x + y * y + z * z - 1., 2.) - (x + y + z) + 1.8);
		return 999.*(pow(exp2(x) + y * y - z - 1., 2.) + pow(x*x + y * y + z * z - 2., 2.)) + vec3(x - 1, y, z - 1).sqr();
	}, P, val, false, 1e-6);

	printf("\n");
	for (int i = 0; i < 3; i++) printf("%lf ", P[d][i]);
	printf(" %lf\n", val[d]);
}

int main() {
	//STL.resize(2 * 400 * 400);
	//stl_fun2trigs([](double x, double y) { return sin(x) + sin(y) + 0.1*(x*x + y * y); }, &STL[0], -40, 40, -40, 40, 400, 400, -10, 10);
	//test2d();
	test3d();
	writeSTL("D:\\t.stl", &STL[0], STL.size(), nullptr, "bac");
}

