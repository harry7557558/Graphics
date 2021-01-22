// test this header
#include "octatree.h"



std::vector<triangle_3d> T;

void test_marching() {

	auto fun = [](vec3 p) {
		double x = p.x, y = p.y, z = p.z;
		//return z;
		//return p.sqr() - 1.;
		//return length(vec2(length(p.xy()) - 1., p.z)) - .5;
		return length(vec2(length(p.xy()*p.xy()) - 1., p.z)) - .5;
	};

	T = ScalarFieldTriangulator_octatree::marching_cube(fun, vec3(-2), vec3(2), ivec3(48));
	//T = ScalarFieldTriangulator_octatree::marching_cube_cylindrical(fun, 2., -2., 2., 32, 64, 32);
	//T = ScalarFieldTriangulator_octatree::marching_cube_cylindrical_x(fun, 2., -2., 2., 32, 64, 32);
	//T = ScalarFieldTriangulator_octatree::marching_cube_cylindrical_y(fun, 2., -2., 2., 32, 64, 32);

}

void test_octatree() {

	auto fun = [](vec3 p) {
		double x = p.x, y = p.y, z = p.z;
		//return length(vec2(length(p.xy()*p.xy()) - 1., p.z)) - .5;
		//return 2 * y*(y*y - 3 * x*x)*(1 - z * z) + (x*x + y * y)*(x*x + y * y) - (9 * z*z - 1)*(1 - z * z);
		//return pow(x*x + 2.*y*y + z*z, 3.) - (9.*x*x + y*y)*z*z*z - 0.5;
		//return 4.0*pow(x*x + 2.*y*y + z * z - 1., 2.) - z * (5.*x*x*x*x - 10.*x*x*z*z + z * z*z*z) - 1.;
		//return exp(10 * (4 * p.xy().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.xz().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.yz().sqr() - pow(p.sqr() + 0.96, 2))) - 1.;
		return max(p.z + sin(10.*atan2(p.y, p.x)) * p.xy().sqr(), length(p) - 1.) + 0.1*sin(10.*x)*sin(10.*y)*sin(10.*z) + 0.2*sin(5.*x)*cos(y);
		//return length(vec2(length(p.xy()) - (.6 + .1*asin(.95*sin(10.*atan2(p.y, p.x) + 8.*p.z))), p.z) - vec2(0.5, 0)) - .5;
		//return exp(20.*(abs(abs(length(vec2(length(p.xy()) - 1., p.z)) - .5) - 0.2) - 0.1)) + exp(20.*(p.z - 0.2 - 0.1*p.xy().sqr())) - 1. + sin(10.*p.x) + sin(10.*p.y);
	};

	//T = ScalarFieldTriangulator_octatree::octatree(fun, vec3(-2), vec3(2), ivec3(12), 3);
	T = ScalarFieldTriangulator_octatree::octatree_cylindrical(fun, 2., -2., 2., 8, 16, 8, 3);
	//T = ScalarFieldTriangulator_octatree::octatree_cylindrical_x(fun, 2., -2., 2., 8, 16, 8, 3);
	//T = ScalarFieldTriangulator_octatree::octatree_cylindrical_y(fun, 2., -2., 2., 8, 16, 8, 3);

}


#include <chrono>
#include "ui/stl_encoder.h"

int main(int argc, char* argv[]) {

	auto time_start = std::chrono::high_resolution_clock::now();

	//test_marching();
	test_octatree();

	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time_start).count();
	printf("%.1lfms\n", 1000.*time_elapsed);

	writeSTL(argv[1], &T[0], T.size(), "", STL_CCW);
	return 0;
}
