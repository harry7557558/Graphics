// test this header
#include "octatree.h"



#include "ui/stl_encoder.h"
std::vector<stl_triangle> Trigs;


void test_marching() {

	auto fun = [](vec3 p) {
		double x = p.x, y = p.y, z = p.z;
		//return z;
		//return p.sqr() - 1.;
		//return length(vec2(length(p.xy()) - 1., p.z)) - .5;
		return length(vec2(length(p.xy()*p.xy()) - 1., p.z)) - .5;
	};

	std::vector<triangle_3d> T;
	T = ScalarFieldTriangulator_octatree::marching_cube(fun, vec3(-2), vec3(2), ivec3(48));
	//T = ScalarFieldTriangulator_octatree::marching_cube_cylindrical(fun, 2., -2., 2., 32, 64, 32);
	//T = ScalarFieldTriangulator_octatree::marching_cube_cylindrical_x(fun, 2., -2., 2., 32, 64, 32);
	//T = ScalarFieldTriangulator_octatree::marching_cube_cylindrical_y(fun, 2., -2., 2., 32, 64, 32);

	convertTriangles(Trigs, &T[0], T.size());
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

	std::vector<triangle_3d> T;
	//T = ScalarFieldTriangulator_octatree::octatree(fun, vec3(-2), vec3(2), ivec3(12), 3);
	T = ScalarFieldTriangulator_octatree::octatree_cylindrical(fun, 2., -2., 2., 8, 16, 8, 3);
	//T = ScalarFieldTriangulator_octatree::octatree_cylindrical_x(fun, 2., -2., 2., 8, 16, 8, 3);
	//T = ScalarFieldTriangulator_octatree::octatree_cylindrical_y(fun, 2., -2., 2., 8, 16, 8, 3);

	convertTriangles(Trigs, &T[0], T.size());
}



void test_octatree_grad() {

	// distance field
	auto fun = [](vec3 p) {
		double x = p.x, y = p.y, z = p.z;
		auto s_min = [](double a, double b, double k) {
			double h = 0.5 + 0.5*(b - a) / k;
			if (h < 0.0) return b; if (h > 1.0) return a;
			return mix(b,a,h) - k*h*(1.0 - h);
		};

		//return length(p) - 1.0;
		//return pow((p*p).sqr(), 1. / 4.) - 1.0;
		//return length(vec2(length(p.xy()) - 1.2, p.z)) - 0.5;
		//return length(vec2(sqrt(length(p.xy()*p.xy())) - 1.2, p.z)) - 0.5;
		//return max(max(abs(x), abs(y)), abs(z)) - 1.;
		//return (abs(x) + abs(y) + abs(z) - 1.) / sqrt(3.);
		//vec3 r = vec3(1.618, 1, 1); double a = length(p / r), b = length(p / (r*r)); return a * (a - 1.) / b;
		//vec3 r = vec3(1.618, 1, 1); double a = length(p / r), b = length(p / (r*r)); return a * (a - 1.) / b + 0.5;
		vec3 r = vec3(1.618, 1, 1); double a = length(p / r), b = length(p / (r*r)); return a * (a - 1.) / b + 0.8;
		//return s_min(length(p - vec3(0.5, 0, 0)), length(p + vec3(0.5, 0, 0)), 0.5) - 0.5;
		//return s_min(length(p - vec3(0, 0, 0.5)) - 0.5, length(p + vec3(0, 0, 0.5)) - 1.0, 1.0);
		//return z - 0.2*sin(2.*x)*sin(2.*y);
		//return z - 0.1*sin(10.*length(p.xy())) / length(p.xy());
		//return length(p) - 1.0 + 0.1*sin(10.*x)*sin(10.*y)*sin(10.*z);
		//p = rotationMatrix_z(p.z)*p; return length(vec2(length(p.xz()) - 1.2, p.y)) - 0.5;
	};

	auto T = ScalarFieldTriangulator_octatree::octatree_with_grad(fun, vec3(-2), vec3(2), ivec3(12), 3);

#include "UI/colors/ColorFunctions.h"

	for (int i = 0; i < (int)T.size(); i++) {
		//vec3 col = vec3(0.5) + 0.5*T[i].grad;
		vec3 col = ColorFunctions::TemperatureMap(0.5*(tanh(10.*(1. - exp2(-length(T[i].grad)) - 0.5)) + 1.));
		Trigs.push_back(stl_triangle(T[i].trig, col));
	}

}


#include <chrono>

int main(int argc, char* argv[]) {

	auto time_start = std::chrono::high_resolution_clock::now();

	//test_marching();
	//test_octatree();
	test_octatree_grad();

	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - time_start).count();
	printf("%.1lfms\n", 1000.*time_elapsed);

	writeSTL(argv[1], &Trigs[0], Trigs.size(), "", STL_CCW);
	return 0;
}
