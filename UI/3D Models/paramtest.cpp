// parametric surface test

#include <vector>
#include <stdio.h>

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"

// temporary function
// vec3 F(double u, double v)
template<typename Fun>
int param2trigs(Fun F, std::vector<triangle> &p, double u0, double u1, double v0, double v1, int uD, int vD) {
	double du = (u1 - u0) / uD, dv = (v1 - v0) / vD;
	for (int ui = 0; ui < uD; ui++) {
		for (int vi = 0; vi < vD; vi++) {
			double u = u0 + ui * du, v = v0 + vi * dv;
			vec3 p00 = F(u, v);
			vec3 p01 = F(u, v + dv);
			vec3 p10 = F(u + du, v);
			vec3 p11 = F(u + du, v + dv);
			p.push_back(triangle{ p10, p00, p01 });
			p.push_back(triangle{ p01, p11, p10 });
		}
	}
	return 2 * uD * vD;
}



std::vector<triangle> comps;

// based on Archimedean spiral
auto snail = [](double u, double v,
	double size, double vert, double offset, double layer_n) {

	// 0 < u < 2π, 0 < v < 1
	// For a right-handed snail, all of the following are positive:
	//  - size: scaling of the snail
	//  - vert: height/sharpness, z_vertice=size*vert
	//  - offset: offset of each layer from z-axis, greater than 1 for an "actual" snail
	//  - layer_n: number of layers of the snail

	return size * ((1. - v)*vec3(
		(cos(u) + offset)*cossin(2.*PI*layer_n*v),
		sin(u)) + vec3(0, 0, vert*v));
};

void generateSnail(vec3 pos, double size, double vert, double offset, double layer_n) {
	// u length: 2π*size
	// v length: 1/2*sqrt(4π²n²+1)+1/4πn*asinh(2πn)  *size in xOy
	// v length is approximately 0.5*layer_n times of u length
	param2trigs([&](double u, double v) {
		return snail(u, v, size, vert, offset, layer_n) + pos;
	}, comps, 0, 2.*PI, 0, 1, 40, clamp(int(40.*abs(layer_n)), 10, 400));
}


void model() {
	// Snail
	{
		//snail(0.18, 1.2, 1, 3);	// land snail
		//snail(0.18, 2.2, 1, 4.5);	// river snail, I don't know why it looks like shit
		//snail(0.22, 1.1, 0.5, 2);	// giant river snail
		//snail(0.2, 1.8, 0.2, 2.4);	// land cone snail
		//snail(0.2, 3.5, 0.2, 8);	// cone snail
	}
}



int main() {
	model();
	FILE* fp = fopen("D:\\test.stl", "wb");
	for (int i = -4; i <= 4; i++)
		generateSnail(vec3(i, 0, 0), 0.25, i, 1, i);
	writeSTL(fp, &comps[0], comps.size(), "bac");
	fclose(fp);
	return 0;
}

