
#include "numerical/geometry.h"
#include <vector>
#include <stdio.h>

double fun(vec3 p) {
	//return p.sqr() - 1.;
	//return p.x*p.x + p.y*p.y / 2.6 + p.z*p.z - 1;
	//return exp(4 * (p.sqr() - 2)) + exp(4 * (0.5 - p.x*p.x - p.y*p.y - p.z*p.z / 4)) - 1;
	//return p.x*p.x*p.x*p.x + p.y*p.y*p.y*p.y + p.z*p.z*p.z*p.z - 1;
	//return pow(p.x, 8) + pow(p.y, 8) + pow(p.z, 8) - 1;
	//return p.x*p.x*p.x*p.x + p.y*p.y*p.y*p.y + p.z*p.z*p.z*p.z - p.x*p.x - p.y*p.y;
	//return p.x*p.x*p.x*p.x + p.y*p.y*p.y*p.y + p.z*p.z*p.z*p.z - p.x*p.x - p.y*p.y + 0.1;
	//return pow(p.sqr() + 0.8, 2) - 4 * p.xy().sqr();
	//return pow(4.*p.sqr() + 3.5, 2) - 60. * p.xy().sqr();
	//return p.sqr()*p.sqr() - p.xy().sqr();
	//return 1.2 / length(p - vec3(1, 0, 0)) + 0.8 / length(p + vec3(1, 0, 0)) - 1.8;
	//return exp(10 * (4 * p.xy().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.xz().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.yz().sqr() - pow(p.sqr() + 0.96, 2))) - 1.;
	//return pow(pow(p.x, 6) + pow(p.y, 6) + pow(p.z, 6) + 3.5, 2) - 16 * (pow(p.x, 6) + pow(p.y, 6));
	//return exp(1 - (p.x - 1)*(p.x - 1) - p.y*p.y - p.z*p.z) + exp(1 - (p.x + 1)*(p.x + 1) - p.y*p.y - p.z*p.z) - 1;
	//return exp(4 * (1. - (p.x - 1)*(p.x - 1) - (p.y - 1)*(p.y - 1) - (p.z - 1)*(p.z - 1))) + exp(4 * (1. - (p.x + 1)*(p.x + 1) - (p.y - 1)*(p.y - 1) - (p.z - 1)*(p.z - 1))) \
		+ exp(4 * (1. - (p.x - 1)*(p.x - 1) - (p.y + 1)*(p.y + 1) - (p.z - 1)*(p.z - 1))) + exp(4 * (1. - (p.x + 1)*(p.x + 1) - (p.y + 1)*(p.y + 1) - (p.z - 1)*(p.z - 1))) \
		+ exp(4 * (1. - (p.x - 1)*(p.x - 1) - (p.y - 1)*(p.y - 1) - (p.z + 1)*(p.z + 1))) + exp(4 * (1. - (p.x + 1)*(p.x + 1) - (p.y - 1)*(p.y - 1) - (p.z + 1)*(p.z + 1))) \
		+ exp(4 * (1. - (p.x - 1)*(p.x - 1) - (p.y + 1)*(p.y + 1) - (p.z + 1)*(p.z + 1))) + exp(4 * (1. - (p.x + 1)*(p.x + 1) - (p.y + 1)*(p.y + 1) - (p.z + 1)*(p.z + 1))) - 1;
	//return exp(2 * (0.5 - p.x*p.x - p.y*p.y - 1.5*(p.z - 0.8)*(p.z - 0.8))) + exp(2 * (1.5 - p.x*p.x - p.y*p.y - 1.5*(p.z + 0.8)*(p.z + 0.8))) + 0.1*sin(20 * p.x)*sin(20 * p.x)*sin(20 * p.z) - 1;
	//return sin(3.*p.x)*sin(3.*p.y)*sin(3.*p.z) - 0.1;
	//return sin(3.*p.x)*cos(3.*p.y)*tan(3.*p.z) - 0.1;
	//return sin(3.*p.x) + sin(3.*p.y) + sin(3.*p.z) - 0.1;
	//return sin(3.*p.x) + cos(3.*p.y) + tan(3.*p.z) - 0.1;
	//return cos(10.*p.x) + cos(10.*p.y) + cos(10.*p.z) + 2.*p.z;
	//return max(max(abs(p.x), abs(p.y)) - 1., abs(p.z) - 0.618);
	//return round(abs(p.x)) + round(abs(p.y)) + round(abs(2.*p.z + .5*p.x)) - 2.;
	//return p.x*p.x + p.y*p.y + p.z*p.z*p.z - p.z*p.z;
	//return ((p.x - 1)*(p.x - 1) + p.y*p.y + p.z*p.z) * ((p.x + 1)*(p.x + 1) + p.y*p.y + p.z*p.z) * (p.x*p.x + (p.y - 1)*(p.y - 1) + p.z*p.z) * (p.x*p.x + (p.y + 1)*(p.y + 1) + p.z*p.z) - 1.1;
	//return ((p.x - 1)*(p.x - 1) + p.y*p.y + p.z*p.z) * ((p.x + 1)*(p.x + 1) + p.y*p.y + p.z*p.z) * (p.x*p.x + (p.y - 1)*(p.y - 1) + p.z*p.z) * (p.x*p.x + (p.y + 1)*(p.y + 1) + p.z*p.z) \
		* (p.x*p.x + p.y*p.y + (p.z - 1)*(p.z - 1)) * (p.x*p.x + p.y*p.y + (p.z + 1)*(p.z + 1)) - 1.5;
	//return 2 * p.y*(p.y*p.y - 3 * p.x*p.x)*(1 - p.z*p.z) + (p.x*p.x + p.y*p.y)*(p.x*p.x + p.y*p.y) - (9 * p.z*p.z - 1)*(1 - p.z*p.z);
	//return pow(p.x*p.x + 2.25*p.y*p.y + p.z*p.z - 1, 3.) - (p.x*p.x + 0.1125*p.y*p.y)*p.z*p.z*p.z;
	//return 4.0*pow(p.x*p.x + 2.*p.y*p.y + p.z*p.z - 1., 2.) - p.z*(5.*p.x*p.x*p.x*p.x - 10.*p.x*p.x*p.z*p.z + p.z*p.z*p.z*p.z) - 1.;
	return pow(p.x*p.x + 2.*p.y*p.y + p.z*p.z, 3.) - (9.*p.x*p.x + p.y*p.y)*p.z*p.z*p.z - 0.5;
	//return (p.x*p.x*p.y*p.y / p.z) / abs(abs((p.x*p.x / p.y) / abs(p.x*p.x / (p.y*p.y)))*p.x*p.x*p.y / (p.z*p.z)) - p.z;  // what a mess
}




void marching_cube(double(*fun)(vec3), std::vector<triangle_3d> &T, vec3 p0, vec3 p1, ivec3 dif) {
	vec3 dp = (p1 - p0) / vec3(dif);

	// sample between two "scan planes" instead of a 3d volume, intended to save memory
	double *xy0 = new double[(dif.x + 1)*(dif.y + 1)];
	double *xy1 = new double[(dif.x + 1)*(dif.y + 1)];
	auto grid = [&](double *xy, int i, int j)->double& {
		return xy[i*(dif.y + 1) + j];
	};

	// initialize the first plane
	for (int i = 0; i <= dif.x; i++) {
		for (int j = 0; j <= dif.y; j++) {
			vec3 p = p0 + vec3(i, j, 0)*dp;
			grid(xy1, i, j) = fun(p);
		}
	}

	// marching cube
	for (int k = 0; k < dif.z; k++) {

		// switch the planes
		std::swap(xy0, xy1);

		// initialize the next plane
		for (int i = 0; i <= dif.x; i++) {
			for (int j = 0; j <= dif.y; j++) {
				vec3 p = p0 + vec3(i, j, k + 1)*dp;
				grid(xy1, i, j) = fun(p);
			}
		}

		// check between planes
		for (int i = 0; i < dif.x; i++) {
			for (int j = 0; j < dif.y; j++) {

				// vertex/edge mapping tables
				const ivec3 vertexList[8] = {
					ivec3(0,0,0), ivec3(0,1,0), ivec3(1,1,0), ivec3(1,0,0),
					ivec3(0,0,1), ivec3(0,1,1), ivec3(1,1,1), ivec3(1,0,1)
				};
				const ivec2 edgeList[12] = {
					ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0),
					ivec2(4,5), ivec2(5,6), ivec2(6,7), ivec2(7,4),
					ivec2(0,4), ivec2(1,5), ivec2(2,6), ivec2(3,7)
				};

				// read values from the cube
				double cube[8] = {
					grid(xy0, i, j),
					grid(xy0, i, j + 1),
					grid(xy0, i + 1, j + 1),
					grid(xy0, i + 1, j),
					grid(xy1, i, j),
					grid(xy1, i, j + 1),
					grid(xy1, i + 1, j + 1),
					grid(xy1, i + 1, j)
				};

				// lookup tables
				// http://paulbourke.net/geometry/polygonise/
				const static int edgeTable[256] = {
					0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
					0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
					0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
					0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
					0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
					0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
					0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
					0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc , 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
					0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
					0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
					0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
					0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
					0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
					0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
					0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
					0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
				};
				const static int triTable[256][16] = {
					{-1}, {0,8,3,-1}, {0,1,9,-1}, {1,8,3,9,8,1,-1}, {1,2,10,-1}, {0,8,3,1,2,10,-1}, {9,2,10,0,2,9,-1}, {2,8,3,2,10,8,10,9,8,-1}, {3,11,2,-1}, {0,11,2,8,11,0,-1}, {1,9,0,2,3,11,-1}, {1,11,2,1,9,11,9,8,11,-1}, {3,10,1,11,10,3,-1}, {0,10,1,0,8,10,8,11,10,-1}, {3,9,0,3,11,9,11,10,9,-1}, {9,8,10,10,8,11,-1},
					{4,7,8,-1}, {4,3,0,7,3,4,-1}, {0,1,9,8,4,7,-1}, {4,1,9,4,7,1,7,3,1,-1}, {1,2,10,8,4,7,-1}, {3,4,7,3,0,4,1,2,10,-1}, {9,2,10,9,0,2,8,4,7,-1}, {2,10,9,2,9,7,2,7,3,7,9,4,-1}, {8,4,7,3,11,2,-1}, {11,4,7,11,2,4,2,0,4,-1}, {9,0,1,8,4,7,2,3,11,-1}, {4,7,11,9,4,11,9,11,2,9,2,1,-1}, {3,10,1,3,11,10,7,8,4,-1}, {1,11,10,1,4,11,1,0,4,7,11,4,-1}, {4,7,8,9,0,11,9,11,10,11,0,3,-1}, {4,7,11,4,11,9,9,11,10,-1},
					{9,5,4,-1}, {9,5,4,0,8,3,-1}, {0,5,4,1,5,0,-1}, {8,5,4,8,3,5,3,1,5,-1}, {1,2,10,9,5,4,-1}, {3,0,8,1,2,10,4,9,5,-1}, {5,2,10,5,4,2,4,0,2,-1}, {2,10,5,3,2,5,3,5,4,3,4,8,-1}, {9,5,4,2,3,11,-1}, {0,11,2,0,8,11,4,9,5,-1}, {0,5,4,0,1,5,2,3,11,-1}, {2,1,5,2,5,8,2,8,11,4,8,5,-1}, {10,3,11,10,1,3,9,5,4,-1}, {4,9,5,0,8,1,8,10,1,8,11,10,-1}, {5,4,0,5,0,11,5,11,10,11,0,3,-1}, {5,4,8,5,8,10,10,8,11,-1},
					{9,7,8,5,7,9,-1}, {9,3,0,9,5,3,5,7,3,-1}, {0,7,8,0,1,7,1,5,7,-1}, {1,5,3,3,5,7,-1}, {9,7,8,9,5,7,10,1,2,-1}, {10,1,2,9,5,0,5,3,0,5,7,3,-1}, {8,0,2,8,2,5,8,5,7,10,5,2,-1}, {2,10,5,2,5,3,3,5,7,-1}, {7,9,5,7,8,9,3,11,2,-1}, {9,5,7,9,7,2,9,2,0,2,7,11,-1}, {2,3,11,0,1,8,1,7,8,1,5,7,-1}, {11,2,1,11,1,7,7,1,5,-1}, {9,5,8,8,5,7,10,1,3,10,3,11,-1}, {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1}, {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1}, {11,10,5,7,11,5,-1},
					{10,6,5,-1}, {0,8,3,5,10,6,-1}, {9,0,1,5,10,6,-1}, {1,8,3,1,9,8,5,10,6,-1}, {1,6,5,2,6,1,-1}, {1,6,5,1,2,6,3,0,8,-1}, {9,6,5,9,0,6,0,2,6,-1}, {5,9,8,5,8,2,5,2,6,3,2,8,-1}, {2,3,11,10,6,5,-1}, {11,0,8,11,2,0,10,6,5,-1}, {0,1,9,2,3,11,5,10,6,-1}, {5,10,6,1,9,2,9,11,2,9,8,11,-1}, {6,3,11,6,5,3,5,1,3,-1}, {0,8,11,0,11,5,0,5,1,5,11,6,-1}, {3,11,6,0,3,6,0,6,5,0,5,9,-1}, {6,5,9,6,9,11,11,9,8,-1},
					{5,10,6,4,7,8,-1}, {4,3,0,4,7,3,6,5,10,-1}, {1,9,0,5,10,6,8,4,7,-1}, {10,6,5,1,9,7,1,7,3,7,9,4,-1}, {6,1,2,6,5,1,4,7,8,-1}, {1,2,5,5,2,6,3,0,4,3,4,7,-1}, {8,4,7,9,0,5,0,6,5,0,2,6,-1}, {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1}, {3,11,2,7,8,4,10,6,5,-1}, {5,10,6,4,7,2,4,2,0,2,7,11,-1}, {0,1,9,4,7,8,2,3,11,5,10,6,-1}, {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1}, {8,4,7,3,11,5,3,5,1,5,11,6,-1}, {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1}, {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1}, {6,5,9,6,9,11,4,7,9,7,11,9,-1},
					{10,4,9,6,4,10,-1}, {4,10,6,4,9,10,0,8,3,-1}, {10,0,1,10,6,0,6,4,0,-1}, {8,3,1,8,1,6,8,6,4,6,1,10,-1}, {1,4,9,1,2,4,2,6,4,-1}, {3,0,8,1,2,9,2,4,9,2,6,4,-1}, {0,2,4,4,2,6,-1}, {8,3,2,8,2,4,4,2,6,-1}, {10,4,9,10,6,4,11,2,3,-1}, {0,8,2,2,8,11,4,9,10,4,10,6,-1}, {3,11,2,0,1,6,0,6,4,6,1,10,-1}, {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1}, {9,6,4,9,3,6,9,1,3,11,6,3,-1}, {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1}, {3,11,6,3,6,0,0,6,4,-1}, {6,4,8,11,6,8,-1},
					{7,10,6,7,8,10,8,9,10,-1}, {0,7,3,0,10,7,0,9,10,6,7,10,-1}, {10,6,7,1,10,7,1,7,8,1,8,0,-1}, {10,6,7,10,7,1,1,7,3,-1}, {1,2,6,1,6,8,1,8,9,8,6,7,-1}, {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1}, {7,8,0,7,0,6,6,0,2,-1}, {7,3,2,6,7,2,-1}, {2,3,11,10,6,8,10,8,9,8,6,7,-1}, {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1}, {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11,-1}, {11,2,1,11,1,7,10,6,1,6,7,1,-1}, {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6,-1}, {0,9,1,11,6,7,-1}, {7,8,0,7,0,6,3,11,0,11,6,0,-1}, {7,11,6,-1},
					{7,6,11,-1}, {3,0,8,11,7,6,-1}, {0,1,9,11,7,6,-1}, {8,1,9,8,3,1,11,7,6,-1}, {10,1,2,6,11,7,-1}, {1,2,10,3,0,8,6,11,7,-1}, {2,9,0,2,10,9,6,11,7,-1}, {6,11,7,2,10,3,10,8,3,10,9,8,-1}, {7,2,3,6,2,7,-1}, {7,0,8,7,6,0,6,2,0,-1}, {2,7,6,2,3,7,0,1,9,-1}, {1,6,2,1,8,6,1,9,8,8,7,6,-1}, {10,7,6,10,1,7,1,3,7,-1}, {10,7,6,1,7,10,1,8,7,1,0,8,-1}, {0,3,7,0,7,10,0,10,9,6,10,7,-1}, {7,6,10,7,10,8,8,10,9,-1},
					{6,8,4,11,8,6,-1}, {3,6,11,3,0,6,0,4,6,-1}, {8,6,11,8,4,6,9,0,1,-1}, {9,4,6,9,6,3,9,3,1,11,3,6,-1}, {6,8,4,6,11,8,2,10,1,-1}, {1,2,10,3,0,11,0,6,11,0,4,6,-1}, {4,11,8,4,6,11,0,2,9,2,10,9,-1}, {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3,-1}, {8,2,3,8,4,2,4,6,2,-1}, {0,4,2,4,6,2,-1}, {1,9,0,2,3,4,2,4,6,4,3,8,-1}, {1,9,4,1,4,2,2,4,6,-1}, {8,1,3,8,6,1,8,4,6,6,10,1,-1}, {10,1,0,10,0,6,6,0,4,-1}, {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3,-1}, {10,9,4,6,10,4,-1},
					{4,9,5,7,6,11,-1}, {0,8,3,4,9,5,11,7,6,-1}, {5,0,1,5,4,0,7,6,11,-1}, {11,7,6,8,3,4,3,5,4,3,1,5,-1}, {9,5,4,10,1,2,7,6,11,-1}, {6,11,7,1,2,10,0,8,3,4,9,5,-1}, {7,6,11,5,4,10,4,2,10,4,0,2,-1}, {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6,-1}, {7,2,3,7,6,2,5,4,9,-1}, {9,5,4,0,8,6,0,6,2,6,8,7,-1}, {3,6,2,3,7,6,1,5,0,5,4,0,-1}, {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8,-1}, {9,5,4,10,1,6,1,7,6,1,3,7,-1}, {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1}, {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1}, {7,6,10,7,10,8,5,4,10,4,8,10,-1},
					{6,9,5,6,11,9,11,8,9,-1}, {3,6,11,0,6,3,0,5,6,0,9,5,-1}, {0,11,8,0,5,11,0,1,5,5,6,11,-1}, {6,11,3,6,3,5,5,3,1,-1}, {1,2,10,9,5,11,9,11,8,11,5,6,-1}, {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1}, {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1}, {6,11,3,6,3,5,2,10,3,10,5,3,-1}, {5,8,9,5,2,8,5,6,2,3,8,2,-1}, {9,5,6,9,6,0,0,6,2,-1}, {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1}, {1,5,6,2,1,6,-1}, {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1}, {10,1,0,10,0,6,9,5,0,5,6,0,-1}, {0,3,8,5,6,10,-1}, {10,5,6,-1},
					{11,5,10,7,5,11,-1}, {11,5,10,11,7,5,8,3,0,-1}, {5,11,7,5,10,11,1,9,0,-1}, {10,7,5,10,11,7,9,8,1,8,3,1,-1}, {11,1,2,11,7,1,7,5,1,-1}, {0,8,3,1,2,7,1,7,5,7,2,11,-1}, {9,7,5,9,2,7,9,0,2,2,11,7,-1}, {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1}, {2,5,10,2,3,5,3,7,5,-1}, {8,2,0,8,5,2,8,7,5,10,2,5,-1}, {9,0,1,5,10,3,5,3,7,3,10,2,-1}, {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1}, {1,3,5,3,7,5,-1}, {0,8,7,0,7,1,1,7,5,-1}, {9,0,3,9,3,5,5,3,7,-1}, {9,8,7,5,9,7,-1},
					{5,8,4,5,10,8,10,11,8,-1}, {5,0,4,5,11,0,5,10,11,11,3,0,-1}, {0,1,9,8,4,10,8,10,11,10,4,5,-1}, {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1}, {2,5,1,2,8,5,2,11,8,4,5,8,-1}, {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1}, {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1}, {9,4,5,2,11,3,-1}, {2,5,10,3,5,2,3,4,5,3,8,4,-1}, {5,10,2,5,2,4,4,2,0,-1}, {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1}, {5,10,2,5,2,4,1,9,2,9,4,2,-1}, {8,4,5,8,5,3,3,5,1,-1}, {0,4,5,1,0,5,-1}, {8,4,5,8,5,3,9,0,5,0,3,5,-1}, {9,4,5,-1},
					{4,11,7,4,9,11,9,10,11,-1}, {0,8,3,4,9,7,9,11,7,9,10,11,-1}, {1,10,11,1,11,4,1,4,0,7,4,11,-1}, {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1}, {4,11,7,9,11,4,9,2,11,9,1,2,-1}, {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1}, {11,7,4,11,4,2,2,4,0,-1}, {11,7,4,11,4,2,8,3,4,3,2,4,-1}, {2,9,10,2,7,9,2,3,7,7,4,9,-1}, {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1}, {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1}, {1,10,2,8,7,4,-1}, {4,9,1,4,1,7,7,1,3,-1}, {4,9,1,4,1,7,0,8,1,8,7,1,-1}, {4,0,3,7,4,3,-1}, {4,8,7,-1},
					{9,10,8,10,11,8,-1}, {3,0,9,3,9,11,11,9,10,-1}, {0,1,10,0,10,8,8,10,11,-1}, {3,1,10,11,3,10,-1}, {1,2,11,1,11,9,9,11,8,-1}, {3,0,9,3,9,11,1,2,9,2,11,9,-1}, {0,2,11,8,0,11,-1}, {3,2,11,-1}, {2,3,8,2,8,10,10,8,9,-1}, {9,10,2,0,9,2,-1}, {2,3,8,2,8,10,0,1,8,1,10,8,-1}, {1,10,2,-1}, {1,3,8,9,1,8,-1}, {0,9,1,-1}, {0,3,8,-1}, {-1}
				};

				// calculate cube index in the table
				int cubeIndex = 0;
				for (int i = 0; i < 8; i++)
					cubeIndex |= (int(cube[i] <= 0.) << i);

				// check table
				if (cubeIndex != 0 && cubeIndex != 0xff) {  // this line may be unnecessary

					// zeros on edges interpolated from samples
					vec3 intp[12];
					for (int e = 0; e < 12; e++)
						if ((1 << e) & edgeTable[cubeIndex]) {
							// linear interpolation
							vec3 p1 = vec3(vertexList[edgeList[e].x]);
							vec3 p2 = vec3(vertexList[edgeList[e].y]);
							double v1 = cube[edgeList[e].x];
							double v2 = cube[edgeList[e].y];
							vec3 pd = p1 + (v1 / (v1 - v2))*(p2 - p1);
							intp[e] = p0 + (vec3(i, j, k) + pd)*dp;
						}

					// construct triangles
					for (int t = 0; triTable[cubeIndex][t] != -1; t += 3) {
						T.push_back(triangle_3d(
							intp[triTable[cubeIndex][t]],
							intp[triTable[cubeIndex][t + 1]],
							intp[triTable[cubeIndex][t + 2]]
							));
					}
				}
			}
		}

	}

	delete xy0; delete xy1;
}


#include "ui/stl_encoder.h"
#include <chrono>

int main(int argc, char* argv[]) {
	std::vector<triangle_3d> T;

	auto time_start = std::chrono::high_resolution_clock::now();

	vec3 bound(2., 2., 2.);
	marching_cube(fun, T, -bound, bound, ivec3(40.*bound));

	auto time_end = std::chrono::high_resolution_clock::now();
	double time_elapsed = std::chrono::duration<double>(time_end - time_start).count();
	printf("%lf\n", time_elapsed);

	writeSTL(argv[1], &T[0], T.size(), nullptr, STL_CCW);
	return 0;
}

