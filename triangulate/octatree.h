
#ifndef __INC_GEOMETRY_H
#include "numerical/geometry.h"
#endif



// Implicit surface triangulation based on marching cube.
// Contain global variables, may have problems with multithreading.

// Functions in namespace ScalarFieldTriangulator_octatree:

// std::vector<triangle_3d> marching_cube(Fun fun, vec3 p0, vec3 p1, ivec3 dif);

// std::vector<triangle_3d> marching_cube_cylindrical_generalized(Fun fun, \
		vec3 i, vec3 j, vec3 k, double r0, double r1, double theta0, double theta1, double h0, double h1, \
		int r_N, int theta_N, int h_N);

// std::vector<triangle_3d> marching_cube_cylindrical(Fun fun, double r1, double h0, double h1, int rN, int thetaN, int hN);
// std::vector<triangle_3d> marching_cube_cylindrical_x(Fun fun, double r1, double x0, double x1, int rN, int thetaN, int xN);
// std::vector<triangle_3d> marching_cube_cylindrical_y(Fun fun, double r1, double y0, double y1, int rN, int thetaN, int yN);

// std::vector<triangle_3d> octatree_cylindrical_generalized(Fun fun, \
		vec3 i, vec3 j, vec3 k, double r0, double r1, double theta0, double theta1, double h0, double h1, \
		int r_N, int theta_N, int h_N, int plot_depth);

// std::vector<triangle_3d> octatree_cylindrical(Fun fun, double r1, double h0, double h1, int rN, int thetaN, int hN, int plot_depth);
// std::vector<triangle_3d> octatree_cylindrical_x(Fun fun, double r1, double x0, double x1, int rN, int thetaN, int xN, int plot_depth);
// std::vector<triangle_3d> octatree_cylindrical_y(Fun fun, double r1, double y0, double y1, int rN, int thetaN, int yN, int plot_depth);

// std::vector<Triangle> marching_cube(Float ***data, Float iso, int NZ, int NY, int NX);


#include <vector>
#include <functional>




namespace ScalarFieldTriangulator_octatree {

#define ScalarFieldTriangulator_octatree_PRIVATE_ namespace __private__ {
#define ScalarFieldTriangulator_octatree__PRIVATE }

	ScalarFieldTriangulator_octatree_PRIVATE_;


	/* LOOKUP TABLES */

	// list of vertice on a unit cube
	const ivec3 VERTICE_LIST[8] = {
		ivec3(0,0,0), ivec3(0,1,0), ivec3(1,1,0), ivec3(1,0,0),
		ivec3(0,0,1), ivec3(0,1,1), ivec3(1,1,1), ivec3(1,0,1)
	};
	const int VERTICE_LIST_INV[2][2][2] = {
		{{0, 4}, {1, 5}}, {{3, 7}, {2, 6}}
	};

	// list of edges connecting two vertices on a unit cube
	const ivec2 EDGE_LIST[12] = {
		ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0),
		ivec2(4,5), ivec2(5,6), ivec2(6,7), ivec2(7,4),
		ivec2(0,4), ivec2(1,5), ivec2(2,6), ivec2(3,7)
	};

	// list of faces; opposite face: (+3)%6
	const int FACE_LIST[6][4] = {
		{0, 1, 5, 4}, {0, 3, 7, 4}, {0, 1, 2, 3},
		{2, 3, 7, 6}, {2, 1, 5, 6}, {4, 5, 6, 7}
	};
	const ivec3 FACE_DIR[6] = {
		ivec3(-1,0,0), ivec3(0,-1,0), ivec3(0,0,-1),
		ivec3(1,0,0), ivec3(0,1,0), ivec3(0,0,1)
	};

	// lookup tables for reconstruction
	// http://paulbourke.net/geometry/polygonise/
	const int EDGE_TABLE[256] = {
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
	const int TRIG_TABLE[256][16] = {
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

	// calculate index for table lookup, be careful about the uppercase function name
	int CalcIndex(const float v[8]) {
		if (isnan(v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7])) return 0;
		return int(v[0] < 0) |
			(int(v[1] < 0) << 1) |
			(int(v[2] < 0) << 2) |
			(int(v[3] < 0) << 3) |
			(int(v[4] < 0) << 4) |
			(int(v[5] < 0) << 5) |
			(int(v[6] < 0) << 6) |
			(int(v[7] < 0) << 7);
	}

	// linear interpolation on an edge
	vec3 getInterpolation(const vec3 pos[8], const float val[8], int i) {
		float v0 = val[EDGE_LIST[i].x];
		float v1 = val[EDGE_LIST[i].y];
		vec3 p0 = pos[EDGE_LIST[i].x];
		vec3 p1 = pos[EDGE_LIST[i].y];
		if (v0 > v1) std::swap(v0, v1), std::swap(p0, p1);  // make it consistant
		float t = v0 / (v0 - v1);
		return p0 * (1 - t) + p1 * t;
	};



	/* DATA */

	std::function<double(vec3)> fun;  // double/float fun(vec3)
	vec3 p0, p1;  // search interval
	ivec3 SEARCH_DIF;  // initial diff
	int PLOT_DPS;  // octatree depth
	int PLOT_SIZE;  // 1 << PLOT_DPS
	ivec3 GRID_SIZE;  // SEARCH_DIF * PLOT_SIZE
	void calcParams() {
		PLOT_SIZE = 1 << PLOT_DPS;
		GRID_SIZE = SEARCH_DIF * PLOT_SIZE;
	}

	// position ID to position
	vec3 i2f(ivec3 p) {
		vec3 d = vec3(p) / vec3(GRID_SIZE);
		return p0 * (vec3(1.) - d) + p1 * d;
	}


	/* OCTATREE */

	// octatree
	float getSample_global(ivec3 p);
	class octatree_node {
	public:
		// total: 96 or 128 bytes
		float v[8]; // 32 bytes, only the first one really matter
		octatree_node *c[8];  // 32 or 64 bytes, child nodes
		ivec3 _p; // 12 bytes, vertice IDs
		ivec3 p(int i) { return _p + VERTICE_LIST[i] * size; }
		int size;  // 4 bytes, top right: p+ivec3(size)
		int index;  // 4 bytes, calculated according to signs of v for table lookup
		bool hasSignChange[6];  // 6 bytes, indicate whether there is a sign change at each face
		bool edge_checked[6];  // 6 bytes, used in looking for missed samples, indicate whether the face is already checked
		octatree_node(int size = 0, ivec3 p = ivec3(-1)) {
			for (int i = 0; i < 8; i++) this->_p = p;
			for (int i = 0; i < 8; i++) v[i] = NAN;
			this->size = size;
			this->index = -1;
			for (int i = 0; i < 8; i++) c[i] = nullptr;
			for (int i = 0; i < 6; i++) hasSignChange[i] = edge_checked[i] = false;
		}
		~octatree_node() {
			for (int i = 0; i < 8; i++) if (c[i]) {
				delete c[i]; c[i] = 0;
			}
		}
		float getSample(ivec3 q) {
			if (q == _p) {
				if (isnan(v[0])) {
					if (c[0]) {
						v[0] = c[0]->getSample(q);
					}
					else v[0] = (float)fun(i2f(_p));
				}
				return v[0];
			}
			ivec3 d = (q - _p) / (size >> 1);
			int i = VERTICE_LIST_INV[d.x][d.y][d.z];
			if (!c[i]) {
				c[i] = new octatree_node(size / 2, _p + d * (size / 2));
			}
			return c[i]->getSample(q);
		}
		octatree_node* getGrid(ivec3 q, int sz) {
			if (q == _p && sz == size) {
				if (isnan(v[0])) {
					v[0] = getSample_global(_p);
				}
				for (int i = 1; i < 8; i++) if (isnan(v[i])) {
					v[i] = getSample_global(p(i));
				}
				return this;
			}
			ivec3 d = (q - _p) / (size >> 1);
			int i = VERTICE_LIST_INV[d.x][d.y][d.z];
			if (!c[i]) {
				c[i] = new octatree_node(size / 2, _p + d * (size / 2));
			}
			return c[i]->getGrid(q, sz);
		}
		int calcIndex() {
			return (index = CalcIndex(v));
		}

		// grid subdivision
		void subdivide();

	};

	octatree_node*** octatree_grid = 0;  // a grid [x][y][z]
	void create_octatree() {  // sample tree initialization
		octatree_grid = new octatree_node**[SEARCH_DIF.x + 1];
		for (int x = 0; x <= SEARCH_DIF.x; x++) {
			octatree_grid[x] = new octatree_node*[SEARCH_DIF.y + 1];
			for (int y = 0; y <= SEARCH_DIF.y; y++) {
				octatree_grid[x][y] = new octatree_node[SEARCH_DIF.z + 1];
				for (int z = 0; z <= SEARCH_DIF.z; z++) {
					octatree_grid[x][y][z] = octatree_node(PLOT_SIZE, ivec3(x, y, z)*PLOT_SIZE);
				}
			}
		}
	}
	void destroy_octatree() {  // sample tree destruction
		for (int x = 0; x <= SEARCH_DIF.x; x++) {
			for (int y = 0; y <= SEARCH_DIF.y; y++)
				delete[] octatree_grid[x][y];
			delete octatree_grid[x];
		}
		delete octatree_grid;
		octatree_grid = 0;
	}
	float getSample_global(ivec3 p) {  // access a sample on the sample tree
		ivec3 pi = p / PLOT_SIZE;
		return octatree_grid[pi.x][pi.y][pi.z].getSample(p);
	}
	octatree_node* getGrid_global(ivec3 p, int size) {
		ivec3 pi = p / PLOT_SIZE;
		return octatree_grid[pi.x][pi.y][pi.z].getGrid(p, size);
	}


	void octatree_node::subdivide() {

		for (int u = 0; u < 8; u++) if (!c[u])
			c[u] = new octatree_node(size / 2, _p + VERTICE_LIST[u] * (size / 2));

		float samples[27];
		samples[0] = v[0];
		samples[2] = v[1];
		samples[4] = v[2];
		samples[6] = v[3];
		samples[18] = v[4];
		samples[20] = v[5];
		samples[22] = v[6];
		samples[24] = v[7];
		samples[1] = isnan(c[1]->v[0]) ? (float)fun(i2f(c[1]->_p)) : c[1]->v[0];
		samples[3] = getSample_global(c[1]->p(2));
		samples[5] = getSample_global(c[3]->p(2));
		samples[7] = isnan(c[3]->v[0]) ? (float)fun(i2f(c[3]->_p)) : c[3]->v[0];
		samples[8] = isnan(c[2]->v[0]) ? (float)fun(i2f(c[2]->_p)) : c[2]->v[0];
		samples[9] = isnan(c[4]->v[0]) ? (float)fun(i2f(c[4]->_p)) : c[4]->v[0];
		samples[10] = isnan(c[5]->v[0]) ? (float)fun(i2f(c[5]->_p)) : c[5]->v[0];
		samples[11] = getSample_global(c[5]->p(1));
		samples[12] = getSample_global(c[5]->p(2));
		samples[13] = getSample_global(c[6]->p(2));
		samples[14] = getSample_global(c[7]->p(2));
		samples[15] = getSample_global(c[7]->p(3));
		samples[16] = isnan(c[7]->v[0]) ? (float)fun(i2f(c[7]->_p)) : c[7]->v[0];
		samples[17] = isnan(c[6]->v[0]) ? (float)fun(i2f(c[6]->_p)) : c[6]->v[0];
		samples[19] = getSample_global(c[4]->p(5));
		samples[21] = getSample_global(c[5]->p(6));
		samples[23] = getSample_global(c[7]->p(6));
		samples[25] = getSample_global(c[4]->p(7));
		samples[26] = getSample_global(c[4]->p(6));

		const static int SUBDIV_LOOKUP[8][8] = {
			{0, 1, 8, 7, 9, 10, 17, 16},
			{1, 2, 3, 8, 10, 11, 12, 17},
			{8, 3, 4, 5, 17, 12, 13, 14},
			{7, 8, 5, 6, 16, 17, 14, 15},
			{9, 10, 17, 16, 18, 19, 26, 25},
			{10, 11, 12, 17, 19, 20, 21, 26},
			{17, 12, 13, 14, 26, 21, 22, 23},
			{16, 17, 14, 15, 25, 26, 23, 24}
		};
		for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++) {
			c[u]->v[v] = samples[SUBDIV_LOOKUP[u][v]];
		}

	}


	// check lookup table to construct triangles from cell
	int addTriangle(const vec3 p[8], const float v[8], std::vector<triangle_3d> &trigs) {
		const auto Si = TRIG_TABLE[CalcIndex(v)];
		for (int u = 0; ; u += 3) {
			if (Si[u] == -1) return u / 3;
			vec3 a = getInterpolation(p, v, Si[u]);
			vec3 b = getInterpolation(p, v, Si[u + 1]);
			vec3 c = getInterpolation(p, v, Si[u + 2]);
			trigs.push_back(triangle_3d(a, b, c));
		}
	}


	/* CALL FUNCTIONS */

	std::vector<octatree_node*> cells;

	// octatree main function, construct octatree and march cubes to @cells
	void octatree_main() {

		// initialize octatree root
		create_octatree();
		for (int x = 0; x <= SEARCH_DIF.x; x++) {
			for (int y = 0; y <= SEARCH_DIF.y; y++) {
				for (int z = 0; z <= SEARCH_DIF.z; z++) {
					octatree_grid[x][y][z].v[0] = (float)fun(i2f(octatree_grid[x][y][z]._p = ivec3(x, y, z)*PLOT_SIZE));
				}
			}
		}
		for (int x = 0; x < SEARCH_DIF.x; x++) {
			for (int y = 0; y < SEARCH_DIF.y; y++) {
				for (int z = 0; z < SEARCH_DIF.z; z++) {
					for (int u = 1; u < 8; u++) {
						ivec3 p = ivec3(x, y, z) + VERTICE_LIST[u];
						octatree_grid[x][y][z].v[u] = octatree_grid[p.x][p.y][p.z].v[0];
					}
				}
			}
		}

		// initial sample cells
		cells.clear();
		for (int x = 0; x < SEARCH_DIF.x; x++) {
			for (int y = 0; y < SEARCH_DIF.y; y++) {
				for (int z = 0; z < SEARCH_DIF.z; z++) {
					octatree_node *n = &octatree_grid[x][y][z];
					if (TRIG_TABLE[n->calcIndex()][0] != -1) {
						cells.push_back(n);
					}
				}
			}
		}

		// subdivide grid cells
		for (int size = PLOT_SIZE; size > 1; size >>= 1) {
			std::vector<octatree_node*> new_cells;
			int s2 = size / 2;
			for (int i = 0, cn = (int)cells.size(); i < cn; i++) {
				octatree_node* ci = cells[i];
				ci->subdivide();
				for (int u = 0; u < 8; u++) {
					if (TRIG_TABLE[ci->c[u]->calcIndex()][0] != -1) {
						new_cells.push_back(ci->c[u]);
					}
				}
			}
			cells = new_cells;

			// try to add missed samples
			for (int i = 0; i < (int)cells.size(); i++) {
				octatree_node* ci = cells[i];
				for (int u = 0; u < 6; u++) {
					ci->hasSignChange[u] = ((int)signbit(ci->v[FACE_LIST[u][0]]) + (int)signbit(ci->v[FACE_LIST[u][1]]) + (int)signbit(ci->v[FACE_LIST[u][2]]) + (int)signbit(ci->v[FACE_LIST[u][3]])) % 4 != 0;
				}
			}
			for (int i = 0; i < (int)cells.size(); i++) {
				octatree_node* ci = cells[i];
				for (int u = 0; u < 6; u++) if (ci->hasSignChange[u] && !ci->edge_checked[u]) {
					ivec3 nb_p = ci->p(0) + FACE_DIR[u] * ci->size;
					if (nb_p.x >= 0 && nb_p.y >= 0 && nb_p.z >= 0 && nb_p.x < GRID_SIZE.x && nb_p.y < GRID_SIZE.y && nb_p.z < GRID_SIZE.z) {
						octatree_node* nb = getGrid_global(nb_p, ci->size);
						if (!nb->hasSignChange[(u + 3) % 6]) {
							for (int u = 0; u < 6; u++)
								nb->hasSignChange[u] = ((int)signbit(nb->v[FACE_LIST[u][0]]) + (int)signbit(nb->v[FACE_LIST[u][1]]) + (int)signbit(nb->v[FACE_LIST[u][2]]) + (int)signbit(nb->v[FACE_LIST[u][3]])) % 4 != 0;
							cells.push_back(nb);
						}
						nb->edge_checked[(u + 3) % 6] = true;
					}
					ci->edge_checked[u] = true;
				}
			}
		}


	}


	void triangulate(std::vector<triangle_3d> &Trigs) {

		octatree_main();

		for (int i = 0, cn = (int)cells.size(); i < cn; i++) {
			vec3 p[8];
			for (int j = 0; j < 8; j++) p[j] = i2f(cells[i]->p(j));
			addTriangle(p, cells[i]->v, Trigs);
		}

		destroy_octatree();
	}

	// construct triangles with gradient information, gradient is calculated at the center of the cell (poses)
	struct triangle_3d_with_grad {
		triangle_3d trig;  // triangle
		vec3 p;  // where gradient is estimated
		vec3 grad;  // estimated gradient
	};
	void triangulate_grad(std::vector<triangle_3d_with_grad> &Trigs) {

		octatree_main();
		int cn = (int)cells.size();

		std::vector<triangle_3d> trigs;
		for (int i = 0; i < cn; i++) {
			vec3 p[8];
			for (int j = 0; j < 8; j++) p[j] = i2f(cells[i]->p(j));
			trigs.clear();
			int n = addTriangle(p, cells[i]->v, trigs);
			// estimate gradient
			vec3 grad = vec3(0);
			for (int j = 0; j < 8; j++)
				grad += cells[i]->v[j] * (vec3(-1.) + 2.*vec3(VERTICE_LIST[j]));
			grad /= 4.*(p[6] - p[0]);
			vec3 pos = 0.5*(p[0] + p[6]);
			// add
			for (int j = 0; j < n; j++) {
				Trigs.push_back(triangle_3d_with_grad{ trigs[j], pos, grad });
			}
		}

		destroy_octatree();
	}


	ScalarFieldTriangulator_octatree__PRIVATE;




	// standard marching cube, independent to the quadtree part under the namespace
	template<typename Fun>
	std::vector<triangle_3d> marching_cube(Fun fun, vec3 p0, vec3 p1, ivec3 dif) {
		std::vector<triangle_3d> Trigs;
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

					// calculate cube index in the table
					int cubeIndex = 0;
					for (int i = 0; i < 8; i++)
						cubeIndex |= (int(cube[i] <= 0.) << i);

					// check table
					auto EDGE_TABLE = __private__::EDGE_TABLE;
					auto VERTICE_LIST = __private__::VERTICE_LIST;
					auto EDGE_LIST = __private__::EDGE_LIST;
					auto TRIG_TABLE = __private__::TRIG_TABLE;
					if (cubeIndex != 0 && cubeIndex != 0xff) {  // this line may be unnecessary
						vec3 intp[12];
						for (int e = 0; e < 12; e++)
							if ((1 << e) & EDGE_TABLE[cubeIndex]) {
								// linear interpolation
								vec3 p1 = vec3(VERTICE_LIST[EDGE_LIST[e].x]);
								vec3 p2 = vec3(VERTICE_LIST[EDGE_LIST[e].y]);
								double v1 = cube[EDGE_LIST[e].x];
								double v2 = cube[EDGE_LIST[e].y];
								double t = v1 / (v1 - v2);
								vec3 pd = p1 * (1 - t) + p2 * t;
								intp[e] = p0 + (vec3(i, j, k) + pd)*dp;
							}

						// construct triangles
						for (int t = 0; TRIG_TABLE[cubeIndex][t] != -1; t += 3) {
							vec3 p[3] = {
								intp[TRIG_TABLE[cubeIndex][t]],
								intp[TRIG_TABLE[cubeIndex][t + 1]],
								intp[TRIG_TABLE[cubeIndex][t + 2]]
							};
							if (p[0] != p[1] && p[0] != p[2] && p[1] != p[2])
								Trigs.push_back(triangle_3d(p[0], p[1], p[2]));
						}
					}
				}
			}

		}

		delete xy0; delete xy1;
		return Trigs;
	}

	// marching cube on discrete sample points, 3D C-style array in ZYX order, results are in [0,NX-1],[0,NY-1],[0,NZ-1]
	template<typename Float, typename vec, typename Triangle>
	std::vector<Triangle> marching_cube(Float ***data, Float iso, int NZ, int NY, int NX) {

		std::vector<Triangle> Trigs;

		auto VERTICE_LIST = __private__::VERTICE_LIST;
		auto EDGE_LIST = __private__::EDGE_LIST;
		auto EDGE_TABLE = __private__::EDGE_TABLE;
		auto TRIG_TABLE = __private__::TRIG_TABLE;

		for (int z = 0; z < NZ - 1; z++) {
			for (int y = 0; y < NY - 1; y++) {
				for (int x = 0; x < NX - 1; x++) {

					// read values on the cube
					Float cube[8];
					for (int i = 0; i < 8; i++)
						cube[i] = data[z + VERTICE_LIST[i].z][y + VERTICE_LIST[i].y][x + VERTICE_LIST[i].x] - iso;

					// calculate cube index in the table
					int cubeIndex = 0;
					for (int i = 0; i < 8; i++)
						cubeIndex |= (int(cube[i] <= 0.) << i);

					// check table
					if (cubeIndex != 0 && cubeIndex != 0xff) {  // this line may be unnecessary
						vec intp[12];
						for (int e = 0; e < 12; e++)
							if ((1 << e) & EDGE_TABLE[cubeIndex]) {
								// linear interpolation
								vec p1 = vec(VERTICE_LIST[EDGE_LIST[e].x]);
								vec p2 = vec(VERTICE_LIST[EDGE_LIST[e].y]);
								Float v1 = cube[EDGE_LIST[e].x];
								Float v2 = cube[EDGE_LIST[e].y];
								if (v1 > v2) std::swap(v1, v2), std::swap(p1, p2);
								Float t = v1 / (v1 - v2);
								vec pd = p1 * (1 - t) + p2 * t;
								intp[e] = vec(x, y, z) + pd;
							}

						// construct triangles
						for (int t = 0; TRIG_TABLE[cubeIndex][t] != -1; t += 3) {
							vec p[3] = {
								intp[TRIG_TABLE[cubeIndex][t]],
								intp[TRIG_TABLE[cubeIndex][t + 1]],
								intp[TRIG_TABLE[cubeIndex][t + 2]]
							};
							if (p[0] != p[1] && p[0] != p[2] && p[1] != p[2])
								Trigs.push_back(Triangle(p[0], p[1], p[2]));
						}
					}
				}
			}

		}

		return Trigs;

	}


	// octatree
	template<typename Fun>
	std::vector<triangle_3d> octatree(Fun fun, vec3 p0, vec3 p1, ivec3 dif, int plot_depth) {
		__private__::fun = fun;
		std::vector<triangle_3d> Trigs;
		__private__::p0 = p0, __private__::p1 = p1, __private__::SEARCH_DIF = dif, __private__::PLOT_DPS = plot_depth;
		__private__::calcParams();
		__private__::triangulate(Trigs);
		return Trigs;
	}
	template<typename Fun>
	std::vector<__private__::triangle_3d_with_grad> octatree_with_grad(Fun fun, vec3 p0, vec3 p1, ivec3 dif, int plot_depth) {
		__private__::fun = fun;
		std::vector<__private__::triangle_3d_with_grad> Trigs;
		__private__::p0 = p0, __private__::p1 = p1, __private__::SEARCH_DIF = dif, __private__::PLOT_DPS = plot_depth;
		__private__::calcParams();
		__private__::triangulate_grad(Trigs);
		return Trigs;
	}


	// marching cube in cylindrical coordinate
	template<typename Fun> std::vector<triangle_3d> marching_cube_cylindrical_generalized(Fun fun,
		vec3 i, vec3 j, vec3 k, double r0, double r1, double theta0, double theta1, double h0, double h1,
		int r_N, int theta_N, int h_N) {

		std::vector<triangle_3d> Trigs = marching_cube([&](vec3 p) {
			double r = p.x, theta = p.y, h = p.z;
			return fun(r * cos(theta) * i + r * sin(theta) * j + h * k);
		}, vec3(r0, theta0, h0), vec3(r1, theta1, h1), ivec3(r_N, theta_N, h_N));

		for (int u = 0; u < (int)Trigs.size(); u++) {
			for (int v = 0; v < 3; v++) {
				vec3 p = Trigs[u][v];
				double r = p.x, theta = p.y, h = p.z;
				Trigs[u][v] = r * cos(theta) * i + r * sin(theta) * j + h * k;
			}
		}
		return Trigs;
	}

	template<typename Fun> std::vector<triangle_3d> marching_cube_cylindrical(Fun fun, double r1, double h0, double h1, int rN, int thetaN, int hN) {
		return marching_cube_cylindrical_generalized(fun,
			vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
			0., r1, 0., 2.*PI, h0, h1, rN, thetaN, hN);
	}
	template<typename Fun> std::vector<triangle_3d> marching_cube_cylindrical_x(Fun fun, double r1, double x0, double x1, int rN, int thetaN, int xN) {
		return marching_cube_cylindrical_generalized(fun,
			vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 0),
			0., r1, 0., 2.*PI, x0, x1, rN, thetaN, xN);
	}
	template<typename Fun> std::vector<triangle_3d> marching_cube_cylindrical_y(Fun fun, double r1, double y0, double y1, int rN, int thetaN, int yN) {
		return marching_cube_cylindrical_generalized(fun,
			vec3(-1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0),
			0., r1, 0., 2.*PI, y0, y1, rN, thetaN, yN);
	}


	// octatree in cylindrical coordinate
	template<typename Fun> std::vector<triangle_3d> octatree_cylindrical_generalized(Fun fun,
		vec3 i, vec3 j, vec3 k, double r0, double r1, double theta0, double theta1, double h0, double h1,
		int r_N, int theta_N, int h_N, int plot_depth) {

		std::vector<triangle_3d> Trigs = octatree([&](vec3 p) {
			double r = p.x, theta = p.y, h = p.z;
			return fun(r * cos(theta) * i + r * sin(theta) * j + h * k);
		}, vec3(r0, theta0, h0), vec3(r1, theta1, h1), ivec3(r_N, theta_N, h_N), plot_depth);

		std::vector<triangle_3d> res;
		res.reserve(Trigs.size());
		for (int u = 0; u < (int)Trigs.size(); u++) {
			triangle_3d T = Trigs[u];
			for (int v = 0; v < 3; v++) {
				vec3 p = T[v];
				double r = p.x, theta = p.y, h = p.z;
				T[v] = r * cos(theta) * i + r * sin(theta) * j + h * k;
			}
			if (T[0] != T[1] && T[0] != T[2] && T[1] != T[2])
				res.push_back(T);
		}
		return res;
	}

	template<typename Fun> std::vector<triangle_3d> octatree_cylindrical(Fun fun, double r1, double h0, double h1, int rN, int thetaN, int hN, int plot_depth) {
		return octatree_cylindrical_generalized(fun,
			vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
			0., r1, 0., 2.*PI, h0, h1, rN, thetaN, hN, plot_depth);
	}
	template<typename Fun> std::vector<triangle_3d> octatree_cylindrical_x(Fun fun, double r1, double x0, double x1, int rN, int thetaN, int xN, int plot_depth) {
		return octatree_cylindrical_generalized(fun,
			vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 0),
			0., r1, 0., 2.*PI, x0, x1, rN, thetaN, xN, plot_depth);
	}
	template<typename Fun> std::vector<triangle_3d> octatree_cylindrical_y(Fun fun, double r1, double y0, double y1, int rN, int thetaN, int yN, int plot_depth) {
		return octatree_cylindrical_generalized(fun,
			vec3(-1, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0),
			0., r1, 0., 2.*PI, y0, y1, rN, thetaN, yN, plot_depth);
	}



#undef ScalarFieldTriangulator_octatree_PRIVATE_
#undef ScalarFieldTriangulator_octatree__PRIVATE

};


