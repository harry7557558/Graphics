// implicit surface triangulation

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"
#include <vector>


double getMemoryUsage();  // in MB


// scalar field
double test_fun_1(vec3 p) {
	p *= 2.;
	for (int i = 0; i < 6; i++) {
		p.x = abs(p.x) - .8;
		p = rotationMatrix_z(PI / 6.)*p;
		p = p.yzx();
	}
	return max(abs(p.x) - 1., length(p.yz()) - .5);
}
double test_fun_2(vec3 p) {
	vec3 w = p;
	for (int i = 0; i < 6; i++) {
		double x = w.x, y = w.y, z = w.z;
		w = vec3(x*x - y * y - z * z, 2.*x*z, 2.*x*y) + p;
	}
	return length(w) - 1.;
}

int evals;
double fun(vec3 p) {
	double x = p.x, y = p.y, z = p.z;
	evals++;
	//return length(p) - 1.;
	//return abs(x) + abs(y) + abs(z) - sqrt(2.);
	//return max(length(p) - 1., abs(x) + abs(y) + abs(z) - sqrt(2.));
	//return x*x + y*y + z*z*z - z*z;
	//return pow(x*x + 2.*y*y + z*z, 3.) - (9.*x*x + y*y)*z*z*z - 0.5;
	//return 4.0*pow(x*x + 2.*y*y + z*z - 1., 2.) - z*(5.*x*x*x*x - 10.*x*x*z*z + z*z*z*z) - 1.;
	//return pow(x*x + 2.25*y*y + z * z - 1, 3.) - (x*x + 0.1125*y*y)*z*z*z;
	//return 2 * y*(y*y - 3 * x*x)*(1 - z * z) + (x*x + y * y)*(x*x + y * y) - (9 * z*z - 1)*(1 - z * z);
	//return exp(10 * (4 * p.xy().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.xz().sqr() - pow(p.sqr() + 0.96, 2))) + exp(10 * (4 * p.yz().sqr() - pow(p.sqr() + 0.96, 2))) - 1.;
	return test_fun_1(p);
}


// triangles
std::vector<stl_triangle> Trigs;
void addBox(vec3 p0, vec3 p1, vec3 col) {
	return;
	const vec3 trig[12][3] = {
		{vec3(1,0,0), vec3(0,0,0), vec3(0,1,0)}, {vec3(0,1,0), vec3(1,1,0), vec3(1,0,0)},
		{vec3(0,0,1), vec3(0,0,0), vec3(1,0,0)}, {vec3(1,0,0), vec3(1,0,1), vec3(0,0,1)},
		{vec3(0,1,0), vec3(0,0,0), vec3(0,0,1)}, {vec3(0,0,1), vec3(0,1,1), vec3(0,1,0)},
		{vec3(1,0,1), vec3(1,1,1), vec3(0,1,1)}, {vec3(0,1,1), vec3(0,0,1), vec3(1,0,1)},
		{vec3(0,1,1), vec3(1,1,1), vec3(1,1,0)}, {vec3(1,1,0), vec3(0,1,0), vec3(0,1,1)},
		{vec3(1,1,0), vec3(1,1,1), vec3(1,0,1)}, {vec3(1,0,1), vec3(1,0,0), vec3(1,1,0)},
	};
	for (int i = 0; i < 12; i++) {
		Trigs.push_back(stl_triangle(p0 + trig[i][0] * (p1 - p0), p0 + trig[i][1] * (p1 - p0), p0 + trig[i][2] * (p1 - p0), col));
	}
}



// http://paulbourke.net/geometry/polygonise/

// list of vertice on a unit cube
const static ivec3 VERTICE_LIST[8] = {
	ivec3(0,0,0), ivec3(0,1,0), ivec3(1,1,0), ivec3(1,0,0),
	ivec3(0,0,1), ivec3(0,1,1), ivec3(1,1,1), ivec3(1,0,1)
};
const static int VERTICE_LIST_INV[2][2][2] = {
	{{0, 4}, {1, 5}}, {{3, 7}, {2, 6}}
};

// list of edges connecting two vertices on a unit cube
const static ivec2 EDGE_LIST[12] = {
	ivec2(0,1), ivec2(1,2), ivec2(2,3), ivec2(3,0),
	ivec2(4,5), ivec2(5,6), ivec2(6,7), ivec2(7,4),
	ivec2(0,4), ivec2(1,5), ivec2(2,6), ivec2(3,7)
};

// list of faces; opposite face: (+3)%6
const static int FACE_LIST[6][4] = {
	{0, 1, 5, 4}, {0, 3, 7, 4}, {0, 1, 2, 3},
	{2, 3, 7, 6}, {2, 1, 5, 6}, {4, 5, 6, 7}
};
const static ivec3 FACE_DIR[6] = {
	ivec3(-1,0,0), ivec3(0,-1,0), ivec3(0,0,-1),
	ivec3(1,0,0), ivec3(0,1,0), ivec3(0,0,1)
};


// lookup tables for reconstruction
const static int TRIG_TABLE[256][16] = {
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



// linear interpolation on an edge
vec3 getInterpolation(vec3 pos[8], float val[8], int i) {
	float v0 = val[EDGE_LIST[i].x];
	float v1 = val[EDGE_LIST[i].y];
	vec3 p0 = pos[EDGE_LIST[i].x];
	vec3 p1 = pos[EDGE_LIST[i].y];
	return p0 + (v0 / (v0 - v1))*(p1 - p0);
};




vec3 p0 = vec3(-2), p1 = vec3(2);
const ivec3 SEARCH_DIF = ivec3(20);
const int PLOT_DPS = 4;
const int PLOT_SIZE = 1 << PLOT_DPS;
const ivec3 GRID_SIZE = SEARCH_DIF * PLOT_SIZE;
vec3 i2f(ivec3 p) {  // position ID to position
	vec3 d = vec3(p) / vec3(GRID_SIZE);
	return p0 * (vec3(1.) - d) + p1 * d;
}



// octatree
// all integer coordinates are absolute
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
#ifdef _DEBUG
		if (q % sz != ivec3(0)) throw(__LINE__);
		if (sz > size) throw(__LINE__);
#endif
		ivec3 d = (q - _p) / (size >> 1);
		int i = VERTICE_LIST_INV[d.x][d.y][d.z];
		if (!c[i]) {
			c[i] = new octatree_node(size / 2, _p + d * (size / 2));
		}
		return c[i]->getGrid(q, sz);
	}
	int calcIndex() {
		if (isnan(v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7])) return (index = 0);
		return (index = int(v[0] < 0) |
			(int(v[1] < 0) << 1) |
			(int(v[2] < 0) << 2) |
			(int(v[3] < 0) << 3) |
			(int(v[4] < 0) << 4) |
			(int(v[5] < 0) << 5) |
			(int(v[6] < 0) << 6) |
			(int(v[7] < 0) << 7));
	}
	void subdivide();
};
octatree_node*** octatree = 0;  // a grid [x][y][z]
void create_octatree() {  // sample tree initialization
	octatree = new octatree_node**[SEARCH_DIF.x + 1];
	for (int x = 0; x <= SEARCH_DIF.x; x++) {
		octatree[x] = new octatree_node*[SEARCH_DIF.y + 1];
		for (int y = 0; y <= SEARCH_DIF.y; y++) {
			octatree[x][y] = new octatree_node[SEARCH_DIF.z + 1];
			for (int z = 0; z <= SEARCH_DIF.z; z++) {
				octatree[x][y][z] = octatree_node(PLOT_SIZE, ivec3(x, y, z)*PLOT_SIZE);
			}
		}
	}
}
void destroy_octatree() {  // sample tree destruction
	for (int x = 0; x <= SEARCH_DIF.x; x++) {
		for (int y = 0; y <= SEARCH_DIF.y; y++)
			delete[] octatree[x][y];
		delete octatree[x];
	}
	delete octatree;
	octatree = 0;
}
float getSample_global(ivec3 p) {  // access a sample on the sample tree
	ivec3 pi = p / PLOT_SIZE;
	return octatree[pi.x][pi.y][pi.z].getSample(p);
}
octatree_node* getGrid_global(ivec3 p, int size) {
	ivec3 pi = p / PLOT_SIZE;
	return octatree[pi.x][pi.y][pi.z].getGrid(p, size);
}


// marching cells
std::vector<octatree_node*> cells;


// grid subdivision
void octatree_node::subdivide() {
	for (int u = 0; u < 8; u++) if (!c[u])
		c[u] = new octatree_node(size / 2, _p + VERTICE_LIST[u] * (size / 2));

#if 0
	for (int u = 0; u < 8; u++) {
		for (int v = 0; v < 8; v++) c[u]->v[v] = getSample_global(c[u]->p[v]);
	}
#else
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
	for (int u = 0; u < 8; u++) for (int v = 0; v < 8; v++)
		c[u]->v[v] = samples[SUBDIV_LOOKUP[u][v]];

#endif

	return;
}


void triangulate() {

	// initialize octatree root
	create_octatree();
	for (int x = 0; x <= SEARCH_DIF.x; x++) {
		for (int y = 0; y <= SEARCH_DIF.y; y++) {
			for (int z = 0; z <= SEARCH_DIF.z; z++) {
				octatree[x][y][z].v[0] = (float)fun(i2f(octatree[x][y][z]._p = ivec3(x, y, z)*PLOT_SIZE));
			}
		}
	}
	for (int x = 0; x < SEARCH_DIF.x; x++) {
		for (int y = 0; y < SEARCH_DIF.y; y++) {
			for (int z = 0; z < SEARCH_DIF.z; z++) {
				for (int u = 1; u < 8; u++) {
					ivec3 p = ivec3(x, y, z) + VERTICE_LIST[u];
					octatree[x][y][z].v[u] = octatree[p.x][p.y][p.z].v[0];
				}
			}
		}
	}

	// initial sample cells
	cells.clear();
	for (int x = 0; x < SEARCH_DIF.x; x++) {
		for (int y = 0; y < SEARCH_DIF.y; y++) {
			for (int z = 0; z < SEARCH_DIF.z; z++) {
				octatree_node *n = &octatree[x][y][z];
				if (TRIG_TABLE[n->calcIndex()][0] != -1) {
					cells.push_back(n);
				}
			}
		}
	}
	// debug visualization
	for (int i = 0, cn = cells.size(); i < cn; i++)
		addBox(i2f(cells[i]->p(0)), i2f(cells[i]->p(6)), vec3(0.5, 0, 0));

	// subdivide grid cells
	for (int size = PLOT_SIZE; size > 1; size >>= 1) {
		std::vector<octatree_node*> new_cells;
		int s2 = size / 2;
		for (int i = 0, cn = cells.size(); i < cn; i++) {
			octatree_node* ci = cells[i];
			ci->subdivide();
			for (int u = 0; u < 8; u++) {
				if (TRIG_TABLE[ci->c[u]->calcIndex()][0] != -1) {
					new_cells.push_back(ci->c[u]);
				}
			}
		}
		cells = new_cells;
		// debug visualization
		for (int i = 0, cn = cells.size(); i < cn; i++)
			addBox(i2f(cells[i]->p(0)), i2f(cells[i]->p(6)), vec3(0.2, 0, 0.2));

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
						addBox(i2f(nb->p(0)), i2f(nb->p(6)), vec3(0.4, 0.4, 0));  // debug visualization
						cells.push_back(nb);
					}
					nb->edge_checked[(u + 3) % 6] = true;
				}
				ci->edge_checked[u] = true;
			}
		}
	}

	// reconstruct segments
	for (int i = 0, cn = cells.size(); i < cn; i++) {
		vec3 p[8];
		for (int j = 0; j < 8; j++) p[j] = i2f(cells[i]->p(j));
		const auto Si = TRIG_TABLE[cells[i]->calcIndex()];
		for (int u = 0; Si[u] != -1; u += 3) {
			vec3 a = getInterpolation(p, cells[i]->v, Si[u]);
			vec3 b = getInterpolation(p, cells[i]->v, Si[u + 1]);
			vec3 c = getInterpolation(p, cells[i]->v, Si[u + 2]);
			Trigs.push_back(stl_triangle(a, b, c, vec3(1.)));
		}
		// debug visualization
		addBox(i2f(cells[i]->p(0)), i2f(cells[i]->p(6)), vec3(0, 0, 1));
	}


	// clean up
	printf("%.2lf MB\n\n", getMemoryUsage());
	cells.clear();
	destroy_octatree();
}



#include <chrono>

#include <Windows.h>
#include <psapi.h>
double getMemoryUsage() {
	PROCESS_MEMORY_COUNTERS statex;
	GetProcessMemoryInfo(GetCurrentProcess(), &statex, sizeof(statex));
	return statex.WorkingSetSize / 1048576.0;  // in MB
}

int main(int argc, char* argv[]) {
	printf("%d byte node\n", sizeof(octatree_node));

	auto t0 = std::chrono::high_resolution_clock::now();
	triangulate();
	auto t1 = std::chrono::high_resolution_clock::now();

	printf("%d samples\n", evals);
	printf("%.1lfms\n", 1000.*std::chrono::duration<double>(t1 - t0).count());

	writeSTL(argv[1], &Trigs[0], Trigs.size(), nullptr, STL_CCW);
	return 0;
}

