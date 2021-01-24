#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"

const int NX = 200, NY = 160, NZ = 160;
double mri[NX][NY][NZ];
std::vector<stl_triangle> data;

void loadData() {
	FILE* fp = fopen("mri.raw", "rb");
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				mri[x][y][NZ - z - 1] = 1.0 - double(fgetc(fp)) / 255.;
			}
		}
	}
}

double sample(vec3 p) {
	double offset = 0.6;
	p *= vec3(0.9999);
	ivec3 p0 = ivec3(p), p1 = p0 + ivec3(1);
	vec3 pf = p - vec3(p0);
	//return mri[p0.x][p0.y][p0.z] - offset;  // nearest??
	return mix( \
		mix( \
			mix(mri[p0.x][p0.y][p0.z], mri[p0.x][p0.y][p1.z], pf.z), \
			mix(mri[p0.x][p1.y][p0.z], mri[p0.x][p1.y][p1.z], pf.z), \
			pf.y), \
		mix( \
			mix(mri[p1.x][p0.y][p0.z], mri[p1.x][p0.y][p1.z], pf.z), \
			mix(mri[p1.x][p1.y][p0.z], mri[p1.x][p1.y][p1.z], pf.z), \
			pf.y), \
		pf.x) - offset;  // tri-linear interpolation, not perfect
}


int main(int argc, char* argv[]) {
	loadData();

	std::vector<triangle_3d> trigs = ScalarFieldTriangulator_octatree::marching_cube(sample, vec3(0.), vec3(NX - 1, NY - 1, NZ - 1), ivec3(NX, NY, NZ));
	for (int i = 0; i < (int)trigs.size(); i++) {
		data.push_back(stl_triangle{ trigs[i][0], trigs[i][1], trigs[i][2] });
	}

	writeSTL(argv[1], &data[0], data.size(), nullptr, STL_CCW);

	return 0;
}
