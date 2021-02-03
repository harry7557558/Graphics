#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"

const int NX = 256, NY = 256, NZ = 113;
double ct[NX][NY][NZ];
std::vector<stl_triangle> data;

void loadData() {
	FILE* fp = fopen("CTHead.raw", "rb");
	int maxval = 0;
	for (int z = 0; z < NZ; z++) {
		for (int y = 0; y < NY; y++) {
			for (int x = 0; x < NX; x++) {
				int c0 = fgetc(fp), c1 = fgetc(fp), val = (c0 << 8) | c1;
				ct[x][y][NZ - 1 - z] = 1.0 - double(val) / 2814.;
				maxval = max(maxval, val);
			}
		}
	}
	printf("%d\n", maxval);  // 2814
}

double sample(vec3 p) {
	// I think I have a bug here...

	double iso = 0.5;
	p *= 0.9999*vec3(1, 1, 0.5);
	ivec3 p0 = ivec3(p), p1 = p0 + ivec3(1);
	vec3 pf = p - vec3(p0);

	//return ct[p0.x][p0.y][p0.z] - iso;  // nearest??
	return mix(\
		mix(\
			mix(ct[p0.x][p0.y][p0.z], ct[p0.x][p0.y][p1.z], pf.z), \
			mix(ct[p0.x][p1.y][p0.z], ct[p0.x][p1.y][p1.z], pf.z), \
			pf.y), \
		mix(\
			mix(ct[p1.x][p0.y][p0.z], ct[p1.x][p0.y][p1.z], pf.z), \
			mix(ct[p1.x][p1.y][p0.z], ct[p1.x][p1.y][p1.z], pf.z), \
			pf.y), \
		pf.x) - iso;  // tri-linear interpolation, not perfect

	// tri-cubic interpolation, slow, still not perfect
	auto interpolate = [](double s0, double s1, double s2, double s3, double t) {
		double t2 = t * t, t3 = t2 * t;
		return s0 * (-0.5*t3 + t2 - 0.5*t)
			+ s1 * (1.5*t3 - 2.5*t2 + 1)
			+ s2 * (-1.5*t3 + 2.*t2 + 0.5*t)
			+ s3 * (0.5*t3 - 0.5*t2);
	};
	double samples[4][4][4];
	for (int i = -1; i <= 2; i++) {
		for (int j = -1; j <= 2; j++) {
			for (int k = -1; k <= 2; k++) {
				samples[i + 1][j + 1][k + 1] = ct[clamp(p0.x + i, 0, NX - 1)][clamp(p0.y + j, 0, NY - 1)][clamp(p0.z + k, 0, NZ - 1)];
			}
		}
	}
	double xy[4][4];
	for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
		xy[i][j] = interpolate(samples[i][j][0], samples[i][j][1], samples[i][j][2], samples[i][j][3], pf.z);
	double x[4];
	for (int i = 0; i < 4; i++) x[i] = interpolate(xy[i][0], xy[i][1], xy[i][2], xy[i][3], pf.y);
	return interpolate(x[0], x[1], x[2], x[3], pf.x) - iso;
}


int main(int argc, char* argv[]) {
	loadData();

	std::vector<triangle_3d> trigs = ScalarFieldTriangulator_octatree::marching_cube(sample, vec3(0.), vec3(NX - 1, NY - 1, 2 * (NZ - 1)), ivec3(NX, NY, NZ));

	for (int i = 0; i < (int)trigs.size(); i++) {
		data.push_back(stl_triangle{ trigs[i][0], trigs[i][1], trigs[i][2] });
	}

	writeSTL(argv[1], &data[0], data.size(), nullptr, STL_CCW);

	return 0;
}
