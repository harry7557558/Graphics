#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"

int NX, NY, NZ;
float ***data;
std::vector<stl_triangle> trigs;
const float iso = 0.5;

void loadData(const char file[], const char info[], mat3 &orient, vec3 &translate) {
	// read file info
	FILE* fp = fopen(info, "r");
	fscanf(fp, "%d %d %d", &NX, &NY, &NZ);
	int bits; float minval, maxval;
	fscanf(fp, "%d %f %f", &bits, &minval, &maxval);
	fscanf(fp, "%lf %lf %lf %lf", &orient[0][0], &orient[0][1], &orient[0][2], &translate.x);
	fscanf(fp, "%lf %lf %lf %lf", &orient[1][0], &orient[1][1], &orient[1][2], &translate.y);
	fscanf(fp, "%lf %lf %lf %lf", &orient[2][0], &orient[2][1], &orient[2][2], &translate.z);
	fclose(fp);

	// read data
	fp = fopen(file, "rb");
	int bytes = bits / 8;  // should be 1 or 2
	uint8_t *fr = new uint8_t[bytes * NX*NY*NZ];
	fread(fr, bytes, NX*NY*NZ, fp);
	fclose(fp);
	int f = 0;  // file pointer
	data = new float**[NZ];
	for (int z = 0; z < NZ; z++) {
		data[z] = new float*[NY];
		for (int y = 0; y < NY; y++) {
			data[z][y] = new float[NX];
			for (int x = 0; x < NX; x++) {
				uint64_t val = 0;
				for (int i = 0; i < bytes; i++) val |= (uint32_t)fr[f++] << (8 * i);
				data[z][y][x] = (float(val) - minval) / (maxval - minval) - iso;
			}
		}
	}
	delete fr;

}



float sample(vec3 p) {
	// I think I have a bug here...

	p *= 0.9999*vec3(1, 1, 1);
	ivec3 p0 = ivec3(p), p1 = p0 + ivec3(1);
	vec3 pf = p - vec3(p0);

	//return data[p0.z][p0.y][p0.x];  // nearest??
	return mix(\
		mix(\
			mix(data[p0.z][p0.y][p0.x], data[p0.z][p0.y][p1.x], (float)pf.x), \
			mix(data[p0.z][p1.y][p0.x], data[p0.z][p1.y][p1.x], (float)pf.x), \
			(float)pf.y), \
		mix(\
			mix(data[p1.z][p0.y][p0.x], data[p1.z][p0.y][p1.x], (float)pf.x), \
			mix(data[p1.z][p1.y][p0.x], data[p1.z][p1.y][p1.x], (float)pf.x), \
			(float)pf.y), \
		(float)pf.z);  // tri-linear interpolation, not perfect

	// tri-cubic interpolation, slow, still not perfect
	auto interpolate = [](float s0, float s1, float s2, float s3, float t)->float {
		float t2 = t * t, t3 = t2 * t;
		return s0 * (-0.5f*t3 + t2 - 0.5f*t)
			+ s1 * (1.5f*t3 - 2.5f*t2 + 1)
			+ s2 * (-1.5f*t3 + 2.f*t2 + 0.5f*t)
			+ s3 * (0.5f*t3 - 0.5f*t2);
	};
	float samples[4][4][4];
	for (int i = -1; i <= 2; i++) {
		for (int j = -1; j <= 2; j++) {
			for (int k = -1; k <= 2; k++) {
				samples[i + 1][j + 1][k + 1] = data[clamp(p0.z + i, 0, NZ - 1)][clamp(p0.y + j, 0, NY - 1)][clamp(p0.x + k, 0, NX - 1)];
			}
		}
	}
	float xy[4][4];
	for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++)
		xy[i][j] = interpolate(samples[i][j][0], samples[i][j][1], samples[i][j][2], samples[i][j][3], (float)pf.x);
	float x[4];
	for (int i = 0; i < 4; i++) x[i] = interpolate(xy[i][0], xy[i][1], xy[i][2], xy[i][3], (float)pf.y);
	return interpolate(x[0], x[1], x[2], x[3], (float)pf.z);
}


int main(int argc, char* argv[]) {
	mat3 orient = mat3(1.0);
	vec3 translate = vec3(0.0);
	loadData(argv[2], argv[3], orient, translate);

	//std::vector<triangle_3d> ts = ScalarFieldTriangulator_octatree::octatree(sample, vec3(0), vec3(NX - 1, NY - 1, NZ - 1), ivec3(NX, NY, NZ) / 4, 0);
	std::vector<triangle_3d> ts = ScalarFieldTriangulator_octatree::marching_cube<float, vec3, triangle_3d>(data, 0.0, NZ, NY, NX);

	for (int i = 0; i < (int)ts.size(); i++) {
		ts[i].applyMatrix(orient);
		ts[i].translate(translate);
	}

	writeSTL(argv[1], &ts[0], (int)ts.size(), nullptr, STL_CCW);

	return 0;
}
