// generade SDF volume files

#include <vector>
#include "numerical/geometry.h"
#include "ui/3d_reader.h"
#include "fitting/sdf/sdf_3d.h"


std::vector<triangle_3d_f> loadTriangles(const char* filename) {
	FILE* fp = fopen(filename, "rb");
	vec3f *vs; ply_triangle* fs;
	int VN, FN;
	COLORREF *v_col, *f_col;
	read3DFile(fp, vs, fs, VN, FN, v_col, f_col);
	fclose(fp);

	std::vector<triangle_3d_f> trigs; trigs.reserve(FN);
	for (int fi = 0; fi < FN; fi++) {
		trigs.push_back(triangle_3d_f{
			vs[fs[fi][0]], vs[fs[fi][1]], vs[fs[fi][2]]
			});
	}
	return trigs;
}

void writeRawFile(const char* filename, float *sdf, int size, float v0, float v255) {
	char *data = new char[size];
	for (int i = 0; i < size; i++) {
		float v = 255.0f * (sdf[i] - v0) / (v255 - v0);
		int d = (int)round(v);
		data[i] = ~(char)clamp(d, 0, 255);
	}
	FILE *fp = fopen(filename, "wb");
	fwrite(data, 1, size, fp);
	fclose(fp);
	delete data;
}

void mainBunny() {
	const ivec3 size(128, 128, 128);
	const int SIZE = size.x * size.y * size.z;
	float *sdf = new float[SIZE];
	sdf_grid_expand(loadTriangles("modeling/bunny_manifold.stl"),
		sdf, vec3f(-0.9), vec3f(0.9), size);
	writeRawFile("raytracing/webgl_volume/v_sdfbunny_128x128x128_uint8.raw",
		sdf, SIZE, -0.5, 0.5);
	delete sdf;
}

void mainSuzanne() {
	const ivec3 size(128, 128, 128);
	const int SIZE = size.x * size.y * size.z;
	float *sdf = new float[SIZE];
	vec3f b = vec3f(1.4f, 1.2f, 1.2f);
	sdf_grid_expand(loadTriangles("modeling/suzanne_manifold_3.stl"),
		sdf, -b, b, size);
	writeRawFile("raytracing/webgl_volume/v_sdfsuzanne_128x128x128_uint8.raw",
		sdf, SIZE, -0.6, 0.6);
	delete sdf;
}

void mainDragon() {
	const ivec3 size(128, 128, 128);
	const int SIZE = size.x * size.y * size.z;
	float *sdf = new float[SIZE];
	vec3f b = vec3f(1.3f, 0.9f, 1.1f);
	sdf_grid_expand(loadTriangles("modeling/stanford_dragon_oriented.stl"),
		sdf, -b, b, size);
	writeRawFile("raytracing/webgl_volume/v_sdfdragon_128x128x128_uint8.raw",
		sdf, SIZE, -0.55, 0.55);
	delete sdf;
}


int main(int argc, char* argv[]) {
	mainDragon();
	return 0;
}

