#include "sdf_3d.h"

#include "UI/3d_reader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

#include <chrono>


bool loadFile(const char* filename, std::vector<triangle_3d_f> &trigs, vec3f &p0, vec3f &p1);




void save_image(const char* filename, float* sdf, int W, int H, float s) {
	COLORREF* img = new COLORREF[W*H];
	for (int i = 0; i < W*H; i++) {
		float d = s * sdf[i];
		vec3f col = d > 0. ? vec3f(.9f, .6f, .3f) : vec3f(.6f, .9f, 1.0f);
		col *= (1.0f - exp(-10.f*abs(d))) * (0.8f + 0.2f*cos(60.f*d));
		col = mix(col, vec3f(1.0f), clamp(1.0f - abs(20.0f*d), 0.0f, 1.0f));
		img[i] = toCOLORREF(col.zyx()) | 0xff000000;
	}
	stbi_write_png(filename, W, H, 4, img, 4 * W);
	delete img;
}

void save_volume_image(float* sdf, ivec3 S, int step, float sc) {
	for (int zi = 0; zi < S.z; zi += step) {
		char filename[256];
		sprintf(filename, "D:\\\\sdf_3d\\z_%04d.png", zi);
		save_image(filename, &sdf[S.x*S.y*zi], S.x, S.y, sc);
	}
}

// ==========================================================================

void test_image() {
	std::vector<triangle_3d_f> trigs;
	vec3f p0 = vec3f((float)INFINITY), p1 = -p0;
	//loadFile("D:\\isosphere.stl", trigs, p0, p1);
	//loadFile("D:\\suzanne_manifold_3.stl", trigs, p0, p1);
	loadFile("D:\\bunny_manifold.stl", trigs, p0, p1);
	//loadFile("D:\\stanford_dragon.ply", trigs, p0, p1);
	//loadFile("D:\\Coding\\Github\\Graphics\\modeling\\volume\\ct_head.reduced.ply", trigs, p0, p1);
	//loadFile("D:\\Explore\\Thingi10K\\Thingi10K\\raw_meshes\\34784.stl", trigs, p0, p1);

	printf("%d\n", (int)trigs.size());

	const ivec3 S = ivec3(256);  // size of voxels
	vec3f r = 1.0f * vec3f(std::max({ p1.x - p0.x, p1.y - p0.y, p1.z - p0.z })) * normalize(vec3f(S));
	vec3f q0 = 0.5f*(p0 + p1) - r, q1 = q0 + 2.0f*r;
	float* sdf = new float[S.z*S.y*S.x];
	for (int i = S.z*S.y*S.x; i--;) sdf[i] = NAN;

	auto t0 = std::chrono::high_resolution_clock::now();
	//sdf_grid_bruteforce(trigs, sdf, q0, q1, S);
	sdf_grid_expand(trigs, sdf, q0, q1, S);
	float time_elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
	printf("%.1lfs\n", time_elapsed);

	float sc = 0.02f * length(vec3f(S)) / length(q1 - q0);
	save_volume_image(sdf, S, 1, sc);

	delete sdf;
}


void write_bunny() {
	// bunny is defined inside the unit sphere
	std::vector<triangle_3d_f> trigs;
	vec3f p0 = vec3f((float)INFINITY), p1 = -p0;
	loadFile("D:\\bunny_manifold.stl", trigs, p0, p1);
	p0 = vec3f(-1.0f), p1 = vec3f(1.0f);

	const ivec3 S = ivec3(64);
	float* sdf = new float[S.z*S.y*S.x];
	sdf_grid_expand(trigs, sdf, p0, p1, S);

	std::vector<vec4f> samples;
	for (int zi = 0; zi < S.z; zi++) {
		for (int yi = 0; yi < S.y; yi++) {
			for (int xi = 0; xi < S.x; xi++) {
				vec3f p = mix(p0, p1, vec3f(ivec3(xi, yi, zi)) / vec3f(S - ivec3(1)));
				float v = sdf[(zi*S.y + yi)*S.x + xi];
				if (length(p) < 1.0f) samples.push_back(vec4f(p, v));
				if (abs(v) < 0.1f) samples.push_back(vec4f(p, v));
			}
		}
	}
	printf("%d\n", samples.size());

	uint32_t seed = 0;
	for (int i = (int)samples.size() - 1; i > 0; i--) {
		int random = int(rand01(seed) * (i + 1));
		std::swap(samples[i], samples[random]);
	}

	FILE* fp = fopen("D:\\bunny_train.raw", "wb");
	fwrite(&samples[0], sizeof(vec4f), samples.size(), fp);
	fclose(fp);
}


int main() {
	//test_image();
	write_bunny();

	return 0;
}



bool loadFile(const char* filename,
	std::vector<triangle_3d_f> &trigs, vec3f &p0, vec3f &p1) {
	vec3f* Vs; int VN;
	ply_triangle* Fs; int FN;
	COLORREF *vcol = 0, *fcol = 0;
	FILE* fp = fopen(filename, "rb");
	if (!read3DFile(fp, Vs, Fs, VN, FN, vcol, fcol)) return false;
	fclose(fp);
	for (int i = 0; i < VN; i++)
		p0 = min(p0, Vs[i]), p1 = max(p1, Vs[i]);
	for (int i = 0; i < FN; i++)
		trigs.push_back(triangle_3d_f{ Vs[Fs[i][0]], Vs[Fs[i][1]], Vs[Fs[i][2]] });
	delete Vs; delete Fs;
	if (vcol) delete vcol; if (fcol) delete fcol;
	return true;
}

