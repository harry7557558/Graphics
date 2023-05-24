#include "sdf_2d.h"

#include "UI/3d_reader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

#include <chrono>


bool loadFile(const char* filename, std::vector<triangle_3d_f> &trigs, vec3f &p0, vec3f &p1);



std::vector<segment_2d_f> clip_triangle(std::vector<triangle_3d_f> trigs, float z) {
	std::vector<segment_2d_f> segs;
	for (triangle_3d_f t : trigs) {
		if (z <= t.min_z() || z >= t.max_z())
			continue;
		float t01 = (z - t[0].z) / (t[1].z - t[0].z);
		float t12 = (z - t[1].z) / (t[2].z - t[1].z);
		float t20 = (z - t[2].z) / (t[0].z - t[2].z);
		if (t01 <= 0.0f || t01 > 1.0f)
			segs.push_back(segment_2d_f(mix(t[1].xy(), t[2].xy(), t12), mix(t[2].xy(), t[0].xy(), t20)));
		else if (t12 <= 0.0f || t12 > 1.0f)
			segs.push_back(segment_2d_f(mix(t[2].xy(), t[0].xy(), t20), mix(t[0].xy(), t[1].xy(), t01)));
		else if (t20 <= 0.0f || t20 > 1.0f)
			segs.push_back(segment_2d_f(mix(t[0].xy(), t[1].xy(), t01), mix(t[1].xy(), t[2].xy(), t12)));
	}
	return segs;
}




void save_image(const char* filename, float* vals, int W, int H, float s) {
	COLORREF* img = new COLORREF[W*H];
	for (int i = 0; i < W*H; i++) {
		float d = s * vals[i];
		vec3f col = d > 0. ? vec3f(.9f, .6f, .3f) : vec3f(.6f, .9f, 1.0f);
		col *= (1.0f - exp(-10.f*abs(d))) * (0.8f + 0.2f*cos(60.f*d));
		col = mix(col, vec3f(1.0f), clamp(1.0f - abs(20.0f*d), 0.0f, 1.0f));
		img[i] = toCOLORREF(col.zyx()) | 0xff000000;
	}
	stbi_write_png(filename, W, H, 4, img, 4 * W);
	delete img;
}

int main() {

	std::vector<triangle_3d_f> trigs;
	vec3f p0 = vec3f((float)INFINITY), p1 = -p0;
	//loadFile("D:\\isosphere.stl", trigs, p0, p1);
	//loadFile("D:\\blender_suzanne3.stl", trigs, p0, p1);
	//loadFile("D:\\stanford_dragon.ply", trigs, p0, p1);
	loadFile("D:\\Coding\\Github\\Graphics\\modeling\\volume\\ct_head.reduced.ply", trigs, p0, p1);
	//loadFile("D:\\Explore\\Thingi10K\\Thingi10K\\raw_meshes\\34784.stl", trigs, p0, p1);

	std::vector<segment_2d_f> segs = clip_triangle(trigs, mix(p0.z, p1.z, 0.5f + 1e-6f));
	printf("%d\n", (int)segs.size());

	const int W = 1024, H = 1024;
	vec2f r = 1.0f * vec2f(max(p1.x - p0.x, p1.y - p0.y)) * normalize(vec2f(W, H));
	vec2f q0 = 0.5f*(p0 + p1).xy() - r, q1 = q0 + 2.0f*r;
	float* vals = new float[W*H];
	for (int i = W * H; i--;) vals[i] = NAN;

	auto t0 = std::chrono::high_resolution_clock::now();
	//sdf_grid_bruteforce(segs, vals, q0, q1, ivec2(W, H));
	//sdf_grid_rasterize(segs, vals, q0, q1, ivec2(W, H));
	sdf_grid_expand(segs, vals, q0, q1, ivec2(W, H));
	float time_elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
	printf("%.1lfms\n", 1000.0f*time_elapsed);

	float s = 0.02f * length(vec2f(W, H)) / length(q1 - q0);
	save_image("D:\\.png", vals, W, H, s);

	delete vals;
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
	//for (int i = 0; i < VN; i++) Vs[i] = Vs[i].rz90().ry90().rz270();
	for (int i = 0; i < VN; i++)
		p0 = min(p0, Vs[i]), p1 = max(p1, Vs[i]);
	for (int i = 0; i < FN; i++)
		trigs.push_back(triangle_3d_f{ Vs[Fs[i][0]], Vs[Fs[i][1]], Vs[Fs[i][2]] });
	delete Vs; delete Fs;
	if (vcol) delete vcol; if (fcol) delete fcol;
	return true;
}

