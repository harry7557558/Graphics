#include <stdio.h>
#include <vector>
#include <algorithm>
#include "numerical/geometry.h"
#include "bvh.h"
#include "brdf.h"


// subsurface scattering on/off
#define SSS true

BVH *Scene;

vec3 calcCol(vec3 ro, vec3 rd, uint32_t &seed) {

	const vec3 light = normalize(vec3(0, 0, 1));
	vec3 m_col = vec3(2.), col;

	bool isInside = false;  // the ray is inside a "glass" or not

	// "recursive" ray-tracing
	for (int iter = 0; iter < 20; iter++) {
		vec3 n, min_n;
		ro += 1e-6*rd;  // alternate of t>1e-6

		// intersect plane
		double min_t = -ro.z / rd.z;
		if (min_t > 0.) {
			vec2 p = ro.xy() + min_t * rd.xy();
			if (SSS) col = (max(abs(p.x) - 0.618, abs(p.y) - 1.) < 0. ? vec3(0.85, 0.75, 0.75) : vec3(0.65, 0.6, 0.55))*exp(-0.001*p.sqr());
			else col = min(max(abs(p.x) - .5, abs(p.y) - 2.), max(abs(p.x - .5) - 2., abs(p.y) - .5)) < 0. ? vec3(0.8) : vec3(0.6);
			min_n = vec3(0, 0, 1);
		}
		else {
			min_t = INFINITY;
			col = vec3(max(dot(rd, light), 0.));
		}

		// intersect scene
		double t = min_t;
		if (intersectScene(Scene, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n;
			col = isInside ? exp(-vec3(1., 1., 0.)*min_t) : vec3(1.);
		}

		// update ray
		m_col *= col;
		if (min_t == INFINITY) {
			return m_col;
		}
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (abs(min_n.z) == 1) {
			rd = randdir_cosWeighted(min_n, seed);
		}
		else {
#if SSS
			if (isInside) {
				double scatter_d = -0.1*log(1. - rand01(seed));
				if (scatter_d < min_t) {
					ro = ro + rd * scatter_d;
					rd = rand3(seed);
					continue;
				}
			}
#endif
			rd = isInside ? randdir_Fresnel(rd, min_n, 1.5, 1.0, seed) : randdir_Fresnel(rd, min_n, 1.0, 1.5, seed);  // very likely that I have a bug
		}
		if (dot(rd, min_n) < 0.) {
			isInside ^= 1;  // reflected ray hits into the surface
		}
	}
	return m_col;
}


#include <thread>
void Render_Exec(void(*task)(int, int, int, bool*), int Max) {
	const int MAX_THREADS = std::thread::hardware_concurrency();
	bool* fn = new bool[MAX_THREADS];
	std::thread** T = new std::thread*[MAX_THREADS];
	for (int i = 0; i < MAX_THREADS; i++) {
		fn[i] = false;
		T[i] = new std::thread(task, i, Max, MAX_THREADS, &fn[i]);
	}
	int count; do {
		using namespace std::chrono_literals;
		std::this_thread::sleep_for(100ms);
		count = 0;
		for (int i = 0; i < MAX_THREADS; i++) count += fn[i];
	} while (count < MAX_THREADS);
	delete fn; delete T;
}


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

#if SSS
const int W = 624 * 3, H = 361 * 3;
const vec3 CamP = vec3(8.254976, -7.445967, 2.012739), ScrO = vec3(-1.411040, -2.648941, -0.642686), ScrA = vec3(3.649288, 4.160781, 0.000000), ScrB = vec3(-0.240311, 0.210769, 3.185787);
//const int W = 580 * 3, H = 360 * 3;
//const vec3 CamP = vec3(5.504111, -3.846323, 1.608597), ScrO = vec3(-1.562658, -3.103406, -0.843286), ScrA = vec3(3.319202, 5.294210, 0.000000), ScrB = vec3(-0.304926, 0.191173, 3.586987);
#else
const int W = 624 * 3, H = 361 * 3;
const vec3 CamP = vec3(9.746397, -3.244285, 2.547210), ScrO = vec3(0.141163, -3.861328, -1.055587), ScrA = vec3(2.146265, 6.700219, 0.000000), ScrB = vec3(-0.655793, 0.210068, 4.011590);
#endif
typedef unsigned char byte;
struct rgba {
	byte r, g, b, a;
} IMG[H][W];

BVH_Triangle* Scene_Trigs; int Scene_N;
int main(int argc, char* argv[]) {
	// load model
	readBinarySTL(argv[1], Scene_Trigs, Scene_N);  // eye_full.stl

	vec3 p0, p1; BVH_BoundingBox(Scene_Trigs, Scene_N, p0, p1);
	for (int i = 0; i < Scene_N; i++) {
		Scene_Trigs[i].P.z -= p0.z;
	}

	// construct BVH
	std::vector<BVH_Triangle*> Scene_Tp;
	for (int i = 0; i < Scene_N; i++) Scene_Tp.push_back(&Scene_Trigs[i]);
	Scene = new BVH;
	vec3 Min(INFINITY), Max(-INFINITY);
	constructBVH(Scene, Scene_Tp, Min, Max);

	// rendering
	Render_Exec([](int beg, int end, int step, bool* sig) {
		const int WIN_SIZE = W * H;
		for (int k = beg; k < end; k += step) {
			int i = k % W, j = k / W;
			const int N = 256;
			vec3 col(0.);
			for (int u = 0; u < N; u++) {
				uint32_t seed = hashu((u*W + i)*H + j);
				vec3 CamD = ScrO + ((i + rand01(seed)) / W)*ScrA + ((j + rand01(seed)) / H)*ScrB;
				col += calcCol(CamP, normalize(CamD - CamP), seed);
			}
			col /= N;
			IMG[H - 1 - j][i] = rgba{
				byte(255.99*clamp(col.x, 0., 1.)),
				byte(255.99*clamp(col.y, 0., 1.)),
				byte(255.99*clamp(col.z, 0., 1.)),
				255 };
			if (beg == 0 && i == 0) printf("%.1lf%%\n", k * 100. / end);  // progress line
		}
		if (sig) *sig = true;
	}, W*H);

	// output
	stbi_write_png(argv[2], W, H, 4, &IMG[0][0], 4 * W);

	return 0;
}
