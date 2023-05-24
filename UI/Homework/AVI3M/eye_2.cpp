#include <stdio.h>
#include <vector>
#include <algorithm>
#include "numerical/geometry.h"
#include "bvh.h"
#include "brdf.h"

BVH *Scene;

vec3 calcCol(vec3 ro, vec3 rd, uint32_t &seed) {

	const vec3 light = normalize(vec3(0, 0, 1));
	vec3 m_col = vec3(2.), col;

	// "recursive" ray-tracing
	for (int iter = 0; iter < 20; iter++) {
		vec3 n, min_n;
		ro += 1e-6*rd;  // alternate of t>1e-6

		// intersect plane
		double min_t = -ro.z / rd.z;
		if (min_t > 0.) {
			vec2 p = ro.xy() + min_t * rd.xy();
			col = int(floor(p.x) + floor(p.y)) & 1 ? vec3(0.8) : vec3(0.6);
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
			col = vec3(0.5) + 0.5*n;
			//col = mix(vec3(.5, .2, .1), col, max(1.0 - exp(20.*(ro.x + t * rd.x - 0.5)), 0.8));  // u what
		}

		// update ray
		m_col *= col;
		if (min_t == INFINITY) {
			return m_col;
		}
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (abs(min_n.z) == 1) {
			rd = rd - min_n * (2.*dot(rd, min_n));
		}
		else {
			rd = randdir_cosWeighted(min_n, seed);
		}
	}
	return m_col;
}

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

const int W = 595 * 3, H = 361 * 3;
const vec3 CamP = vec3(10.021777, -3.745871, 1.106030), ScrO = vec3(-1.630352, -4.256079, -1.869166), ScrA = vec3(2.912566, 8.233708, 0.000000), ScrB = vec3(-0.149845, 0.053006, 5.296532);
typedef unsigned char byte;
struct rgba {
	byte r, g, b, a;
} IMG[H][W];

int main(int argc, char* argv[]) {
	// load model
	BVH_Triangle* Scene_Trigs; int Scene_N;
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
	for (int j = 0; j < H; j++) {
		for (int i = 0; i < W; i++) {
			const int N = 256;
			vec3 col(0.);
			for (int u = 0; u < N; u++) {
				uint32_t seed = hashu((u*W + i)*H + j);
				vec3 CamD = ScrO + ((i + rand01(seed)) / W)*ScrA + ((j + rand01(seed)) / H)*ScrB;
				col += calcCol(CamP, normalize(CamD - CamP), seed);
			}
			col /= N;
			rgba c;
			c.r = byte(255.99*clamp(col.x, 0., 1.));
			c.g = byte(255.99*clamp(col.y, 0., 1.));
			c.b = byte(255.99*clamp(col.z, 0., 1.));
			c.a = 255;
			IMG[H - 1 - j][i] = c;
		}
		printf("%d/%d\n", j + 1, H);  // progress line
	}

	// output
	stbi_write_png(argv[2], W, H, 4, &IMG[0][0], 4 * W);

	return 0;
}
