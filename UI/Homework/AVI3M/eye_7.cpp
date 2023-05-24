#include <stdio.h>
#include <vector>
#include <algorithm>
#include "numerical/geometry.h"
#include "bvh.h"
#include "brdf.h"

BVH *Scene = 0;

// intersection between ray and sphere
bool intersectSphere(vec3 O, double r, vec3 ro, vec3 rd, double &t, vec3 &n) {
	ro -= O;
	double b = -dot(ro, rd), c = dot(ro, ro) - r * r;
	double delta = b * b - c;
	if (delta < 0.0) return false;
	delta = sqrt(delta);
	double t1 = b - delta, t2 = b + delta;
	if (t1 > t2) std::swap(t1, t2);
	if (t1 > t || t2 < 0.) return false;
	t = t1 > 0. ? t1 : t2;
	n = normalize(ro + rd * t);
	return true;
}

vec3 calcCol(vec3 ro, vec3 rd, uint32_t &seed) {

	const vec3 light = normalize(vec3(0, 0, 1));
	vec3 m_col = vec3(1.5), col;

	bool isInside = false;  // the ray is inside a "glass" or not

	// "recursive" ray-tracing
	for (int iter = 0; iter < 20; iter++) {
		vec3 n, min_n;
		ro += 1e-6*rd;  // alternate of t>1e-6

		int intersect_id = -1;  // which object the ray hits

		// intersect plane
		double min_t = -ro.z / rd.z;
		if (min_t > 0.) {
			vec2 p = ro.xy() + min_t * rd.xy();
			col = int(floor(p.x) + floor(p.y)) & 1 ? vec3(0.8) : vec3(0.6);
			min_n = vec3(0, 0, 1);
			intersect_id = 0;
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
			col = vec3(0.8);
		}

		// update ray
		m_col *= col;
		if (min_t == INFINITY) {
			return m_col;
		}
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (abs(min_n.z) == 1.) {
			//rd = rd - min_n * (2.*dot(rd, min_n));
			rd = randdir_cosWeighted(min_n, seed);

		}
		else {
			rd = rd - min_n * (2.*dot(rd, min_n));
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

const int W = 624 * 3, H = 361 * 3;


typedef unsigned char byte;
struct rgba {
	byte r, g, b, a;
} IMG[H][W];


BVH_Triangle *STL;
int STL_N;

int main(int argc, char* argv[]) {
	// load models
	auto loadObject = [](const char filename[], BVH* &R) {
		readBinarySTL(filename, STL, STL_N);
		for (int i = 0; i < STL_N; i++) {
			STL[i].P.z += .5;
		}
		std::vector<BVH_Triangle*> T;
		for (int i = 0; i < STL_N; i++) T.push_back(&STL[i]);
		R = new BVH;
		vec3 Min(INFINITY), Max(-INFINITY);
		constructBVH(R, T, Min, Max);
	};
	loadObject(argv[1], Scene);  // eye_full.stl

	// rendering
	Render_Exec([](int beg, int end, int step, bool* sig) {
		const int WIN_SIZE = W * H;
		for (int k = beg; k < end; k += step) {
			int i = k % W, j = k / W;
			const int N = 256;
			vec3 col(0.);
			for (int u = 0; u < N; u++) {
				uint32_t seed = hashu((u*W + i)*H + j);
				if (1) {
					const vec3 CamP = vec3(2.422877, -4.215974, 0.759319), ScrO = vec3(-4.359890, -1.855400, -1.121128), ScrA = vec3(6.311185, 4.805726, 0.000000), ScrB = vec3(0.194458, -0.255374, 4.577970);
					vec3 CamD = ScrO + ((i + rand01(seed)) / W)*ScrA + ((j + rand01(seed)) / H)*ScrB;
					col += calcCol(CamP, normalize(CamD - CamP), seed);
				}
				else {
					const vec3 CamP = vec3(7.560180, -9.952623, 2.862026), ScrO = vec3(-1.625106, -2.272841, -0.831738), ScrA = vec3(5.111755, 3.578765, 0.000000), ScrB = vec3(-0.329854, 0.471149, 3.563890);
					vec3 CamD = ScrO + ((i + rand01(seed)) / W)*ScrA + ((j + rand01(seed)) / H)*ScrB;
					vec2 uv = 0.05*cossin(2.*PI*rand01(seed));
					vec3 pos = CamP + uv.x*normalize(ScrA) + uv.y*normalize(ScrB);
					col += calcCol(pos, normalize(CamD - pos), seed);
				}
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
