#include <stdio.h>
#include <vector>
#include <algorithm>
#include "numerical/geometry.h"
#include "bvh.h"
#include "brdf.h"

BVH *Eyeball = 0, *Eyelash = 0, *Iris = 0;

vec3 calcCol(vec3 ro, vec3 rd, uint32_t &seed) {

	const vec3 light = normalize(vec3(0, 0, 1));
	vec3 m_col = vec3(2.), col;

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
			col = max(abs(p.x) - 0.618, abs(p.y) - 1.) < 0. ? vec3(0.8) : vec3(0.6);
			col *= exp(-0.01*p.sqr());
			min_n = vec3(0, 0, 1);
			intersect_id = 0;
		}
		else {
			min_t = INFINITY;
			col = vec3(max(dot(rd, light), 0.));
		}

		// intersect scene
		double t = min_t;
		if (intersectScene(Eyeball, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n;
			col = isInside ? exp(-vec3(0., 0., 1.)*min_t) : vec3(1.);
			intersect_id = 1;
		}
		if (intersectScene(Eyelash, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n;
			col = isInside ? exp(-vec3(2., 5., 0.)*min_t) : vec3(1.);
			intersect_id = 2;
		}
		if (intersectScene(Iris, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n;
			col = isInside ? exp(-vec3(10., 10., 0.)*min_t) : vec3(1.);  // note that there is a "bug" there since the iris is inside the eyeball
			intersect_id = 3;
		}

		// update ray
		m_col *= col;
		if (min_t == INFINITY) {
			return m_col;
		}
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (intersect_id == 0) {
			//rd = rd - min_n * (2.*dot(rd, min_n));
			rd = randdir_cosWeighted(min_n, seed);
		}
		else {
			//rd = rd - min_n * (2.*dot(rd, min_n));
			//rd = randdir_cosWeighted(min_n, seed);
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

#if 1
const int W = 624 * 3, H = 361 * 3;
const vec3 CamP = vec3(10.204190, 5.840131, 3.498755), ScrO = vec3(1.768514, -2.323521, -0.881495), ScrA = vec3(-2.991615, 5.476115, 0.000000), ScrB = vec3(-0.691367, -0.377696, 3.522990);
#else
// un-comment line 29
const int W = 624 * 3, H = 361 * 3;
const vec3 CamP = vec3(11.145052, 3.384808, 4.084967), ScrO = vec3(1.308189, -2.690256, -0.744334), ScrA = vec3(-1.724459, 5.996986, 0.000000), ScrB = vec3(-0.891918, -0.256475, 3.488668);
#endif
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
	loadObject(argv[1], Eyeball);  // eye_eyeball.stl
	loadObject(argv[2], Eyelash);  // eye_lash.stl
	loadObject(argv[3], Iris);  // eye_iris.stl

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
	stbi_write_png(argv[4], W, H, 4, &IMG[0][0], 4 * W);

	return 0;
}
