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

vec3 calcCol(vec3 ro, vec3 rd, uint32_t &seed, int recurse_remain = 20, bool hitsInside = false) {

	const bool directLightSample = true;

	// dome light
	const vec3 light = normalize(vec3(0, 0, 1));
	// light sphere
	const vec3 light_pos = vec3(2, 0, 1.2) + pow(rand01(seed), 2.)*vec3(0.1);  // with motion blur
	const double light_r = 0.5;
	const vec3 light_intensity = vec3(50., 30., 20.);

	vec3 m_col = vec3(1.), t_col = vec3(0.), col;  // convert recursion to iteration
	bool isInside = false;  // the ray is inside a "glass" or not
	vec3 n, min_n;
	bool bouncedFromSpecular = true;

	// "recursive" ray-tracing
	for (int iter = 0; iter < 20; iter++) {
		ro += 1e-6*rd;  // alternate of t>1e-6

		// intersect plane
		double min_t = -ro.z / rd.z;
		if (min_t > 0.) {
			vec2 p = ro.xy() + min_t * rd.xy();
			p *= 1.5;
			col = min(max(abs(p.x) - 1., abs(p.y) - .5), max(.5 - p.x, p.x + abs(p.y) - 1.5)) < 0 ? vec3(0.8) : vec3(0.6);
			min_n = vec3(0, 0, 1);
		}
		else {
			min_t = INFINITY;
			col = 0.1 * vec3(max(dot(rd, light), 0.));
		}

		// intersect light sphere
		double t = min_t;
		if (iter && intersectSphere(light_pos, light_r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = vec3(0.);
			col = light_intensity;
			//col *= 0.01;
		}

		// intersect scene
		t = min_t;
		if (intersectScene(Scene, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n;
			//col = vec3(0.5) + 0.5*n;
			col = isInside ? exp(-vec3(1., 1., 0.)*min_t) : vec3(1.);
		}

		// update ray
		m_col *= col;
		if (min_n == vec3(0.)) {  // hit light
			if ((!directLightSample)
				|| bouncedFromSpecular) {
				t_col += m_col;
			}
			break;
		}
		if (min_t == INFINITY) {  // hit skydome
			t_col += m_col;
			break;
		}
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		bouncedFromSpecular = abs(min_n.z) != 1.;
		//bouncedFromSpecular = false;  // specular scene: better use bidirectional path tracing :(
		if (!bouncedFromSpecular) {  // hit plane (diffuse)
			rd = randdir_cosWeighted(min_n, seed);
		}
		else {  // hit eyes (glass)
			if (isInside) {
				// subsurface scattering
				double scatter_d = -0.1*log(1. - rand01(seed));
				if (scatter_d < min_t) {
					ro = ro + rd * scatter_d;
					rd = rand3(seed);
					continue;
				}
			}
			rd = isInside ? randdir_Fresnel(rd, min_n, 1.5, 1.0, seed) : randdir_Fresnel(rd, min_n, 1.0, 1.5, seed);  // very likely that I have a bug
		}
		if (dot(rd, min_n) < 0.) {
			isInside ^= 1;  // reflected ray hits into the surface
		}

		// direct light sampling
		if (directLightSample && !bouncedFromSpecular) {
			vec3 pol = light_pos + light_r * rand3(seed);  // random point on light source
			vec3 dl = normalize(pol - ro);
			t = length(pol - ro);
			if (dot(dl, min_n) > 0. && !intersectScene(Scene, ro + 1e-6*dl, dl, t, n)) {  // can be optimized but I'm lazy
				// from https://www.shadertoy.com/view/4tl3z4, no idea how does it work
				double cos_a_max = sqrt(1. - clamp(light_r*light_r / (pol - ro).sqr(), 0., 1.));
				double weight = 2. * (1. - cos_a_max);
				t_col += (m_col * light_intensity) * (weight * clamp(dot(dl, min_n), 0., 1.));
			}
		}
	}

	// gamma
	double gamma = 0.8;
	t_col = vec3(pow(t_col.x, gamma), pow(t_col.y, gamma), pow(t_col.z, gamma));
	return t_col;

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
//const vec3 CamP = vec3(9.341668, -9.285457, 2.743465), ScrO = vec3(-0.913199, -3.038208, -0.834524), ScrA = vec3(4.519552, 4.302470, 0.000000), ScrB = vec3(-0.371965, 0.390732, 3.569464);
const vec3 CamP = vec3(-9.744200, -6.765993, 2.743465), ScrO = vec3(-2.489517, 3.025109, -1.318631), ScrA = vec3(4.489501, -6.539911, 0.000000), ScrB = vec3(0.565400, 0.388134, 4.537678);

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
			const int N = 1024;
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
