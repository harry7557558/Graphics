// calculate a 3D voxel grid of signed distance samples from triangle mesh
// mesh must form a manifold

#include "numerical/geometry.h"
#include "numerical/random.h"
#include <vector>
#include <algorithm>


// distance from a point to a 3D triangle, slow
float triangle_dist2(vec3f p, vec3f a, vec3f b, vec3f c) {
	// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
	vec3f ba = b - a; vec3f pa = p - a;
	vec3f cb = c - b; vec3f pb = p - b;
	vec3f ac = a - c; vec3f pc = p - c;
	vec3f n = cross(ba, ac);
	return (
		sign(dot(cross(ba, n), pa)) +
		sign(dot(cross(cb, n), pb)) +
		sign(dot(cross(ac, n), pc)) < 2.0)
		? min(min(
		(ba*clamp(dot(ba, pa) / ba.sqr(), 0.0f, 1.0f) - pa).sqr(),
			(cb*clamp(dot(cb, pb) / cb.sqr(), 0.0f, 1.0f) - pb).sqr()),
			(ac*clamp(dot(ac, pc) / ac.sqr(), 0.0f, 1.0f) - pc).sqr())
		: dot(n, pa)*dot(n, pa) / n.sqr();
}

// test if a point is inside a triangle in 2D
bool in_triangle(vec2f p, vec2f A, vec2f B, vec2f C) {
	vec2f a = A - p, b = B - p, c = C - p;
	float ab = det(a, b), bc = det(b, c), ca = det(c, a);
	return (ab > 0 && bc > 0 && ca > 0) || (ab < 0 && bc < 0 && ca < 0);
}




// bruteforce reference solution
float calc_sdf(const triangle_3d_f trigs[], int N, vec3f p) {
	float sd = (float)INFINITY;
	float sgn = 1.0f;
	for (int i = 0; i < N; i++) {
		vec3f a = trigs[i][0], b = trigs[i][1], c = trigs[i][2];
		sd = min(triangle_dist2(p, a, b, c), sd);
		if (in_triangle(p.xy(), a.xy(), b.xy(), c.xy())) {
			vec3f ap = p - a, ab = b - a, ac = c - a;
			sgn *= sign(dot(cross(ab, ac), -ap) * det(ab.xy(), ac.xy()));
		}
	}
	return sgn * sqrt(sd);
}
void sdf_grid_bruteforce(std::vector<triangle_3d_f> trigs,
	float* sdf, vec3f p0, vec3f p1, ivec3 dif) {

	for (int zi = 0; zi < dif.z; zi++) {
		float z = mix(p0.z, p1.z, zi / float(dif.z - 1));
		for (int yi = 0; yi < dif.y; yi++) {
			float y = mix(p0.y, p1.y, yi / float(dif.y - 1));
			for (int xi = 0; xi < dif.x; xi++) {
				float x = mix(p0.x, p1.x, xi / float(dif.x - 1));
				sdf[(zi*dif.y + yi)*dif.x + xi] =
					calc_sdf(&trigs[0], (int)trigs.size(), vec3f(x, y, z));
			}
		}
	}
}




// this looks good but does not guarantee to work

void sdf_grid_expand(std::vector<triangle_3d_f> trigs,
	float* sdf, vec3f p0, vec3f p1, ivec3 dif) {

	// shuffle triangle list, not necessary
	int n = (int)trigs.size();
	if (0) {
		uint32_t seed = 0;
		for (int i = n - 1; i > 0; i--) {
			int random = int(rand01(seed) * (i + 1));
			std::swap(trigs[i], trigs[random]);
		}
	}

	// this buffer stores the index of the closest triangle
	int *indices = new int[dif.x*dif.y*dif.z];
	for (int i = dif.x*dif.y*dif.z; i--;) indices[i] = -1;

	// this stores the z-coordinates crossed by each vertical line
	std::vector<float> *ints = new std::vector<float>[dif.x*dif.y];

	// stores a list of pixels to be searched
	std::vector<ivec3> ps;


	auto add_voxel = [&](int id, int x, int y, int z, bool append) -> bool {
		int i = (z * dif.y + y) * dif.x + x;
		if (i < 0 || i > dif.x*dif.y*dif.z || indices[i] == id) return false;
		if (x < 0 || x >= dif.x || y < 0 || y >= dif.y || z < 0 || z >= dif.z) return false;
		vec3f p = mix(p0, p1, vec3f(ivec3(x, y, z)) / vec3f(dif - ivec3(1)));
		if (indices[i] == -1) {
			indices[i] = id;
			sdf[i] = triangle_dist2(p, trigs[id][0], trigs[id][1], trigs[id][2]);
			if (append) ps.push_back(ivec3(x, y, z));
			return true;
		}
		else {
			float d = triangle_dist2(p, trigs[id][0], trigs[id][1], trigs[id][2]);
			if (d < sdf[i]) {
				sdf[i] = d;
				indices[i] = id;
				if (append) ps.push_back(ivec3(x, y, z));
				return true;
			}
			return false;
		}
	};

	// add each segment
	for (int id = 0; id < n; id++) {
		triangle_3d_f s = trigs[id];
		vec3f a = mix(vec3f(0), vec3f(dif - ivec3(1)), (s[0] - p0) / (p1 - p0));
		vec3f b = mix(vec3f(0), vec3f(dif - ivec3(1)), (s[1] - p0) / (p1 - p0));
		vec3f c = mix(vec3f(0), vec3f(dif - ivec3(1)), (s[2] - p0) / (p1 - p0));

		// initial voxels: samples
		if (sqrt(length(cross(b - a, c - a))) < 10) {
			ivec3 pi = ivec3((a + b + c) / 3.0f + vec3f(0.5));
			for (int w = -1; w <= 1; w++) for (int v = -1; v <= 1; v++) for (int u = -1; u <= 1; u++) {
				add_voxel(id, pi.x + u, pi.y + v, pi.z + w, true);
			}
		}
		else {
			const int SN = 3;
			for (int ui = 0; ui <= SN; ui++) {
				float u = float(ui) / float(SN);
				for (int vi = 0; ui + vi <= SN; vi++) {
					float v = float(vi) / float(SN);
					vec3f p = a + u * (b - a) + v * (c - a);
					ivec3 pi = ivec3(p + vec3f(0.5));
					for (int w = -1; w <= 1; w++) for (int v = -1; v <= 1; v++) for (int u = -1; u <= 1; u++) {
						add_voxel(id, pi.x + u, pi.y + v, pi.z + w, true);
					}
				}
			}
		}

		// working with the sign of the distance field
		int x0 = max((int)min(min(a.x, b.x), c.x), 0);
		int x1 = min((int)max(max(a.x, b.x), c.x) + 1, dif.x - 1);
		int y0 = max((int)min(min(a.y, b.y), c.y), 0);
		int y1 = min((int)max(max(a.y, b.y), c.y) + 1, dif.y - 1);
		for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
			if (in_triangle(vec2f((float)x, (float)y), a.xy(), b.xy(), c.xy())) {
				vec3f pa = a - vec3f(x, y, 0.0f), ab = b - a, ac = c - a;
				ints[y*dif.x + x].push_back(dot(cross(ab, ac), pa) / det(ab.xy(), ac.xy()));
			}
		}

	}

	// expanding voxels, similar to BFS
	while (!ps.empty()) {
	//for (int i = 0; i < 5; i++) {
		std::vector<ivec3> ps1;
		for (int i = 0; i < (int)ps.size(); i++) {
			ivec3 p = ps[i];
			int id = indices[(p.z*dif.y + p.y)*dif.x + p.x];
			for (int u = -1; u <= 1; u++) for (int v = -1; v <= 1; v++) for (int w = -1; w <= 1; w++) {
				if (u && v && add_voxel(id, p.x + u, p.y + v, p.z + w, false))
					ps1.push_back(ivec3(p.x + u, p.y + v, p.z + w));
				// ps1 possibly contains duplicate points
			}
		}
		if (1) {
			// usually but not always faster
			ps = ps1;
		}
		else {
			if (!ps1.empty()) {
				std::sort(ps1.begin(), ps1.end(), [](ivec3 a, ivec3 b) {
					return a.x == b.x ? a.y == b.y ? a.z < b.z : a.y < b.y : a.x < b.x;
				});
				ps.clear(); ps.reserve(ps1.size());
				ps.push_back(ps1[0]);
				for (int i = 1; i < (int)ps1.size(); i++)
					if (ps1[i] != ps.back()) ps.push_back(ps1[i]);
			}
			else ps = ps1;
		}
		//printf("%d ", ps.size());
	}

	for (int i = dif.x*dif.y*dif.z; i--;) sdf[i] = sqrt(sdf[i]);

	// sign the distance field, require the shape to be a manifold
	for (int y = 0; y < dif.y; y++) {
		for (int x = 0; x < dif.x; x++) {
			int i = y * dif.x + x;
			std::sort(ints[i].begin(), ints[i].end());
			float sgn = 1.0; int zi = 0;
			for (int u = 0; u < (int)ints[i].size(); u++) {
				float z = ints[i][u];
				while (zi < z && zi < dif.z) {
					sdf[(zi*dif.y + y)*dif.x + x] *= sgn;
					zi++;
				}
				sgn *= -1.0;
			}
			while (zi < dif.z) {
				sdf[(zi*dif.y + y)*dif.x + x] *= sgn;
				zi++;
			}
		}
	}

	// clean-up
	delete indices;
	delete[] ints;
}
