// calculate a 2D grid of signed distance samples from boundary defined by segments
// boundary must form a manifold

#include "numerical/geometry.h"
#include "numerical/random.h"
#include <vector>
#include <algorithm>


// bruteforce reference solution
float calc_sdf(const segment_2d_f segs[], int N, vec2f p) {
	float sd = (float)INFINITY;
	bool sgn = false;
	for (int i = 0; i < N; i++) {
		vec2f q = p - segs[i][0], r = segs[i][1] - segs[i][0];
		float h = dot(q, r) / dot(r, r);
		float d = (q - r * clamp(h, 0.0f, 1.0f)).sqr();
		sd = min(d, sd);
		if (q.x*(p.x - segs[i][1].x) < 0.0) sgn ^= det(q, r) * r.x < 0.0f;
	}
	return (sgn ? -1.0f : 1.0f) * sqrt(sd);
}
void sdf_grid_bruteforce(std::vector<segment_2d_f> segs,
	float* sdf, vec2f p0, vec2f p1, ivec2 dif) {

	for (int yi = 0; yi < dif.y; yi++) {
		float y = mix(p0.y, p1.y, yi / float(dif.y - 1));
		for (int xi = 0; xi < dif.x; xi++) {
			float x = mix(p0.x, p1.x, xi / float(dif.x - 1));
			sdf[yi*dif.x + xi] = calc_sdf(&segs[0], segs.size(), vec2f(x, y));
		}
	}
}



// looks pretty nice but I cannot guarantee this one will always work

void sdf_grid_rasterize(std::vector<segment_2d_f> segs,
	float* sdf, vec2f p0, vec2f p1, ivec2 dif) {

	// shuffle segment list to avoid worst case runtime
	int n = (int)segs.size();
	for (int i = n - 1; i > 0; i--)
		std::swap(segs[i], segs[randi(0, i + 1)]);

	// this buffer stores the index of the closest triangle
	int *indices = new int[dif.x*dif.y];
	for (int i = dif.x*dif.y; i--;) indices[i] = -1;

	// this stores the y-coordinates crossed by each line with equation x=ints[i]
	std::vector<float> *ints = new std::vector<float>[dif.x];

	// add each segment
	for (int id = 0; id < n; id++) {
		segment_2d_f s = segs[id];
		vec2f q0 = mix(vec2f(0), vec2f(dif - ivec2(1)), (s[0] - p0) / (p1 - p0));
		vec2f q1 = mix(vec2f(0), vec2f(dif - ivec2(1)), (s[1] - p0) / (p1 - p0));
		if (q0.x > q1.x) std::swap(q0, q1);

		// store a list of pixels to be searched
		std::vector<ivec2> ps;
		auto add_pixel = [&](int x, int y) -> bool {
			int i = y * dif.x + x;
			if (i < 0 || i > dif.x*dif.y || indices[i] == id) return false;
			if (x < 0 || x >= dif.x || y < 0 || y >= dif.y) return false;
			vec2f p = mix(p0, p1, vec2f(ivec2(x, y)) / vec2f(dif - ivec2(1)));
			if (indices[i] == -1) {
				indices[i] = id;
				sdf[i] = s.dist2(p);
				ps.push_back(ivec2(x, y));
				return true;
			}
			else {
				float d = s.dist2(p);
				if (d <= sdf[i]) {
					sdf[i] = d;
					indices[i] = id;
					ps.push_back(ivec2(x, y));
					return true;
				}
				return false;
			}
		};

		// initial pixels: y=m*x+b
		{
			float m = (q1.y - q0.y) / (q1.x - q0.x), b = q0.y - m * q0.x;
			int x0 = max((int)min(q0.x, q1.x), 0);
			int x1 = min((int)max(q0.x, q1.x) + 1, dif.x - 1);
			for (int x = x0; x <= x1; x++) {
				float y = m * (float)x + b;
				add_pixel(x, (int)y);
				add_pixel(x, (int)y + 1);
			}
		}
		// x=m*y+b
		{
			float m = (q1.x - q0.x) / (q1.y - q0.y), b = q0.x - m * q0.y;
			int y0 = max((int)min(q0.y, q1.y), 0);
			int y1 = min((int)max(q0.y, q1.y) + 1, dif.y - 1);
			for (int y = y0; y <= y1; y++) {
				float x = m * (float)y + b;
				add_pixel((int)x, y);
				add_pixel((int)x + 1, y);
			}
		}

		// expanding pixels, similar to BFS
		while (!ps.empty()) {
			std::vector<ivec2> ps1;
			for (int i = 0; i < (int)ps.size(); i++) {
				ivec2 p = ps[i];
				for (int u = -1; u <= 1; u++) for (int v = -1; v <= 1; v++) {
					if (u && v && add_pixel(p.x + u, p.y + v))
						ps1.push_back(ivec2(p.x + u, p.y + v));
				}
			}
			ps = ps1;
		}

		// working with the sign of the distance field
		float m = (q1.y - q0.y) / (q1.x - q0.x), b = q0.y - m * q0.x;
		int x0 = max((int)min(q0.x, q1.x), 0);
		int x1 = min((int)max(q0.x, q1.x) + 1, dif.x - 1);
		for (int x = x0; x <= x1; x++) if (x > q0.x && x < q1.x) {
			float y = m * (float)x + b;
			ints[x].push_back(y);
		}
	}
	for (int i = dif.x*dif.y; i--;) sdf[i] = sqrt(sdf[i]);

	// sign the distance field, require the shape to be a manifold
	for (int x = 0; x < dif.x; x++) {
		//printf("%d ", ints[x].size());
		std::sort(ints[x].begin(), ints[x].end());
		float sgn = 1.0; int yi = 0;
		for (int i = 0; i < (int)ints[x].size(); i++) {
			float y = ints[x][i];
			while (yi < y && yi < dif.y) {
				sdf[yi*dif.x + x] *= sgn;
				yi++;
			}
			sgn *= -1.0;
		}
		if (ints[x].size() % 2 == 0) {
			while (yi < dif.y) {
				sdf[yi*dif.x + x] *= sgn;
				yi++;
			}
		}
	}

	// clean-up
	delete indices;
	delete[] ints;
}
