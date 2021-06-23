// a model to test rasterization rendering

#include "numerical/geometry.h"
#include "ui/stl_encoder.h"
#include "ui/debug_font_rendering.h"
#include "triangulate/octatree.h"

float smax(float d1, float d2, float k) {
	float h = clamp(0.5f - 0.5f*(d2 - d1) / k, 0.0f, 1.0f);
	return mix(d2, d1, h) + k * h*(1.0f - h);
}

float sd_glyph(char c1, char c2, vec3f p) {
	float d1 = sdFont(c1, p.x, 1.0f - p.z, 1);
	float d2 = sdFont(c2, p.y, 1.0f - p.z, 1);
	return smax(d1, d2, 0.01f);
}

int main(int argc, char* argv[]) {

	std::vector<triangle_3d> Trigs, trigs;

	for (char c1 = 'A'; c1 <= 'Z'; c1++) {
		for (char c2 = 'A'; c2 <= 'Z'; c2++) {
			vec3 dp = vec3(c1 - 'A', c2 - 'A', 0);
			trigs = ScalarFieldTriangulator_octatree::marching_cube([&](vec3 p) {
				return (double)sd_glyph(c1, c2, vec3f(p - dp));
			}, dp + vec3(0), dp + vec3(1), ivec3(32));
			Trigs.insert(Trigs.end(), trigs.begin(), trigs.end());
		}
	}

	writeSTL(argv[1], &Trigs[0], (int)Trigs.size());
	return 0;
}
