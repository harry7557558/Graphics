#include <vector>
#include <functional>
#include "noise.h"
#include "UI/stl_encoder.h"


float Fbm1(float(*noise)(vec2f), vec2f xy) {
	float h = 0.0f;
	float ampl = 1.0f;
	for (int k = 0; k < 8; k++) {
		h += ampl * noise(xy);
		ampl *= 0.5f;
		xy = 2.0f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * xy + vec2f(2.0);
	}
	return h;
}

float Fbm2(float(*noise)(vec2f), vec2f xy) {
	float h = 0.0;
	float ampl = 1.0;
	for (int i = 0; i < 8; i++) {
		const float eps = 0.01f;
		float val = noise(xy);
		vec2f grad = vec2f(noise(xy + vec2f(eps, 0.f)) - noise(xy - vec2f(eps, 0.f)), noise(xy + vec2f(0.f, eps)) - noise(xy - vec2f(0.f, eps))) / (2.0f*eps);
		h += ampl * val / (1.0f + grad.sqr());
		ampl *= 0.5;
		xy = 2.0f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * xy + vec2f(2.0);
	}
	return h;
}


void DiscretizeSquare(
	std::function<float(vec2f)> fun, vec2f b0, vec2f b1, ivec2 dif,
	std::vector<vec3f> &points, std::vector<ivec3> &trigs) {

	const int ti0 = (int)points.size();
	points.reserve(points.size() + (dif.x + 1)*(dif.y + 1));
	for (int yi = 0; yi <= dif.y; yi++) {
		for (int xi = 0; xi <= dif.x; xi++) {
			vec2f p = mix(b0, b1, vec2f((float)xi, (float)yi) / vec2f(dif));
			float v = fun(p);
			points.push_back(vec3f(p, v));
		}
	}

	trigs.reserve(trigs.size() + 2 * dif.x*dif.y);
	int row_size = dif.x + 1;
	for (int yi = 0; yi < dif.y; yi++) {
		for (int xi = 0; xi < dif.x; xi++) {
			trigs.push_back(ivec3(yi * row_size + xi, yi * row_size + (xi + 1), (yi + 1) * row_size + xi));
			trigs.push_back(ivec3((yi + 1) * row_size + (xi + 1), (yi + 1) * row_size + xi, yi * row_size + (xi + 1)));
		}
	}

}

int main(int argc, char* argv[]) {
	std::vector<vec3f> points;
	std::vector<ivec3> trigs;
	DiscretizeSquare([](vec2f p)->float {
		float v = Fbm2(ValueNoise2D, p);
		//v += 0.2 * (p.x - 3.0);
		//return v;
		return max(v - 1.0f, 0.0f);
		//return max(v - 0.8f, 0.0f);
	}, vec2f(0.0), vec2f(6, 4), 10 * ivec2(60, 40), points, trigs);
	WritePLY("D:\\.ply", (float*)&points[0], (int)points.size(), (int*)&trigs[0], (int)trigs.size());
	return 0;
}
