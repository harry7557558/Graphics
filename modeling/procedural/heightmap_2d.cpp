#include <vector>
#include <functional>
#include "noise.h"
#include "UI/stl_encoder.h"
#include <chrono>


// Numerical gradient
vec2f nGrad(float(*fun)(vec2f), vec2f xy) {
	const float eps = 0.005f;
	return vec2f(
		fun(xy + vec2f(eps, 0.f)) - fun(xy - vec2f(eps, 0.f)),
		fun(xy + vec2f(0.f, eps)) - fun(xy - vec2f(0.f, eps))) / (2.0f*eps);
}


// FBM functions

// Simple FBM
float Fbm1(float(*noise)(vec2f), vec2f xy, int n = 8) {
	float h = 0.0f;
	float ampl = 1.0f;
	for (int k = 0; k < n; k++) {
		ampl *= 0.5f;
		h += ampl * noise(xy);
		xy = 2.0f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * xy + vec2f(2.0);
	}
	return h;
}

// Simple FBM with gradient
vec3f Fbm1g(vec3f(*noiseg)(vec2f), vec2f xy, int n = 8) {
	vec3f h = vec3f(0.0f);
	float ampl = 1.0f;
	for (int k = 0; k < n; k++) {
		ampl *= 0.5f;
		h += ampl * noiseg(xy);
		xy = 2.0f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * xy + vec2f(2.0);
	}
	return h;
}

// Terrain with faked (poor?) erosion
float Fbm2(float(*noise)(vec2f), vec2f xy, int n = 8) {
	float h = 0.0f;
	float ampl = 1.0f;
	vec2f sum_grad(0.0f);
	for (int i = 0; i < n; i++) {
		const float eps = 0.01f;
		float val = noise(xy);
		sum_grad += nGrad(noise, xy);
		ampl *= 0.5f;
		h += ampl * val / (1.0f + sum_grad.sqr());
		xy = 2.0f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * xy + vec2f(2.0);
	}
	return h;
}

// Terrain with faked erosion, uses noise with analytical gradient
float Fbm2g(vec3f(*noiseg)(vec2f), vec2f xy, int n = 8) {
	float h = 0.0f;
	float ampl = 1.0f;
	vec2f sum_grad(0.0f);
	for (int i = 0; i < n; i++) {
		vec3f gradval = noiseg(xy);
		sum_grad += gradval.xy();
		ampl *= 0.5f;
		h += ampl * gradval.z / (1.0f + sum_grad.sqr());
		xy = 2.0f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * xy + vec2f(2.0);
	}
	return h;
}

// Water??!
float Fbm3(float(*noise)(vec2f), vec2f xy, int n = 4) {
	const float vt = 9.75f;
	float h = 0.0f;
	vec2f p = xy + vec2f(vt);
	float ampl = 1.0f;
	for (int k = 0; k < n; k++) {
		ampl *= 0.25f;
		h += ampl * noise(p);
		p = 1.9f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * p + vec2f(2.0);
	}
	p = xy - vec2f(vt);
	ampl = 1.0f;
	for (int k = 0; k < n; k++) {
		ampl *= 0.25f;
		h += ampl * noise(p);
		p = 1.9f * mat2f(0.6f, -0.8f, 0.8f, 0.6f) * p + vec2f(2.0);
	}
	return 0.5f * h;
}

// Swirling fluid ??
float Effect1(vec3f(*noiseg)(vec2f), vec2f xy, int n = 4) {
	for (int i = 0; i < n; i++) {
		xy += Fbm1g(noiseg, xy + 100.0f, 3).xy().rot() / float(n);
	}
	return Fbm1g(noiseg, xy, 5).z;
}


// Scenes

// Pure FBM / Debug
float Scene0(vec2f p) {
	return SimplexNoise2D(p);
	//return nGrad(NormalizedSimplexNoise2D, p).y - NormalizedSimplexNoise2Dg(p).y;
	//return Fbm2(SimplexNoise2D, p, 1) - Fbm2g(SimplexNoise2Dg, p, 1);
	//return Fbm2g(ValueNoise2Dg, p);
	//return Fbm2g(NormalizedGradientNoise2Dg, p);
	return 2.0f*Fbm2g(NormalizedSimplexNoise2Dg, 0.5f*p);
	//return Fbm1(ValueNoise2D, p, 8);
	//return Fbm1(ValueNoise2D, p, 3) - Fbm1(ValueNoise2D, p - 0.05f, 3);
	return Effect1(ValueNoise2Dg, p, 4);
	return Effect1(NormalizedSimplexNoise2Dg, p, 4);
}

// Terrain and flat water
float Scene1(vec2f p) {
	//float v = Fbm2g(ValueNoise2Dg, p, 8);
	float v = 2.0f*Fbm2g(SimplexNoise2Dg, 0.5f*p, 8);
	return max(v, 0.0f);
}

// Terrain and organic water
float Scene2(vec2f p) {
	float vt = Fbm2g(ValueNoise2Dg, p);
	float vw = Fbm3(GradientNoise2D, 10.0f * p) / 20.0f;
	return max(vt, vw);
}


// Visualization

void DiscretizeSquare(
	float(*fun)(vec2f), vec2f b0, vec2f b1, ivec2 dif,
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

	auto t0 = std::chrono::high_resolution_clock::now();
	DiscretizeSquare(Scene1, vec2f(0.0), 1.0f*vec2f(6, 4), ivec2(600, 400), points, trigs);
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("%f secs elapsed\n", std::chrono::duration<float>(t1 - t0).count());

	WritePLY("D:\\.ply", (float*)&points[0], (int)points.size(), (int*)&trigs[0], (int)trigs.size());
	return 0;
}
