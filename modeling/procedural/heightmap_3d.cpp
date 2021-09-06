#include <vector>
#include <functional>
#include "noise.h"
#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"
#include <chrono>


// Numerical gradient
vec3f nGrad(float(*fun)(vec3f), vec3f xy) {
	const float eps = 0.005f;
	return vec3f(
		fun(xy + vec3f(eps, 0.f, 0.f)) - fun(xy - vec3f(eps, 0.f, 0.f)),
		fun(xy + vec3f(0.f, eps, 0.f)) - fun(xy - vec3f(0.f, eps, 0.f)),
		fun(xy + vec3f(0.f, 0.f, eps)) - fun(xy - vec3f(0.f, 0.f, eps))) / (2.0f*eps);
}


// FBM functions

// Simple FBM
float Fbm1(float(*noise)(vec3f), vec3f p, int n = 8) {
	float h = 0.0f;
	float ampl = 1.0f;
	for (int k = 0; k < n; k++) {
		ampl *= 0.5f;
		h += ampl * noise(p);
		p = 2.0f * mat3f(0.8f, 0.6f, 0.0f, -0.36f, 0.48f, -0.8f, -0.48f, 0.64f, 0.6f) * p + 2.0;
	}
	return h;
}



// Scenes

// Pure FBM / Debug
float Scene0(vec3f p) {
	//return SimplexValueNoise3D(vec3f(p.xy(), 1.0f)) + p.z;
	return SimplexValueNoise3D(p) + 2.0f*SimplexValueNoise2D(0.5f*p.xy());
	return Fbm1(SimplexValueNoise3D, p, 1) + 0.0f * p.z;
	return ValueNoise3D(p) + 0.0f * p.z;
}


// Visualization


int main(int argc, char* argv[]) {

	auto t0 = std::chrono::high_resolution_clock::now();
	std::vector<triangle_3d> trigs = ScalarFieldTriangulator_octatree::octatree([](vec3 p) {
		return (double)Scene0(vec3f(p)); }, vec3(-3, -3, -1), vec3(3, 3, 1), 24 * ivec3(3, 3, 1), 2);
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("%f secs elapsed\n", std::chrono::duration<float>(t1 - t0).count());

	writeSTL("D:\\.stl", &trigs[0], (int)trigs.size());
	return 0;
}
