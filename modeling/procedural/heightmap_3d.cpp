#include <vector>
#include <functional>
#include "noise.h"
#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"
#include <chrono>


// Numerical gradient
vec3f nGrad(float(*fun)(vec3f), vec3f p, float eps) {
	return vec3f(
		fun(p + vec3f(eps, 0.f, 0.f)) - fun(p - vec3f(eps, 0.f, 0.f)),
		fun(p + vec3f(0.f, eps, 0.f)) - fun(p - vec3f(0.f, eps, 0.f)),
		fun(p + vec3f(0.f, 0.f, eps)) - fun(p - vec3f(0.f, 0.f, eps))) / (2.0f*eps);
}


// test the correctness of analytical gradient
void RandomGradientTest(float(*noise)(vec3f), vec4f(*noiseg)(vec3f), int N) {
	const float eps = 0.01f, tol = 0.005f;
	int val_correct_count = 0, grad_correct_count = 0, nan_count = 0;
	float valdiff_mean = 0.0f, graddiff_mean = 0.0f;
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++) {
		vec3f p = 100.0f*(2.0f*hash31((float)i) - 1.0f);
		float val = noise(p);
		vec3f grad = nGrad(noise, p, eps);
		vec4f gradval = noiseg(p);
		float val_dif = abs(val - gradval.w);
		float grad_dif = length(grad - gradval.xyz());
		//printf("%f \n", grad_dif);
		if (val_dif < tol) val_correct_count++;
		if (grad_dif < tol) grad_correct_count++;
		if (isnan(val_dif*grad_dif)) nan_count++;
		valdiff_mean += val_dif / N;
		graddiff_mean += grad_dif / N;
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("%f secs elapsed\n\n", std::chrono::duration<float>(t1 - t0).count());
	printf("%d/%d vals correct\n%d/%d grads correct\n%d/%d nan\n\n",
		val_correct_count, N, grad_correct_count, N, nan_count, N);
	printf("valdiff mean %g\ngraddiff mean %g\n\n",
		valdiff_mean, graddiff_mean);
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
	return SimplexValueNoise3D(vec3f(p.xy(), 0.8f)) + p.z;
	//return SimplexValueNoise3D(p) + 2.0f*SimplexValueNoise2D(0.5f*p.xy());
	return Fbm1(SimplexValueNoise3D, p, 1) + 0.0f * p.z;
	return ValueNoise3D(p) + 0.0f * p.z;
}


// Visualization


int main(int argc, char* argv[]) {
	//RandomGradientTest(SimplexValueNoise3D, SimplexValueNoise3Dg, 0x40000); exit(0);

	auto t0 = std::chrono::high_resolution_clock::now();
	std::vector<triangle_3d> trigs = ScalarFieldTriangulator_octatree::octatree([](vec3 p) {
		return (double)Scene0(vec3f(p)); }, vec3(-3, -3, -1), vec3(3, 3, 1), 24 * ivec3(3, 3, 1), 2);
	auto t1 = std::chrono::high_resolution_clock::now();
	printf("%f secs elapsed\n", std::chrono::duration<float>(t1 - t0).count());

	writeSTL("D:\\.stl", &trigs[0], (int)trigs.size());
	return 0;
}
