#include "modeling/procedural/noise.h"
#include "raytracing/variance/variance.h"
#include <stdio.h>

void noiseStat2D(float (*noise)(vec2f)) {
	const int N = 2 << 22;

	VarianceObject<float> varobj;
	float maxv = 0.0, minv = 0.0;

	for (int i = 0; i < N; i++) {
		vec2f uv = hash21(float(i));
		vec2f xy = 1e+6 * (0.5f*uv + 0.5f);
		float v = noise(xy);
		varobj.addElement(v);
		maxv = max(maxv, v), minv = min(minv, v);
	}

	printf("mu  = %.3f\n", varobj.getMean());
	printf("var = %.3f\n", varobj.getVariance());
	printf("max = %.3f\n", maxv);
	printf("min = %.3f\n", minv);

	// Name - Variance - Max
	// CosineNoise2D - 0.249 (1/4) - 1.000 (1)
	// ValueNoise2D - 0.206 (32761/160083) - 0.998 (1)
	// GradientNoise2D - 0.030 (193670/6243237) - 0.790
	// NormalizedGradientNoise2D - 0.023 - 0.545
	// WaveNoise2D - 0.303 - 1.000 (1)
	// NormalizedWaveNoise2D - 0.305 - 1.000 (1)
	// SimplexNoise2D - 0.020 - 0.432
	// NormalizedSimplexNoise2D - 0.015 - 0.313
}

void noiseStat2Dg(vec3f (*noiseg)(vec2f)) {
	// off-diagonal terms of covariance should be zero
	const int N = 2 << 22;

	VarianceObject<vec3f> varobj;

	for (int i = 0; i < N; i++) {
		vec2f uv = hash21(float(i));
		vec2f xy = 1e+6 * (0.5f*uv + 0.5f);
		vec3f s = noiseg(xy);
		varobj.addElement(s);
	}

	vec3f mu = varobj.getMean();
	vec3f var = varobj.getVariance();
	printf("mu  = (%.3f, %.3f, %.3f)\n", mu.x, mu.y, mu.z);
	printf("var = (%.3f, %.3f, %.3f)\n", var.x, var.y, var.z);

	// Name - Variance
	// CosineNoise2Dg - (2.387, 2.414, 0.249) (pi^2/4, pi^2/4, 1/4)
	// ValueNoise2Dg - (0.740, 0.750, 0.206) (3620/4851, 3620/4851, 32761/160083)
	// GradientNoise2Dg - (0.262, 0.260, 0.030)
	// NormalizedGradientNoise2Dg - (0.196, 0.196, 0.023)
	// WaveNoise2Dg - (1.495, 1.511, 0.303)
	// NormalizedWaveNoise2Dg - (1.404, 1.397, 0.305)
	// SimplexNoise2Dg - (0.461, 0.466, 0.020)
	// NormalizedSimplexNoise2Dg - (0.348, 0.346, 0.015)
}



int main(int argc, char* argv[]) {

	noiseStat2Dg(NormalizedSimplexNoise2Dg);

	return 0;
}
