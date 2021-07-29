#include "hash.h"

float ValueNoise1D(float x) {
	int i0 = (int)floor(x), i1 = i0 + 1;
	float v0 = hash11(float(i0)), v1 = hash11(float(i1));
	return mix(v0, v1, smoothstep(0.0f, 1.0f, x - float(i0)));
}

float CosineNoise2D(vec2f xy) {
	return 0.5f + 0.5f * cos(3.1415926f*xy.x) * cos(3.1415926f*xy.y);
}

float ValueNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float xf = smoothstep(0.0f, 1.0f, xy.x - i0);
	float yf = smoothstep(0.0f, 1.0f, xy.y - j0);
	return mix(
		mix(hash12(vec2f(i0, j0)), hash12(vec2f(i0, j1)), yf),
		mix(hash12(vec2f(i1, j0)), hash12(vec2f(i1, j1)), yf),
		xf);
}

