#include "hash.h"

vec2f UnitVector2D(float u) {
	float a = 2.0f*3.1415926f*u;
	return vec2f(cos(a), sin(a));
}

float ValueNoise1D(float x) {
	int i0 = (int)floor(x), i1 = i0 + 1;
	float v0 = hash11(float(i0)), v1 = hash11(float(i1));
	return 2.0f * mix(v0, v1, smoothstep(0.0f, 1.0f, x - float(i0))) - 1.0f;
}

float CosineNoise2D(vec2f xy) {
	return cos(3.1415926f*xy.x) * cos(3.1415926f*xy.y);
}

float ValueNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	return 2.0f * mix(
		mix(hash12(vec2f(i0, j0)), hash12(vec2f(i0, j1)), yf),
		mix(hash12(vec2f(i1, j0)), hash12(vec2f(i1, j1)), yf),
		xf) - 1.0f;
}

float GradientNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float v00 = dot(UnitVector2D(hash12(vec2f(i0, j0))), xy - vec2f(i0, j0));
	float v01 = dot(UnitVector2D(hash12(vec2f(i0, j1))), xy - vec2f(i0, j1));
	float v10 = dot(UnitVector2D(hash12(vec2f(i1, j0))), xy - vec2f(i1, j0));
	float v11 = dot(UnitVector2D(hash12(vec2f(i1, j1))), xy - vec2f(i1, j1));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	float v = mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
	return v * 1.414214f;
}
