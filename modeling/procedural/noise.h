#include "hash.h"

vec2f UnitVector2D(float u) {
	float a = 2.0f*PIf*u;
	return vec2f(cos(a), sin(a));
}

float ValueNoise1D(float x) {
	int i0 = (int)floor(x), i1 = i0 + 1;
	float v0 = hash11(float(i0)), v1 = hash11(float(i1));
	return 2.0f * mix(v0, v1, smoothstep(0.0f, 1.0f, x - float(i0))) - 1.0f;
}

float CosineNoise2D(vec2f xy) {
	return cos(PIf*xy.x) * cos(PIf*xy.y);
}

vec3f CosineNoise2Dg(vec2f xy) {
	float sinx = sin(PIf*xy.x), siny = sin(PIf*xy.y);
	float cosx = cos(PIf*xy.x), cosy = cos(PIf*xy.y);
	return vec3f(-PIf * sinx*cosy, -PIf * cosx*siny, cosx*cosy);
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

vec3f ValueNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float h00 = hash12(vec2f(i0, j0));
	float h01 = hash12(vec2f(i0, j1));
	float h10 = hash12(vec2f(i1, j0));
	float h11 = hash12(vec2f(i1, j1));
	vec2f xyf = xy - vec2f(i0, j0);
	vec2f intp = ((xyf * 6.0f - 15.0f) * xyf + 10.0f) * xyf * xyf * xyf;
	vec2f intpd = ((xyf * 30.0f - 60.0f) * xyf + 30.0f) * xyf * xyf;
	return vec3f(
		2.0f * mix(h10 - h00, h11 - h01, intp.y) * intpd.x,
		2.0f * mix(h01 - h00, h11 - h10, intp.x) * intpd.y,
		2.0f * mix(mix(h00, h01, intp.y), mix(h10, h11, intp.y), intp.x) - 1.0f
	);
}

float GradientNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float v00 = dot(2.0f * hash22(vec2f(i0, j0)) - 1.0f, xy - vec2f(i0, j0));
	float v01 = dot(2.0f * hash22(vec2f(i0, j1)) - 1.0f, xy - vec2f(i0, j1));
	float v10 = dot(2.0f * hash22(vec2f(i1, j0)) - 1.0f, xy - vec2f(i1, j0));
	float v11 = dot(2.0f * hash22(vec2f(i1, j1)) - 1.0f, xy - vec2f(i1, j1));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	//return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
	return v00 + (v10 - v00)*xf + (v01 - v00)*yf + (v00 + v11 - v01 - v10) * xf*yf;
}

vec3f GradientNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	vec2f g00 = 2.0f*hash22(vec2f(i0, j0)) - 1.0f;
	vec2f g01 = 2.0f*hash22(vec2f(i0, j1)) - 1.0f;
	vec2f g10 = 2.0f*hash22(vec2f(i1, j0)) - 1.0f;
	vec2f g11 = 2.0f*hash22(vec2f(i1, j1)) - 1.0f;
	float v00 = dot(g00, xy - vec2f(i0, j0));
	float v01 = dot(g01, xy - vec2f(i0, j1));
	float v10 = dot(g10, xy - vec2f(i1, j0));
	float v11 = dot(g11, xy - vec2f(i1, j1));
	vec2f xyf = xy - vec2f(i0, j0);
	vec2f intp = ((xyf * 6.0f - 15.0f) * xyf + 10.0f) * xyf * xyf * xyf;
	vec2f intpd = ((xyf * 30.0f - 60.0f) * xyf + 30.0f) * xyf * xyf;
	return vec3f(
		//g00 + ((g10 - g00)*intp.x + (v10 - v00)*vec2f(intpd.x, 0.0f)) + ((g01 - g00)*intp.y + (v01 - v00)*vec2f(0.0f, intpd.y))
		//+ (g00 + g11 - g01 - g10) * intp.x*intp.y + (v00 + v11 - v01 - v10) * intp.yx() * intpd,
		g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10)*intp.x*intp.y
		+ (vec2f(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx()) * intpd,
		v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
	);
}

float NormalizedGradientNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float v00 = dot(UnitVector2D(hash12(vec2f(i0, j0))), xy - vec2f(i0, j0));
	float v01 = dot(UnitVector2D(hash12(vec2f(i0, j1))), xy - vec2f(i0, j1));
	float v10 = dot(UnitVector2D(hash12(vec2f(i1, j0))), xy - vec2f(i1, j0));
	float v11 = dot(UnitVector2D(hash12(vec2f(i1, j1))), xy - vec2f(i1, j1));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3f NormalizedGradientNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	vec2f g00 = UnitVector2D(hash12(vec2f(i0, j0)));
	vec2f g01 = UnitVector2D(hash12(vec2f(i0, j1)));
	vec2f g10 = UnitVector2D(hash12(vec2f(i1, j0)));
	vec2f g11 = UnitVector2D(hash12(vec2f(i1, j1)));
	float v00 = dot(g00, xy - vec2f(i0, j0));
	float v01 = dot(g01, xy - vec2f(i0, j1));
	float v10 = dot(g10, xy - vec2f(i1, j0));
	float v11 = dot(g11, xy - vec2f(i1, j1));
	vec2f xyf = xy - vec2f(i0, j0);
	vec2f intp = ((xyf * 6.0f - 15.0f) * xyf + 10.0f) * xyf * xyf * xyf;
	vec2f intpd = ((xyf * 30.0f - 60.0f) * xyf + 30.0f) * xyf * xyf;
	return vec3f(
		g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10)*intp.x*intp.y
		+ (vec2f(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx()) * intpd,
		v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
	);
}

float WaveNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	const float freq = 2.0f;
	float v00 = sin(freq*dot(2.0f*hash22(vec2f(i0, j0)) - 1.0f, xy));
	float v01 = sin(freq*dot(2.0f*hash22(vec2f(i0, j1)) - 1.0f, xy));
	float v10 = sin(freq*dot(2.0f*hash22(vec2f(i1, j0)) - 1.0f, xy));
	float v11 = sin(freq*dot(2.0f*hash22(vec2f(i1, j1)) - 1.0f, xy));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3f WaveNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	vec2f n00 = 2.0f*hash22(vec2f(i0, j0)) - 1.0f;
	vec2f n01 = 2.0f*hash22(vec2f(i0, j1)) - 1.0f;
	vec2f n10 = 2.0f*hash22(vec2f(i1, j0)) - 1.0f;
	vec2f n11 = 2.0f*hash22(vec2f(i1, j1)) - 1.0f;
	const float freq = 2.0f;
	float v00 = sin(freq*dot(n00, xy));
	float v01 = sin(freq*dot(n01, xy));
	float v10 = sin(freq*dot(n10, xy));
	float v11 = sin(freq*dot(n11, xy));
	vec2f g00 = freq * cos(freq*dot(n00, xy)) * n00;
	vec2f g01 = freq * cos(freq*dot(n01, xy)) * n01;
	vec2f g10 = freq * cos(freq*dot(n10, xy)) * n10;
	vec2f g11 = freq * cos(freq*dot(n11, xy)) * n11;
	vec2f xyf = xy - vec2f(i0, j0);
	vec2f intp = ((xyf * 6.0f - 15.0f) * xyf + 10.0f) * xyf * xyf * xyf;
	vec2f intpd = ((xyf * 30.0f - 60.0f) * xyf + 30.0f) * xyf * xyf;
	return vec3f(
		g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10) * intp.x*intp.y
		+ (vec2f(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx()) * intpd,
		v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
	);
}

float NormalizedWaveNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	const float freq = 2.0f;
	float v00 = sin(freq*dot(UnitVector2D(hash12(vec2f(i0, j0))), xy));
	float v01 = sin(freq*dot(UnitVector2D(hash12(vec2f(i0, j1))), xy));
	float v10 = sin(freq*dot(UnitVector2D(hash12(vec2f(i1, j0))), xy));
	float v11 = sin(freq*dot(UnitVector2D(hash12(vec2f(i1, j1))), xy));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3f NormalizedWaveNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	vec2f n00 = UnitVector2D(hash12(vec2f(i0, j0)));
	vec2f n01 = UnitVector2D(hash12(vec2f(i0, j1)));
	vec2f n10 = UnitVector2D(hash12(vec2f(i1, j0)));
	vec2f n11 = UnitVector2D(hash12(vec2f(i1, j1)));
	const float freq = 2.0f;
	float v00 = sin(freq*dot(n00, xy));
	float v01 = sin(freq*dot(n01, xy));
	float v10 = sin(freq*dot(n10, xy));
	float v11 = sin(freq*dot(n11, xy));
	vec2f g00 = freq * cos(freq*dot(n00, xy)) * n00;
	vec2f g01 = freq * cos(freq*dot(n01, xy)) * n01;
	vec2f g10 = freq * cos(freq*dot(n10, xy)) * n10;
	vec2f g11 = freq * cos(freq*dot(n11, xy)) * n11;
	vec2f xyf = xy - vec2f(i0, j0);
	vec2f intp = ((xyf * 6.0f - 15.0f) * xyf + 10.0f) * xyf * xyf * xyf;
	vec2f intpd = ((xyf * 30.0f - 60.0f) * xyf + 30.0f) * xyf * xyf;
	return vec3f(
		g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10) * intp.x*intp.y
		+ (vec2f(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx()) * intpd,
		v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
	);
}

// Simplex noise: ???

