#include "hash.h"

// return vectors uniformly distributed inside an unit circle from hash uv
vec2f uniform_unit_circle(vec2f uv) {
	float a = 2.0f*PIf*uv.x;
	return sqrt(uv.y)*vec2f(cos(a), sin(a));
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
		g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10)*intp.x*intp.y
		+ (vec2f(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx()) * intpd,
		v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
	);
}

float NormalizedGradientNoise2D(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	float v00 = dot(uniform_unit_circle(hash22(vec2f(i0, j0))), xy - vec2f(i0, j0));
	float v01 = dot(uniform_unit_circle(hash22(vec2f(i0, j1))), xy - vec2f(i0, j1));
	float v10 = dot(uniform_unit_circle(hash22(vec2f(i1, j0))), xy - vec2f(i1, j0));
	float v11 = dot(uniform_unit_circle(hash22(vec2f(i1, j1))), xy - vec2f(i1, j1));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3f NormalizedGradientNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	vec2f g00 = uniform_unit_circle(hash22(vec2f(i0, j0)));
	vec2f g01 = uniform_unit_circle(hash22(vec2f(i0, j1)));
	vec2f g10 = uniform_unit_circle(hash22(vec2f(i1, j0)));
	vec2f g11 = uniform_unit_circle(hash22(vec2f(i1, j1)));
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
	float v00 = sin(freq*dot(uniform_unit_circle(hash22(vec2f(i0, j0))), xy));
	float v01 = sin(freq*dot(uniform_unit_circle(hash22(vec2f(i0, j1))), xy));
	float v10 = sin(freq*dot(uniform_unit_circle(hash22(vec2f(i1, j0))), xy));
	float v11 = sin(freq*dot(uniform_unit_circle(hash22(vec2f(i1, j1))), xy));
	float xf = xy.x - i0; xf = xf * xf * xf * (10.0f + xf * (-15.0f + xf * 6.0f));
	float yf = xy.y - j0; yf = yf * yf * yf * (10.0f + yf * (-15.0f + yf * 6.0f));
	return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3f NormalizedWaveNoise2Dg(vec2f xy) {
	float i0 = floor(xy.x), i1 = i0 + 1.0f;
	float j0 = floor(xy.y), j1 = j0 + 1.0f;
	vec2f n00 = uniform_unit_circle(hash22(vec2f(i0, j0)));
	vec2f n01 = uniform_unit_circle(hash22(vec2f(i0, j1)));
	vec2f n10 = uniform_unit_circle(hash22(vec2f(i1, j0)));
	vec2f n11 = uniform_unit_circle(hash22(vec2f(i1, j1)));
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

float SimplexNoise2D(vec2f xy) {
	const float K1 = 0.3660254038f;  // (sqrt(3)-1)/2
	const float K2 = 0.2113248654f;  // (-sqrt(3)+3)/6
	vec2f p = xy + (xy.x + xy.y)*K1;
	vec2f i = floor(p);
	vec2f f1 = xy - (i - (i.x + i.y)*K2);
	vec2f s = f1.x < f1.y ? vec2f(0.0f, 1.0f) : vec2f(1.0f, 0.0f);
	vec2f f2 = f1 - s + K2;
	vec2f f3 = f1 - 1.0f + 2.0f*K2;
	vec2f n1 = 2.0f * hash22(i) - 1.0f;
	vec2f n2 = 2.0f * hash22(i + s) - 1.0f;
	vec2f n3 = 2.0f * hash22(i + 1.0) - 1.0f;
	vec3f v = vec3f(dot(f1, n1), dot(f2, n2), dot(f3, n3));
	vec3f w = max(-vec3f(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5f, vec3f(0.0f));
	return dot((w*w*w*w) * v, vec3f(32.0f));
}

vec3f SimplexNoise2Dg(vec2f xy) {
	const float K1 = 0.3660254038f;  // (sqrt(3)-1)/2
	const float K2 = 0.2113248654f;  // (-sqrt(3)+3)/6
	vec2f p = xy + (xy.x + xy.y)*K1;
	vec2f i = floor(p);
	vec2f f1 = xy - (i - (i.x + i.y)*K2);
	vec2f s = f1.x < f1.y ? vec2f(0.0f, 1.0f) : vec2f(1.0f, 0.0f);
	vec2f f2 = f1 - s + K2;
	vec2f f3 = f1 - 1.0f + 2.0f*K2;
	vec2f n1 = 2.0f * hash22(i) - 1.0f;
	vec2f n2 = 2.0f * hash22(i + s) - 1.0f;
	vec2f n3 = 2.0f * hash22(i + 1.0) - 1.0f;
	vec3f v = vec3f(dot(f1, n1), dot(f2, n2), dot(f3, n3));
	vec3f w = max(-vec3f(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5f, vec3f(0.0f));
	vec3f w3 = w * w * w, w4 = w3 * w;
	return 32.0f * vec3f(
		(w.x == 0.0f ? vec2f(0.0f) : -8.0f * w3.x * f1 * v.x + w4.x * n1) +
		(w.y == 0.0f ? vec2f(0.0f) : -8.0f * w3.y * f2 * v.y + w4.y * n2) +
		(w.z == 0.0f ? vec2f(0.0f) : -8.0f * w3.z * f3 * v.z + w4.z * n3),
		w4.x * v.x + w4.y * v.y + w4.z * v.z);
}

float NormalizedSimplexNoise2D(vec2f xy) {
	const float K1 = 0.3660254038f;  // (sqrt(3)-1)/2
	const float K2 = 0.2113248654f;  // (-sqrt(3)+3)/6
	vec2f p = xy + (xy.x + xy.y)*K1;
	vec2f i = floor(p);
	vec2f f1 = xy - (i - (i.x + i.y)*K2);
	vec2f s = f1.x < f1.y ? vec2f(0.0f, 1.0f) : vec2f(1.0f, 0.0f);
	vec2f f2 = f1 - s + K2;
	vec2f f3 = f1 - 1.0f + 2.0f*K2;
	vec2f n1 = uniform_unit_circle(hash22(i));
	vec2f n2 = uniform_unit_circle(hash22(i + s));
	vec2f n3 = uniform_unit_circle(hash22(i + 1.0));
	vec3f v = vec3f(dot(f1, n1), dot(f2, n2), dot(f3, n3));
	vec3f w = max(-vec3f(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5f, vec3f(0.0f));
	return dot((w*w*w*w) * v, vec3f(32.0f));
}

vec3f NormalizedSimplexNoise2Dg(vec2f xy) {
	const float K1 = 0.3660254038f;  // (sqrt(3)-1)/2
	const float K2 = 0.2113248654f;  // (-sqrt(3)+3)/6
	vec2f p = xy + (xy.x + xy.y)*K1;
	vec2f i = floor(p);
	vec2f f1 = xy - (i - (i.x + i.y)*K2);
	vec2f s = f1.x < f1.y ? vec2f(0.0f, 1.0f) : vec2f(1.0f, 0.0f);
	vec2f f2 = f1 - s + K2;
	vec2f f3 = f1 - 1.0f + 2.0f*K2;
	vec2f n1 = uniform_unit_circle(hash22(i));
	vec2f n2 = uniform_unit_circle(hash22(i + s));
	vec2f n3 = uniform_unit_circle(hash22(i + 1.0));
	vec3f v = vec3f(dot(f1, n1), dot(f2, n2), dot(f3, n3));
	vec3f w = max(-vec3f(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5f, vec3f(0.0f));
	vec3f w3 = w * w * w, w4 = w3 * w;
	return 32.0f * vec3f(
		(w.x == 0.0f ? vec2f(0.0f) : -8.0f * w3.x * f1 * v.x + w4.x * n1) +
		(w.y == 0.0f ? vec2f(0.0f) : -8.0f * w3.y * f2 * v.y + w4.y * n2) +
		(w.z == 0.0f ? vec2f(0.0f) : -8.0f * w3.z * f3 * v.z + w4.z * n3),
		w4.x * v.x + w4.y * v.y + w4.z * v.z);
}

// Note that the gradient of this thing is discontinuous
float SimplexValueNoise2D(vec2f xy) {
	// simplex grid
	const float K1 = 0.3660254038f;  // (sqrt(3)-1)/2
	const float K2 = 0.2113248654f;  // (-sqrt(3)+3)/6
	vec2f p = xy + (xy.x + xy.y)*K1;
	vec2f p1 = floor(p);
	vec2f s = xy.x - p1.x < xy.y - p1.y ? vec2f(0.0f, 1.0f) : vec2f(1.0f, 0.0f);
	vec2f p2 = p1 + s;
	vec2f p3 = p1 + 1.0;
	float v1 = 2.0f * hash12(p1) - 1.0f;
	float v2 = 2.0f * hash12(p2) - 1.0f;
	float v3 = 2.0f * hash12(p3) - 1.0f;
	// interpolation
	vec2f f = p - p1, c = -s + 1.0;
	float m = 1.0f / det(s, c);
	float u = m * det(f, c);
	float uv = m * det(s, f);
	return v1 + u * (v2 - v1) + uv * (v3 - v2);  // mix(v1, mix(v2, v3, v), u)
}

vec3f SimplexValueNoise2Dg(vec2f xy) {
	// simplex grid
	const float K1 = 0.3660254038f;  // (sqrt(3)-1)/2
	const float K2 = 0.2113248654f;  // (-sqrt(3)+3)/6
	vec2f p = xy + (xy.x + xy.y)*K1;
	vec2f p1 = floor(p);
	vec2f s = xy.x - p1.x < xy.y - p1.y ? vec2f(0.0f, 1.0f) : vec2f(1.0f, 0.0f);
	vec2f p2 = p1 + s;
	vec2f p3 = p1 + 1.0;
	float v1 = 2.0f * hash12(p1) - 1.0f;
	float v2 = 2.0f * hash12(p2) - 1.0f;
	float v3 = 2.0f * hash12(p3) - 1.0f;
	// interpolation
	vec2f f = p - p1, c = -s + 1.0;
	float m = 1.0f / det(s, c);
	float u = m * det(f, c);
	float uv = m * det(s, f);
	vec2f grad_u = m * vec2f(c.y, -c.x);
	vec2f grad_uv = m * vec2f(-s.y, s.x);
	float val = v1 + u * (v2 - v1) + uv * (v3 - v2);
	vec2f grad = grad_u * (v2 - v1) + grad_uv * (v3 - v2);
	return vec3f(grad + (grad.x + grad.y)*K1, val);
}
