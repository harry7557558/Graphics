precision highp float;

#define PI 3.1415926


// from https://www.shadertoy.com/view/4djSRW

float hash12(vec2 p) {
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

vec2 uniform_unit_circle(vec2 uv) {
	float a = 2.0*PI*uv.x;
	return sqrt(uv.y)*vec2(cos(a), sin(a));
}

float det(vec2 a, vec2 b) {
    return a.x*b.y-a.y*b.x;
}


// 2D noise functions, without and with gradient

float CosineNoise2D(vec2 xy) {
    return cos(PI*xy.x) * cos(PI*xy.y);
}

vec3 CosineNoise2Dg(vec2 xy) {
    float sinx = sin(PI*xy.x), siny = sin(PI*xy.y);
    float cosx = cos(PI*xy.x), cosy = cos(PI*xy.y);
    return vec3(-PI * sinx*cosy, -PI * cosx*siny, cosx*cosy);
}

float ValueNoise2D(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    float xf = xy.x-i0; xf = xf*xf*xf*(10.0+xf*(-15.0+xf*6.0));
    float yf = xy.y-j0; yf = yf*yf*yf*(10.0+yf*(-15.0+yf*6.0));
    return 2.0 * mix(
        mix(hash12(vec2(i0,j0)), hash12(vec2(i0,j1)), yf),
        mix(hash12(vec2(i1,j0)), hash12(vec2(i1,j1)), yf),
        xf) - 1.0;
}

vec3 ValueNoise2Dg(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    float h00 = hash12(vec2(i0, j0));
    float h01 = hash12(vec2(i0, j1));
    float h10 = hash12(vec2(i1, j0));
    float h11 = hash12(vec2(i1, j1));
    vec2 xyf = xy - vec2(i0, j0);
    vec2 intp = ((xyf * 6.0 - 15.0) * xyf + 10.0) * xyf * xyf * xyf;
    vec2 intpd = ((xyf * 30.0 - 60.0) * xyf + 30.0) * xyf * xyf;
    return vec3(
        2.0 * mix(h10 - h00, h11 - h01, intp.y) * intpd.x,
        2.0 * mix(h01 - h00, h11 - h10, intp.x) * intpd.y,
        2.0 * mix(mix(h00, h01, intp.y), mix(h10, h11, intp.y), intp.x) - 1.0
    );
}

float GradientNoise2D(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    float v00 = dot(2.0 * hash22(vec2(i0, j0)) - 1.0, xy - vec2(i0, j0));
    float v01 = dot(2.0 * hash22(vec2(i0, j1)) - 1.0, xy - vec2(i0, j1));
    float v10 = dot(2.0 * hash22(vec2(i1, j0)) - 1.0, xy - vec2(i1, j0));
    float v11 = dot(2.0 * hash22(vec2(i1, j1)) - 1.0, xy - vec2(i1, j1));
    float xf = xy.x - i0; xf = xf * xf * xf * (10.0 + xf * (-15.0 + xf * 6.0));
    float yf = xy.y - j0; yf = yf * yf * yf * (10.0 + yf * (-15.0 + yf * 6.0));
    //return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
    return v00 + (v10 - v00)*xf + (v01 - v00)*yf + (v00 + v11 - v01 - v10) * xf*yf;
}

vec3 GradientNoise2Dg(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    vec2 g00 = 2.0*hash22(vec2(i0, j0)) - 1.0;
    vec2 g01 = 2.0*hash22(vec2(i0, j1)) - 1.0;
    vec2 g10 = 2.0*hash22(vec2(i1, j0)) - 1.0;
    vec2 g11 = 2.0*hash22(vec2(i1, j1)) - 1.0;
    float v00 = dot(g00, xy - vec2(i0, j0));
    float v01 = dot(g01, xy - vec2(i0, j1));
    float v10 = dot(g10, xy - vec2(i1, j0));
    float v11 = dot(g11, xy - vec2(i1, j1));
    vec2 xyf = xy - vec2(i0, j0);
    vec2 intp = ((xyf * 6.0 - 15.0) * xyf + 10.0) * xyf * xyf * xyf;
    vec2 intpd = ((xyf * 30.0 - 60.0) * xyf + 30.0) * xyf * xyf;
    return vec3(
        g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10)*intp.x*intp.y
        + (vec2(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx) * intpd,
        v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
    );
}

float NormalizedGradientNoise2D(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    float v00 = dot(uniform_unit_circle(hash22(vec2(i0, j0))), xy - vec2(i0, j0));
    float v01 = dot(uniform_unit_circle(hash22(vec2(i0, j1))), xy - vec2(i0, j1));
    float v10 = dot(uniform_unit_circle(hash22(vec2(i1, j0))), xy - vec2(i1, j0));
    float v11 = dot(uniform_unit_circle(hash22(vec2(i1, j1))), xy - vec2(i1, j1));
    float xf = xy.x - i0; xf = xf * xf * xf * (10.0 + xf * (-15.0 + xf * 6.0));
    float yf = xy.y - j0; yf = yf * yf * yf * (10.0 + yf * (-15.0 + yf * 6.0));
    return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3 NormalizedGradientNoise2Dg(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    vec2 g00 = uniform_unit_circle(hash22(vec2(i0, j0)));
    vec2 g01 = uniform_unit_circle(hash22(vec2(i0, j1)));
    vec2 g10 = uniform_unit_circle(hash22(vec2(i1, j0)));
    vec2 g11 = uniform_unit_circle(hash22(vec2(i1, j1)));
    float v00 = dot(g00, xy - vec2(i0, j0));
    float v01 = dot(g01, xy - vec2(i0, j1));
    float v10 = dot(g10, xy - vec2(i1, j0));
    float v11 = dot(g11, xy - vec2(i1, j1));
    vec2 xyf = xy - vec2(i0, j0);
    vec2 intp = ((xyf * 6.0 - 15.0) * xyf + 10.0) * xyf * xyf * xyf;
    vec2 intpd = ((xyf * 30.0 - 60.0) * xyf + 30.0) * xyf * xyf;
    return vec3(
        g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10)*intp.x*intp.y
        + (vec2(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx) * intpd,
        v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
    );
}

float WaveNoise2D(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    const float freq = 2.0;
    float v00 = sin(freq*dot(2.0*hash22(vec2(i0, j0)) - 1.0, xy));
    float v01 = sin(freq*dot(2.0*hash22(vec2(i0, j1)) - 1.0, xy));
    float v10 = sin(freq*dot(2.0*hash22(vec2(i1, j0)) - 1.0, xy));
    float v11 = sin(freq*dot(2.0*hash22(vec2(i1, j1)) - 1.0, xy));
    float xf = xy.x - i0; xf = xf * xf * xf * (10.0 + xf * (-15.0 + xf * 6.0));
    float yf = xy.y - j0; yf = yf * yf * yf * (10.0 + yf * (-15.0 + yf * 6.0));
    return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3 WaveNoise2Dg(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    vec2 n00 = 2.0*hash22(vec2(i0, j0)) - 1.0;
    vec2 n01 = 2.0*hash22(vec2(i0, j1)) - 1.0;
    vec2 n10 = 2.0*hash22(vec2(i1, j0)) - 1.0;
    vec2 n11 = 2.0*hash22(vec2(i1, j1)) - 1.0;
    const float freq = 2.0;
    float v00 = sin(freq*dot(n00, xy));
    float v01 = sin(freq*dot(n01, xy));
    float v10 = sin(freq*dot(n10, xy));
    float v11 = sin(freq*dot(n11, xy));
    vec2 g00 = freq * cos(freq*dot(n00, xy)) * n00;
    vec2 g01 = freq * cos(freq*dot(n01, xy)) * n01;
    vec2 g10 = freq * cos(freq*dot(n10, xy)) * n10;
    vec2 g11 = freq * cos(freq*dot(n11, xy)) * n11;
    vec2 xyf = xy - vec2(i0, j0);
    vec2 intp = ((xyf * 6.0 - 15.0) * xyf + 10.0) * xyf * xyf * xyf;
    vec2 intpd = ((xyf * 30.0 - 60.0) * xyf + 30.0) * xyf * xyf;
    return vec3(
        g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10) * intp.x*intp.y
        + (vec2(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx) * intpd,
        v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
    );
}

float NormalizedWaveNoise2D(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    const float freq = 2.0;
    float v00 = sin(freq*dot(uniform_unit_circle(hash22(vec2(i0, j0))), xy));
    float v01 = sin(freq*dot(uniform_unit_circle(hash22(vec2(i0, j1))), xy));
    float v10 = sin(freq*dot(uniform_unit_circle(hash22(vec2(i1, j0))), xy));
    float v11 = sin(freq*dot(uniform_unit_circle(hash22(vec2(i1, j1))), xy));
    float xf = xy.x - i0; xf = xf * xf * xf * (10.0 + xf * (-15.0 + xf * 6.0));
    float yf = xy.y - j0; yf = yf * yf * yf * (10.0 + yf * (-15.0 + yf * 6.0));
    return mix(mix(v00, v01, yf), mix(v10, v11, yf), xf);
}

vec3 NormalizedWaveNoise2Dg(vec2 xy) {
    float i0 = floor(xy.x), i1 = i0 + 1.0;
    float j0 = floor(xy.y), j1 = j0 + 1.0;
    vec2 n00 = uniform_unit_circle(hash22(vec2(i0, j0)));
    vec2 n01 = uniform_unit_circle(hash22(vec2(i0, j1)));
    vec2 n10 = uniform_unit_circle(hash22(vec2(i1, j0)));
    vec2 n11 = uniform_unit_circle(hash22(vec2(i1, j1)));
    const float freq = 2.0;
    float v00 = sin(freq*dot(n00, xy));
    float v01 = sin(freq*dot(n01, xy));
    float v10 = sin(freq*dot(n10, xy));
    float v11 = sin(freq*dot(n11, xy));
    vec2 g00 = freq * cos(freq*dot(n00, xy)) * n00;
    vec2 g01 = freq * cos(freq*dot(n01, xy)) * n01;
    vec2 g10 = freq * cos(freq*dot(n10, xy)) * n10;
    vec2 g11 = freq * cos(freq*dot(n11, xy)) * n11;
    vec2 xyf = xy - vec2(i0, j0);
    vec2 intp = ((xyf * 6.0 - 15.0) * xyf + 10.0) * xyf * xyf * xyf;
    vec2 intpd = ((xyf * 30.0 - 60.0) * xyf + 30.0) * xyf * xyf;
    return vec3(
        g00 + (g10 - g00)*intp.x + (g01 - g00)*intp.y + (g00 + g11 - g01 - g10) * intp.x*intp.y
        + (vec2(v10 - v00, v01 - v00) + (v00 + v11 - v01 - v10)*intp.yx) * intpd,
        v00 + (v10 - v00)*intp.x + (v01 - v00)*intp.y + (v00 + v11 - v01 - v10) * intp.x*intp.y
    );
}

float SimplexNoise2D(vec2 xy) {
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 i = floor(p);
    vec2 f1 = xy - (i - (i.x + i.y)*K2);
    vec2 s = f1.x < f1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 f2 = f1 - s + K2;
    vec2 f3 = f1 - 1.0 + 2.0*K2;
    vec2 n1 = 2.0 * hash22(i) - 1.0;
    vec2 n2 = 2.0 * hash22(i + s) - 1.0;
    vec2 n3 = 2.0 * hash22(i + 1.0) - 1.0;
    vec3 v = vec3(dot(f1, n1), dot(f2, n2), dot(f3, n3));
    vec3 w = max(-vec3(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5, vec3(0.0));
    return dot((w*w*w*w) * v, vec3(32.0));
}

vec3 SimplexNoise2Dg(vec2 xy) {
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 i = floor(p);
    vec2 f1 = xy - (i - (i.x + i.y)*K2);
    vec2 s = f1.x < f1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 f2 = f1 - s + K2;
    vec2 f3 = f1 - 1.0 + 2.0*K2;
    vec2 n1 = 2.0 * hash22(i) - 1.0;
    vec2 n2 = 2.0 * hash22(i + s) - 1.0;
    vec2 n3 = 2.0 * hash22(i + 1.0) - 1.0;
    vec3 v = vec3(dot(f1, n1), dot(f2, n2), dot(f3, n3));
    vec3 w = max(-vec3(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5, vec3(0.0));
    vec3 w3 = w * w * w, w4 = w3 * w;
    return 32.0 * vec3(
        (w.x == 0.0 ? vec2(0.0) : -8.0 * w3.x * f1 * v.x + w4.x * n1) +
        (w.y == 0.0 ? vec2(0.0) : -8.0 * w3.y * f2 * v.y + w4.y * n2) +
        (w.z == 0.0 ? vec2(0.0) : -8.0 * w3.z * f3 * v.z + w4.z * n3),
        w4.x * v.x + w4.y * v.y + w4.z * v.z);
}

float NormalizedSimplexNoise2D(vec2 xy) {
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 i = floor(p);
    vec2 f1 = xy - (i - (i.x + i.y)*K2);
    vec2 s = f1.x < f1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 f2 = f1 - s + K2;
    vec2 f3 = f1 - 1.0 + 2.0*K2;
    vec2 n1 = uniform_unit_circle(hash22(i));
    vec2 n2 = uniform_unit_circle(hash22(i + s));
    vec2 n3 = uniform_unit_circle(hash22(i + 1.0));
    vec3 v = vec3(dot(f1, n1), dot(f2, n2), dot(f3, n3));
    vec3 w = max(-vec3(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5, vec3(0.0));
    return dot((w*w*w*w) * v, vec3(32.0));
}

vec3 NormalizedSimplexNoise2Dg(vec2 xy) {
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 i = floor(p);
    vec2 f1 = xy - (i - (i.x + i.y)*K2);
    vec2 s = f1.x < f1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 f2 = f1 - s + K2;
    vec2 f3 = f1 - 1.0 + 2.0*K2;
    vec2 n1 = uniform_unit_circle(hash22(i));
    vec2 n2 = uniform_unit_circle(hash22(i + s));
    vec2 n3 = uniform_unit_circle(hash22(i + 1.0));
    vec3 v = vec3(dot(f1, n1), dot(f2, n2), dot(f3, n3));
    vec3 w = max(-vec3(dot(f1, f1), dot(f2, f2), dot(f3, f3)) + 0.5, vec3(0.0));
    vec3 w3 = w * w * w, w4 = w3 * w;
    return 32.0 * vec3(
        (w.x == 0.0 ? vec2(0.0) : -8.0 * w3.x * f1 * v.x + w4.x * n1) +
        (w.y == 0.0 ? vec2(0.0) : -8.0 * w3.y * f2 * v.y + w4.y * n2) +
        (w.z == 0.0 ? vec2(0.0) : -8.0 * w3.z * f3 * v.z + w4.z * n3),
        w4.x * v.x + w4.y * v.y + w4.z * v.z);
}

// Note that the gradient of this thing is discontinuous
float SimplexValueNoise2D(vec2 xy) {
    // simplex grid
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 p1 = floor(p);
    vec2 s = xy.x - p1.x < xy.y - p1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 p2 = p1 + s;
    vec2 p3 = p1 + 1.0;
    float v1 = 2.0 * hash12(p1) - 1.0;
    float v2 = 2.0 * hash12(p2) - 1.0;
    float v3 = 2.0 * hash12(p3) - 1.0;
    // interpolation
    vec2 f = p - p1, c = -s + 1.0;
    float m = 1.0 / det(s, c);
    float u = m * det(f, c);
    float uv = m * det(s, f);
    return v1 + u * (v2 - v1) + uv * (v3 - v2);  // mix(v1, mix(v2, v3, v), u)
}

vec3 SimplexValueNoise2Dg(vec2 xy) {
    // simplex grid
    const float K1 = 0.3660254038;  // (sqrt(3)-1)/2
    const float K2 = 0.2113248654;  // (-sqrt(3)+3)/6
    vec2 p = xy + (xy.x + xy.y)*K1;
    vec2 p1 = floor(p);
    vec2 s = xy.x - p1.x < xy.y - p1.y ? vec2(0.0, 1.0) : vec2(1.0, 0.0);
    vec2 p2 = p1 + s;
    vec2 p3 = p1 + 1.0;
    float v1 = 2.0 * hash12(p1) - 1.0;
    float v2 = 2.0 * hash12(p2) - 1.0;
    float v3 = 2.0 * hash12(p3) - 1.0;
    // interpolation
    vec2 f = p - p1, c = -s + 1.0;
    float m = 1.0 / det(s, c);
    float u = m * det(f, c);
    float uv = m * det(s, f);
    vec2 grad_u = m * vec2(c.y, -c.x);
    vec2 grad_uv = m * vec2(-s.y, s.x);
    float val = v1 + u * (v2 - v1) + uv * (v3 - v2);
    vec2 grad = grad_u * (v2 - v1) + grad_uv * (v3 - v2);
    return vec3(grad + (grad.x + grad.y)*K1, val);
}


// Variance of noise functions
const vec3 CosineNoise2Dv = vec3(vec2(PI*PI/4.), 1./4.);
const vec3 ValueNoise2Dv = vec3(vec2(3620./4851.), 36721./160083.);
const vec3 GradientNoise2Dv = vec3(vec2(77320./297297.), 193670./6243237.);
const vec3 NormalizedGradientNoise2Dv = vec3(vec2(19330./99099.), 96835./4162158.);
const vec3 WaveNoise2Dv = vec3(1.495, 1.511, 0.303);
const vec3 NormalizedWaveNoise2Dv = vec3(1.404, 1.397, 0.305);
const vec3 SimplexNoise2Dv = vec3(0.461, 0.466, 0.020);
const vec3 NormalizedSimplexNoise2Dv = vec3(0.348, 0.346, 0.015);
const vec3 SimplexValueNoise2Dv = vec3(0.979, 1.006, 0.170);
