#include "numerical/geometry.h"
#include "UI/stl_encoder.h"
#include <vector>
#include <functional>

struct triangle {
	vec3 a, b, c;
	vec3 col;
};
std::vector<triangle> trigs;

typedef std::function<vec3(double u, double v)> ParamSurface;

void triangulate(ParamSurface F, int un, int vn) {
	// u: 0-2pi; v: 0-1
	for (int ui = 0; ui < un; ui++) {
		double u0 = ui * 2.*PI / un;
		double u1 = (ui + 1) * 2.*PI / un;
		for (int vi = 0; vi < vn; vi++) {
			double v0 = vi * 1. / vn;
			double v1 = (vi + 1) * 1. / vn;
			vec3 p00 = F(u0, v0);
			vec3 p10 = F(u1, v0);
			vec3 p01 = F(u0, v1);
			vec3 p11 = F(u1, v1);
			if ((p10 - p01).sqr() < (p11 - p00).sqr()) {
				trigs.push_back(triangle{ p01, p00, p10, vec3(1.) });
				trigs.push_back(triangle{ p10, p11, p01, vec3(1.) });
			}
			else {
				trigs.push_back(triangle{ p11, p01, p00, vec3(1.) });
				trigs.push_back(triangle{ p00, p10, p11, vec3(1.) });
			}
		}
	}
}

double smoothstep(double x) {
	//return x<0. ? 0. : x>1. ? 1. : x;
	return x<0. ? 0. : x>1. ? 1. : x * x*(3. - 2.*x);
	//return x<0. ? 0. : x>1. ? 1. : x * x*x*(10. + x * (6.*x - 15.));
}
double inverse_smoothstep(double x) {
	return 0.5 - sin(asin(1. - 2.*x) / 3.);
}

vec3 smooth_fuse_G1(ParamSurface s1, ParamSurface s2, double r1, double r2, double u, double v) {
	v *= 2.;
	double a = 1. - r1, b = 1. + r2;
	if (v < a) return s1(u, v);
	if (v > b) return s2(u, v - 1);
	double t = (v - a) / (b - a);
	vec3 p1 = s1(u, a + r1 * t);
	vec3 p2 = s2(u, r2 * t);
	return mix(p1, p2, t);
	return p2;
}



// in cm
const double h = 10.0;
const double r0 = 3.0;
const double r1 = 4.0;
const double r0_r1 = r0 - 0.2;
const double r0_h1 = 0.4;
const double r0_r2 = r0;
const double r0_h2 = 0.5;
const double r1_r1 = r1 + 0.3;
const double r1_h1 = h - 0.1;
const double r1_r2 = r1 + 0.1;
const double r1_h2 = h - 0.2;

vec3 map_test(double u, double v) {
	vec2 c = vec2(1. - 0.2*sin(4.*PI*v) - 0.5*v*v, 2.*v + 0.1*cos(4.*PI*v));
	return vec3(c.x * cossin(u), c.y);
}

vec3 map_lower_0(double u, double v) {
	vec2 c = mix(vec2(r0, 0), vec2(r0_r1, r0_h1), v);
	return vec3(c.x*cossin(u), c.y);
}
vec3 map_lower_1(double u, double v) {
	vec2 c = mix(vec2(r0_r1, r0_h1), vec2(r0_r2, r0_h2), v);
	return vec3(c.x*cossin(u), c.y);
}
vec3 map_lower(double u, double v) {
	return smooth_fuse_G1(map_lower_0, map_lower_1, 0.2, 0.5, u, v);
}

vec3 map_ring_0(double u, double v) {
	vec2 c = mix(vec2(r1, h), vec2(r1_r1, r1_h1), v);
	return vec3(c.x*cossin(u), c.y);
}
vec3 map_ring_1(double u, double v) {
	vec2 c = mix(vec2(r1_r1, h), vec2(r1_r2, r1_h2), v);
	return vec3(c.x*cossin(u), c.y);
}
vec3 map_ring(double u, double v) {
	return smooth_fuse_G1(map_ring_0, map_ring_1, 0.8, 0.8, u, v);
}
vec3 map_slant(double u, double v) {
	vec2 c = mix(vec2(r0_r2, r0_h2), vec2(r1, h), v);
	return vec3(c.x*cossin(u), c.y);
}
vec3 map_body(double u, double v) {
	return smooth_fuse_G1(map_slant, map_ring, 0.01, 0.3, u, v);
}

vec3 map_bottom(double u, double v) {
	return vec3(r0*v*cossin(u), 0.4*r0_h1*(1. - tanh(40.*(v - 0.8))));
}

vec3 map_cup(double u, double v) {
	return smooth_fuse_G1(
		map_bottom,
		[](double u, double v) { return smooth_fuse_G1(map_lower, map_body, 0.1, 0.005, u, v); },
		0.05, 0.2, u, v);
}



#define STB_IMAGE_IMPLEMENTATION
#include ".libraries\stb_image.h"

int main(int argc, char* argv[]) {
	const int UN = 200;
	const int VN = 200;

	const double thickness = 0.01;
	const double eps = 1e-5;

	// outer layer
	triangulate([&](double u, double v)->vec3 {
		if (v == 0.) return map_cup(u, v) - vec3(0, 0, thickness);
		vec3 p = map_cup(u, v);
		vec3 dpdu = (map_cup(u + eps, v) - map_cup(u - eps, v)) / (2.*eps);
		vec3 dpdv = (map_cup(u, v + eps) - map_cup(u, v - eps)) / (2.*eps);
		vec3 n = ncross(dpdu, dpdv);
		return p + thickness * n;
	}, UN, VN * 2);

	// color
	struct rgba { uint8_t r, g, b; };
	int W, H;
	rgba *img = (rgba*)stbi_load("D:\\conch.jpg", &W, &H, nullptr, 3);
	printf("%d %d\n", W, H);
	vec3 *colors = new vec3[W*H];
	for (int i = 0; i < W; i++) for (int j = 0; j < H; j++)
		colors[(H - 1 - j)*W + i] = vec3(img[j*W + i].r, img[j*W + i].g, img[j*W + i].b) * (1. / 255.);
	vec2 cupdim = vec2(2.*PI*r0, hypot(r1 - r0, h));
	vec2 sc = 1. * vec2(W, H) * cupdim / min(cupdim.x, cupdim.y);
	for (int t = 0; t < (int)trigs.size(); t++) {
		vec3 p = (trigs[t].a + trigs[t].b + trigs[t].c) / 3.;
		vec2 uv = vec2(atan2(p.y, p.x) / (2.*PI) + 0.5, p.z / h);
		uv = (uv - vec2(0.5, 0.5))*sc + vec2(0.5*W, 0.5*H);
		int i = (int)(uv.x), j = (int)(uv.y);
		if (i >= 0 && i < W && j >= 0 && j < H) {
			trigs[t].col = colors[j*W + i];
		}
		//trigs[t].col *= vec3(0.9, 0.85, 0.65);
	}

	// inner layer
	triangulate([&](double u, double v)->vec3 {
		v = 1. - v;
		if (v == 0.) return map_cup(u, v) + vec3(0, 0, thickness);
		vec3 p = map_cup(u, v);
		vec3 dpdu = (map_cup(u + eps, v) - map_cup(u - eps, v)) / (2.*eps);
		vec3 dpdv = (map_cup(u, v + eps) - map_cup(u, v - eps)) / (2.*eps);
		vec3 n = ncross(dpdu, dpdv);
		return p - thickness * n;
	}, UN, VN);

	// mouth
	triangulate([&](double u, double t)->vec3 {
		vec3 p = map_cup(u, 1.);
		vec3 dpdu = (map_cup(u + eps, 1.) - map_cup(u - eps, 1.)) / (2.*eps);
		vec3 dpdv = (map_cup(u, 1. + eps) - map_cup(u, 1. - eps)) / (2.*eps);
		vec3 n = ncross(dpdu, dpdv);
		vec3 p1 = p + thickness * n;
		vec3 p2 = p - thickness * n;
		vec3 d1 = 3. * thickness * normalize(dpdv);
		vec3 d2 = -d1;
		double t2 = t * t, t3 = t2 * t;
		return (2.*t3 - 3.*t2 + 1.)*p1 + (t3 - 2.*t2 + t)*d1 + (-2.*t3 + 3.*t2)*p2 + (t3 - t2)*d2;
	}, UN, VN / 20);

	// output
	std::vector<stl_triangle> stl;
	for (int i = 0; i < (int)trigs.size(); i++) {
		stl.push_back(stl_triangle(trigs[i].a, trigs[i].b, trigs[i].c, trigs[i].col));
	}
	writeSTL(argv[1], &stl[0], stl.size(), nullptr, STL_CCW);
	return 0;
}

