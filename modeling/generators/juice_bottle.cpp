// reference a plastic juice bottle
// unit: cm

#include "ui/stl_encoder.h"
#include "triangulate/parametric_surface_adaptive_dist.h"


// Bezier curve
template<typename vec>
vec bezier2(vec a0, vec a1, vec a2, double t) {
	vec b0 = mix(a0, a1, t), b1 = mix(a1, a2, t);
	return mix(b0, b1, t);
}
template<typename vec>
vec bezier3(vec a0, vec a1, vec a2, vec a3, double t) {
	vec b0 = mix(a0, a1, t), b1 = mix(a1, a2, t), b2 = mix(a2, a3, t);
	vec c0 = mix(b0, b1, t), c1 = mix(b1, b2, t);
	return mix(c0, c1, t);
}


// signed distance functions
double sdSegment(vec2 a, vec2 b, vec2 p) {
	vec2 pa = p - a, ba = b - a;
	double h = dot(pa, ba) / dot(ba, ba);
	return length(pa - ba * clamp(h, 0.0, 1.0));
}


// smooth blending functions
template<typename vec> vec sstep(vec a, vec b, double t) {
	t = clamp(t, 0.0, 1.0);
	return mix(a, b, t*t*t * (10.0 + t * (-15.0 + 6.0*t)));
}
double sabs(double x, double k) {
	return sqrt(x * x + k * k);
}
double smin(double a, double b, double k) {
	double h = max(k - abs(a - b), 0.0) / k;
	return min(a, b) - h * h*h*k*(1.0 / 6.0);
}
double smax(double a, double b, double k) {
	return -smin(-a, -b, k);
}
vec3 sfuse(std::function<vec3(double u, double v)> s1, std::function<vec3(double u, double v)> s2, double r1, double r2, double u, double v) {
	v *= 2.0;
	double a = 1.0 - r1, b = 1.0 + r2;
	if (v < a) return s1(u, v);
	if (v > b) return s2(u, v - 1.0);
	double t = (v - a) / (b - a);
	vec3 p1 = s1(u, a + r1 * t);
	vec3 p2 = s2(u, r2 * t);
	return sstep(p1, p2, t);
}


// modeling the bottle

vec3 map_cap(double u, double v) {
	v = 1.0 - v;
	auto fun1 = [](double u, double v)->vec3 {
		double h = 15.4;
		double r = 1.8*v;
		return vec3(r*cossin(u), h);
	};
	auto fun2 = [](double u, double v)->vec3 {
		v *= 1.8;
		double h = 15.4 - v, r;
		if (v < 1.0) r = bezier3(1.7, 1.75, 1.8, 1.8, v / 1.0);
		else if (v < 1.2) r = bezier3(1.8, 1.8, 1.7, 1.7, (v - 1.0) / 0.2);
		else if (v < 1.3) r = bezier3(1.7, 1.7, 1.8, 1.8, (v - 1.2) / 0.1);
		else if (v < 1.4) r = 1.8;
		else if (v < 1.5) r = bezier3(1.8, 1.8, 1.5, 1.5, (v - 1.4) / 0.1);
		else r = 1.5;
		double texture = smax(0.2*(0.1 - acos(0.995*sin(60.0*u)) / PI), 0.0, 0.02) * sstep(1.0, 0.0, 40.0*(abs(v - 0.58) - 0.42));
		texture = sstep(0.04, texture, 40.0*(abs(v - 1.0) - 0.05));
		r += texture;
		return vec3(r*cossin(u), h);
	};
	vec3 p = sfuse(fun1, fun2, 0.4, 0.1, u, v);
	return p;
}

vec3 map_top(double u, double v) {
	double t = 3.0*v;
	vec2 hr;
	if (t < 1.0) hr = bezier3(vec2(0.0, 2.8), vec2(1.8, 2.8), vec2(1.7, 2.3), vec2(2.3, 2.3), t - 0.0);
	else if (t < 2.0) hr = bezier3(vec2(2.3, 2.3), vec2(3.3, 2.3), vec2(3.0, 2.5), vec2(4.0, 2.5), t - 1.0);
	else hr = bezier3(vec2(4.0, 2.5), vec2(5.0, 2.5), vec2(5.2, 1.5), vec2(5.8, 1.5), t - 2.0);
	double r = hr.y, h = hr.x + 7.8;
	double a = u - 0.2*sstep(0.0, 4.5, (h - 9.0) / 3.5);
	double texture = smin(0.5 * acos(0.995*sin(10.0*a)) / PI
		- sstep(0.12, -0.05, 0.7*(sabs(h - 10.5, 0.5)) - 0.9), 0.0, 0.1);
	r = r + texture;
	vec3 p = vec3(r*cossin(u), h);
	return p;
}

vec3 map_body(double u, double v) {
	double h = 6.2*v + 1.6;
	double texture = INFINITY;
	for (double k = 0; k < 4; k++)
		texture = min(texture, sabs(h - (3.0 + 1.2*k), 0.1));
	texture = -smax(0.2 - texture, 0.0, 0.1);
	double r = 2.77 + texture;
	r = sstep(r, 2.8, (h - 7.7) / 0.1);
	r = sstep(r, 2.8, (1.7 - h) / 0.1);
	vec3 p = vec3(r*cossin(u), h);
	return p;
}

vec3 map_bottom(double u, double v) {
	auto fun1 = [](double u, double v)->vec3 {
		vec2 rh = bezier2(vec2(0.0, 0.3), vec2(2.0, 0.3), vec2(2.8, -0.65), v);
		double r = rh.x, h = rh.y;
		vec2 tp = r * cossin(abs(mod(2.5*(u + 0.1*PI) + 0.5*PI, PI) - 0.5*PI) / 2.5);
		vec2 tq = r * cossin(abs(mod(2.5*(u + 0.3*PI) + 0.5*PI, PI) - 0.5*PI) / 2.5);
		double texture = smin(max(abs(r - 0.38) - 0.12, 0.0),
			smin(sdSegment(vec2(0.5, 0.0), vec2(2.0, 0.0), tp),
				sdSegment(vec2(1.2, 0.0), vec2(2.0, 0.0), tq), 0.1), 0.1);
		texture = smax(0.2 - 1.4*sabs(texture, 0.05), 0.0, 0.1);
		return vec3(r*cossin(u), h + texture);
	};
	auto fun2 = [](double u, double v)->vec3 {
		double h = 0.6 + 1.0 * v;
		double r = 2.8;
		return vec3(r*cossin(u), h);
	};
	vec3 p = sfuse(fun1, fun2, 0.5, 0.4, u, v);
	vec2 tp = 5.0 * v * cossin(abs(mod(5.0*(u + 0.1*PI) + 0.5*PI, PI) - 0.5*PI) / 5.0);
	double texture = sdSegment(vec2(2.2, 0.0), vec2(2.35, 0.0), tp);
	texture = smax(0.15 - 1.4*sabs(texture, 0.05), 0.0, 0.08);
	texture -= smin(sabs(length(tp - vec2(2.4, 0.0)), 0.1) - 0.15, 0.0, 0.3);
	p += texture * normalize(vec3(0, 0, 0.5) - p);
	return p;
}

vec3 map(double u, double v) {
	u *= 2.0*PI;
	v = 4.0 * v;
	if (v <= 1.0) return map_bottom(u, v);
	if (v <= 2.0) return map_body(u, v - 1.0);
	if (v <= 3.0) return map_top(u, v - 2.0);
	if (v <= 4.0) return map_cap(u, v - 3.0);
}


// triangulate and export

int main(int argc, char* argv[]) {

	// not a perfect mesh generator
	std::vector<triangle_3d> Trigs
		= AdaptiveParametricSurfaceTriangulator_dist
		<std::function<vec3(double, double)>>([](double u, double v)->vec3 {
		return map(u, v);
	}).triangulate_adaptive(
		0.0, 1.0, 0.0, 1.0, 32, 60, 8, 0.001, true, false);

	// write file
	writeSTL(argv[1], &Trigs[0], (int)Trigs.size());
	return 0;
}
