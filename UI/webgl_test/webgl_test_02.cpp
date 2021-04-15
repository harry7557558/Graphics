#include "numerical/geometry.h"
#include <stdio.h>

#define W 1920
#define H 1080

struct vec4 {
	double x, y, z, w;
	vec4() {}
	vec4(double x, double y, double z, double w) :x(x), y(y), z(z), w(w) {}
	vec4(vec3 p, double w) :x(p.x), y(p.y), z(p.z), w(w) {}
	vec2 xy() const { return vec2(x, y); }
};

vec3 clamp(vec3 p, double a, double b) {
	return vec3(clamp(p.x, a, b), clamp(p.y, a, b), clamp(p.z, a, b));
}

vec4 gl_FragCoord, gl_FragColor;



const vec2 iRotate = vec2(0.1, -2.0);
const double iDist = 8.0;
const vec2 iResolution = vec2(W, H);

const vec3 light = normalize(vec3(0, 0, 1));

bool intersect_sphere(vec3 ce, double r, vec3 ro, vec3 rd, double &t, vec3 &n) {
	vec3 p = ro - ce;
	double b = dot(p, rd), c = dot(p, p) - r * r;
	double delta = b * b - c; if (delta <= 0.0) return false;
	delta = sqrt(delta);
	t = -b - delta; if (t <= 0.0) t = -b + delta;
	if (t <= 0.0) return false;
	n = (p + rd * t) / r; return true;
}

bool intersect_rod(vec3 pa, vec3 pb, double r, vec3 ro, vec3 rd, double &t, vec3 &n) {
	vec3 ab = pb - pa, p = ro - pa;
	double ab2 = dot(ab, ab), abrd = dot(ab, rd), abp = dot(ab, p);
	double a = ab2 - abrd * abrd;
	double b = ab2 * dot(p, rd) - abp * abrd;
	double c = ab2 * dot(p, p) - abp * abp - r * r*ab2;
	double delta = b * b - a * c; if (delta <= 0.0) return false;
	delta = sqrt(delta);
	t = (-b - delta) / a;
	if (t > 0.0) {
		double h = abp + t * abrd;
		if (h > 0.0 && h < ab2) {
			n = (p + rd * t - ab * h / ab2) / r;
			return true;
		}
	}
#if 1
	t = (-b + delta) / a;
	if (t > 0.0) {
		double h = abp + t * abrd;
		if (h > 0.0 && h < ab2) {
			n = (p + rd * t - ab * h / ab2) / r;
			return true;
		}
	}
#endif
	return false;
}

vec3 traceRay(vec3 ro, vec3 rd) {

	const double R = 0.2, r = 0.1;
	const double rt3_2 = 0.8660254037844386;

	vec3 t_col = vec3(1.0), f_col;

	for (int i = 0; i < 64; i++) {
		ro += 1e-4*rd;
		double min_t = 1e+12, t;
		vec3 min_n = vec3(0.0), n;

		// intersect with the plane
		t = -(ro.z + (rt3_2 + R)) / rd.z;
		if (t > 0.0) {
			min_t = t;
			f_col = vec3(0.5, 0.8, 1.0);
			vec2 p = ro.xy() + rd.xy()*t;
			if (mod(floor(p.x) + floor(p.y), 2.0) == 0.0) f_col *= 0.99;
			min_n = vec3(0, 0, 1);
		}

		// intersect with balls
		if (intersect_sphere(vec3(0, 0, 0), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}
		if (intersect_sphere(vec3(1, 0, 0), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}
		if (intersect_sphere(vec3(0, 1, 0), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}
		if (intersect_sphere(vec3(0.5, -0.5, rt3_2), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}
		if (intersect_sphere(vec3(0.5, 0.5, rt3_2), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}
		if (intersect_sphere(vec3(-0.5, 0.5, rt3_2), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}
		if (intersect_sphere(vec3(0.5, 0.5, -rt3_2), R, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = vec3(1.0);
		}

		// intersect with rods
		const vec3 rod_col = vec3(1.0, 0.7, 0.75);
		if (intersect_rod(vec3(-1, 0, 0), vec3(1, 0, 0), r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = rod_col;
		}
		if (intersect_rod(vec3(0, -1, 0), vec3(0, 1, 0), r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = rod_col;
		}
		if (intersect_rod(vec3(-0.5, -0.5, -rt3_2), vec3(0.5, 0.5, rt3_2), r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = rod_col;
		}
		if (intersect_rod(vec3(-0.5, 0.5, -rt3_2), vec3(0.5, -0.5, rt3_2), r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = rod_col;
		}
		if (intersect_rod(vec3(0.5, -0.5, -rt3_2), vec3(-0.5, 0.5, rt3_2), r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = rod_col;
		}
		if (intersect_rod(vec3(-0.5, -0.5, rt3_2), vec3(0.5, 0.5, -rt3_2), r, ro, rd, t, n) && t < min_t) {
			min_t = t, min_n = n, f_col = rod_col;
		}

		if (min_n == vec3(0.0)) break;
		min_n = normalize(min_n), rd = normalize(rd);
		ro = ro + rd * min_t;
		rd = rd - 2.0*dot(rd, min_n)*min_n;
		t_col *= f_col;
	}

	t_col *= max(dot(rd, light), 0.0);
	return t_col;
}

#define MAX_AA 5
const int AA = 5;

void glsl_main() {

	double rx = iRotate.x, rz = iRotate.y;
	vec3 w = vec3(cos(rx)*vec2(cos(rz), sin(rz)), sin(rx));
	vec3 u = vec3(-sin(rz), cos(rz), 0);
	vec3 v = cross(w, u);

	vec3 ro = iDist * w;

	vec3 col = vec3(0.0);
	for (int i = 0; i < MAX_AA; i++) {
		for (int j = 0; j < MAX_AA; j++) {
			vec2 aVertexPosition = 2.0 * (gl_FragCoord.xy() + vec2(i, j) / double(AA)) / iResolution - vec2(1.0);
			vec3 rd = normalize(mat3(u, v, -w)*vec3(aVertexPosition*iResolution, 2.0*length(iResolution)));
			//col += clamp(traceRay(ro, normalize(rd)), 0.0, 1.0);
			col += traceRay(ro, normalize(rd));
			if (j == AA - 1) break;
		}
		if (i == AA - 1) break;
	}
	col /= double(AA*AA);

	gl_FragColor = vec4(col, 1.0);
}




#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

uint32_t image[H][W];

int main(int argc, char* argv[]) {
	for (int i = 0; i < W; i++) {
		for (int j = 0; j < H; j++) {
			gl_FragCoord = vec4(i, j, 0, 0);
			glsl_main();
			if (gl_FragColor.x < 0.0 || gl_FragColor.y > 2.0) printf("%d %d\n", i, j);
			for (int u = 0; u < 4; u++)
				((uint8_t*)&image[H - j - 1][i])[u] =
				(uint8_t)(255.*((double*)&gl_FragColor)[u]);
		}
	}

	stbi_write_png("D:\\webgl_test_02.png", W, H, 4, image, 4 * W);
	return 0;
}
