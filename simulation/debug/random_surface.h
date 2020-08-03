// generate random closed surface (triangulated)

#pragma GCC optimize "Ofast"
#include <stdio.h>
#include <algorithm>
#include "D:\Coding\Github\Graphics\fitting\numerical\random.h"

#define triangle _geometry_triangle<vec3>

// square of distance to a triangle
double d2Trig(vec3 p, triangle t) {
	// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
	vec3 ba = t.b - t.a; vec3 pa = p - t.a;
	vec3 cb = t.c - t.b; vec3 pb = p - t.b;
	vec3 ac = t.a - t.c; vec3 pc = p - t.c;
	vec3 nor = cross(ba, ac);
	auto sign = [](double x) {return x > 0 ? 1 : -1; };
	return
		(sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc)) < 2.0) ?
		min(min(
		(ba*clamp(dot(ba, pa) / ba.sqr(), 0.0, 1.0) - pa).sqr(),
			(cb*clamp(dot(cb, pb) / cb.sqr(), 0.0, 1.0) - pb).sqr()),
			(ac*clamp(dot(ac, pc) / ac.sqr(), 0.0, 1.0) - pc).sqr())
		: dot(nor, pa)*dot(nor, pa) / nor.sqr();
}

// https://www.researchgate.net/figure/Generation-of-random-convex-polyhedron_fig4_263093577, S_sd = 2
void randomPolyhedron(triangle* T, int N) {
	// generate random planar quadrilateral
	double a[4];
	do {
		for (int i = 0; i < 4; i++) a[i] = randf(0, 2 * PI);
		std::sort(a, a + 4);
	} while (a[1] - a[0] < .68 || a[2] - a[1] < .68 || a[3] - a[2] < .68 || a[0] + 2 * PI - a[3] < .68);
	vec3 s[6]; for (int i = 0; i < 4; i++) s[i] = vec3(cos(a[i]), sin(a[i]), 0);

	// generate octahedral
	vec3 n = rand3_c();
	s[4] = randf(0.5, 1.5)*n;
	s[5] = randf(-1.5, -0.5)*n;

	for (int i = 0; i < 4; i++) {
		T[i] = triangle{ s[4], s[i], s[(i + 1) % 4] };
		T[i + 4] = triangle{ s[5], s[i], s[(i + 3) % 4] };
	}

	//
}

// N: # of triangles, should be even, at least 4
void randomSurface(triangle* T, int N) {
	_SRAND(3);
	vec3 p0[4];
	for (int i = 0; i < 4; i++) p0[i] = rand3_n(5.0);
	if (det(p0[1] - p0[0], p0[2] - p0[0], p0[3] - p0[0]) < 0.) std::swap(p0[1], p0[2]);
	T[0] = triangle{ p0[0], p0[1], p0[3] };
	T[1] = triangle{ p0[0], p0[3], p0[2] };
	T[2] = triangle{ p0[0], p0[2], p0[1] };
	T[3] = triangle{ p0[1], p0[2], p0[3] };
	/*{
		for (int i = 0; i < 4; i++) {
			triangle t = T[i];
			vec3 n = normalize(cross(t.b - t.a, t.c - t.a));
			vec3 c = (t.a + t.b + t.c) / 3;
			vec3 p = n + c;
			T[i + 4] = triangle{ p + 0.1*(t.a - c), p + 0.1*(t.b - c), p + 0.1*(t.c - c) };
		}
		return;
	}*/
	for (int k = 4; k < N;) {
		int d = randi(0, k);
		triangle t = T[d];
		vec3 n = abs(randf_n(2.0))*0.5*sqrt(length(cross(t.b - t.a, t.c - t.a))) * normalize(cross(t.b - t.a, t.c - t.a));
		double u, v; do { u = randf(0, 1), v = randf(0, 1); } while (u + v > 1.0);
		vec3 p = t.a + u * (t.b - t.a) + v * (t.c - t.a) + n;
		T[d] = triangle{ p, t.a, t.b };
		T[k++] = triangle{ p, t.b, t.c };
		T[k++] = triangle{ p, t.c, t.a };
	}
}

bool writeSTL(triangle* T, int N, const char* filename) {
	FILE* fp = fopen(filename, "wb");
	if (!fp) return false;

	// stl header
	char s[80]; for (int i = 0; i < 80; i++) s[i] = 0;
	sprintf(s, "%d triangles", N);
	fwrite(s, 1, 80, fp);
	fwrite(&N, 4, 1, fp);

	// triangles
	auto writev = [&](vec3 p) {
		float f = p.x; fwrite(&f, sizeof(float), 1, fp);
		f = p.y; fwrite(&f, sizeof(float), 1, fp);
		f = p.z; fwrite(&f, sizeof(float), 1, fp);
	};
	for (int i = 0; i < N; i++) {
		triangle t = T[i];
		writev(normalize(cross(t.c - t.a, t.b - t.a)));
		writev(t.a); writev(t.b); writev(t.c);
		fputc(0, fp); fputc(0, fp);
	}

	fclose(fp); return true;
}

int main() {
	_SRAND(0);
	const int N = 8;
	triangle T[N];
	randomPolyhedron(T, N);
	writeSTL(T, N, "D:\\test.stl");
	return 0;
}

#undef triangle
