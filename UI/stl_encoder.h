// a very sketchy binary STL writer
// for plotting 2d math functions


#pragma once

#include <stdio.h>
#include <stdint.h>

#include "numerical/geometry.h"


#pragma pack(push, 1)

// 12 bytes
struct stl_vec3 {
	float x, y, z;
	stl_vec3() {}
	stl_vec3(float x, float y, float z) :x(x), y(y), z(z) {}
#ifdef __INC_GEOMETRY_H
	stl_vec3(vec3 p) :x((float)p.x), y((float)p.y), z((float)p.z) {}
#endif
};

// 50 bytes
struct stl_triangle {
	stl_vec3 n, a, b, c;
	int16_t col = 0;
};

#pragma pack(pop)



// correct_normal: a lowercase string with 3 characters indicating the direction of the normals
// right-hand rule; ex. "bac" => n=cross(b-a,c-a)  "abc" => n=cross(a-b,c-b)

bool writeSTL(FILE* fp, stl_triangle data[], unsigned N,
	const char header[80] = nullptr, const char* correct_normal = "\0\0\0") {

	// 80-byte header
	if (header == 0) {
		for (int i = 0; i < 80; i++)
			if (fputc(0, fp) == EOF) return false;
	}
	else {
		if (fwrite(header, 1, 80, fp) != 80) return false;
	}

	// triangle count
	if (fwrite(&N, sizeof(N), 1, fp) != 1)
		return false;

	// normal correction
	int cnt[256];  // interpret string
	for (int i = 0; i < 256; i++) cnt[i] = 0;
	for (int i = 0; i < 3; i++)
		cnt[correct_normal[i]]++;
	if (cnt[0] == 3);  // no normal correction
	else {
		if (!(cnt['a'] == 1 && cnt['b'] == 1 && cnt['c'] == 1))
			return false;  // invalid string
		else {
			int ai = cnt[1] - 'a', bi = cnt[0] - 'a', ci = cnt[2] - 'a';
			for (unsigned i = 0; i < N; i++) {
				stl_vec3* p = &data[i].a;
				stl_vec3 a = p[ai], b = p[bi], c = p[ci];
				stl_vec3 u(b.x - a.x, b.y - a.y, b.z - a.z);
				stl_vec3 v(c.x - a.x, c.y - a.y, c.z - a.z);
				stl_vec3 n(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
				double m = 1.0 / sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
				data[i].n = stl_vec3((float)(m*n.x), (float)(m*n.y), (float)(m*n.z));
			}
		}
	}

	// write triangles
	for (unsigned i = 0; i < N; i++) {
		if (fwrite(&data[i], 1, sizeof(stl_triangle), fp) != 50)
			return false;
	}

	fflush(fp);
	return true;
}




// discretize a 2d function to an array of STL triangles
// F(x, y)

template<typename Fun>
int stl_fun2trigs(Fun F, stl_triangle* trigs, double x0, double x1, double y0, double y1, int xD, int yD, double z_min = -INFINITY, double z_max = INFINITY) {
	int CNT = 0;
	double dx = (x1 - x0) / xD, dy = (y1 - y0) / yD;

	// a sketchy function that returns a point with given x and y coordinates
	auto funp = [&](double x, double y)->vec3 {
		double z = clamp(F(x, y), z_min, z_max);
		if (isnan(z)) z = 0.;
		return vec3(x, y, z);
	};

	// store points calculated in the previous row
	vec3 *ps = new vec3[xD + 1];
	for (int i = 0; i <= xD; i++) ps[i] = funp(x0 + i * dx, y0);
	vec3 p0, p1, p2, p3;

	// draw triangles
	for (int j = 0; j < yD; j++) {
		double y = y0 + j * dy;
		vec3 px1_ = funp(x0, y + dy);
		for (int i = 0; i < xD; i++) {
			// calculate/read points
			double x = x0 + i * dx;
			p0 = ps[i], p1 = ps[i + 1];
			p2 = px1_, p3 = funp(x + dx, y + dy);
			// construct triangles
			stl_triangle T1, T2;
			if (length(p2 - p1) < length(p3 - p0)) {
				T1.a = p0, T1.b = p1, T1.c = p2, T1.n = ncross(p1 - p0, p2 - p0);
				T2.a = p3, T2.b = p2, T2.c = p1, T2.n = ncross(p2 - p3, p1 - p3);
			}
			else {
				T1.a = p0, T1.b = p1, T1.c = p3, T1.n = ncross(p1 - p0, p3 - p0);
				T2.a = p0, T2.b = p3, T2.c = p2, T2.n = ncross(p3 - p0, p2 - p0);
			}
			trigs[CNT++] = T1, trigs[CNT++] = T2;
			// update calculation results
			px1_ = p3, ps[i] = p2;
		}
		ps[xD] = p3;
	}

	delete ps;
	return CNT;  // should be 2*xD*yD
}





// testing

#if 0

#include "numerical/integration.h"
#include "numerical/random.h"
#include "numerical/optimization.h"
#include <cmath>

double fun(double x, double y) {
	//return cos(x*x + y * y) / (x*x + y * y + 1);
	//return sin(x*x + y * y) / (x*x + y * y);
	//return log10((x - 1)*(x - 1) + 100 * (y - x * x)*(y - x * x) + 1);
	//return NIntegrate_Gaussian2<double>([&](double t) { return sqrt(x*x*sin(t)*sin(t) + y * y*cos(t)*cos(t)); }, 0, 0.5 * PI, 64);
	//auto F = [](vec2 p) { return pow(p.x, 4.) - 4 * p.x*p.x*p.y + pow(p.y, 4.) + 0.1*pow(p.x, 3.); }; return -F(Newton_Iteration_2d(F, vec2(x, y)));
	//vec2 z(0.); for (int i = 0; i < 8; i++) z = vec2(z.x*z.x - z.y*z.y + x, 2.*z.x*z.y + y); return 1.0 / sqrt(z.sqr() + .1);
	//return sin(x)*cos(y);
	//return sin(x * x + abs(x) * y + y * y);
	//return 10.*(.2*x - x * x*x - y * y*y*y*y)*exp(-1.8*(x*x + y * y));
	//return exp(x)*sin(5 * y);
	//return min(x, y);
	//return sqrt(x*x + y * y);
	//return sqrt(2 - x * x - y * y);
	//return pow(1.0 - 4 * pow(x*x + y * y - 1, 2.) + (5 * x*x*x*x*y - 10 * x*x*y*y*y + y * y*y*y*y), 0.2);
	//return sqrt(1 - pow(sqrt(x*x + y * y) - 1.001, 2.));
	//return sqrt(sin(x*x + 1.618*y * y));
	//return pow((x + 1)*(x + 1) + y * y, -1.5) - pow((x - 1)*(x - 1) + y * y, -1.5);
	//return pow(x, y);
	//return log(y) / log(x);
	//return tgamma(x)*tgamma(y) / tgamma(x + y);
	//return x / y;
	//return (x*x + y * y) / (x*x - y * y);
	//return 0.5*log(x*x + y * y);
	//return atan2(0.5*(x + y), 0.5*(x - y));
	//return floor(x) + floor(y);
	//return round(x) / round(y);
	//return fmod(sin(12.9898*x + 78.233*y)*43758.5453, 1.);
	//return hashu((unsigned)(123456 * x + 987654 * y)) / 4294967296.;
	//return rand() / (double)RAND_MAX;
	return 0;
}

int main() {
	const int N = 80;
	const int TN = 2 * N*N;
	stl_triangle D[TN];
	stl_fun2trigs(fun, D, -3, 3, -3, 3, N, N, -10, 10);

	FILE* fp = fopen("D:\\test.stl", "wb");
	writeSTL(fp, D, TN);
	fclose(fp);
}

#endif
