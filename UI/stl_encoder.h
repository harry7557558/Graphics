// a very sketchy binary STL writer
// for plotting 2d math functions


#pragma once

#ifndef _INC_STDIO
#include <stdio.h>
#endif
#ifndef _STDINT
#include <stdint.h>
#endif
#ifndef _VECTOR_
#include <vector>
#endif

#ifndef __INC_GEOMETRY_H
#include "numerical/geometry.h"
#endif

#include "ui/colors/ColorFunctions.h"

#pragma pack(push, 1)

// 12 bytes
struct stl_vec3 {
	float x, y, z;
	stl_vec3() {}
	stl_vec3(float x, float y, float z) :x(x), y(y), z(z) {}
	stl_vec3(vec3f p) :x(p.x), y(p.y), z(p.z) {}
	stl_vec3(vec3 p) :x((float)p.x), y((float)p.y), z((float)p.z) {}
};

// 50 bytes
struct stl_triangle {
	stl_vec3 n, a, b, c;
	int16_t col = 0;
	stl_triangle() {}
	template<typename _trig>
	stl_triangle(_trig T) {
		a = stl_vec3(T[0]), b = stl_vec3(T[1]), c = stl_vec3(T[2]), n = stl_vec3(0., 0., 0.);
		this->setColor(vec3f(NAN));
	}
	template<typename _trig, typename _cvec3>
	stl_triangle(_trig T, _cvec3 col = _cvec3(NAN)) {
		a = stl_vec3(T[0]), b = stl_vec3(T[1]), c = stl_vec3(T[2]), n = stl_vec3(0., 0., 0.);
		this->setColor(col);
	}
	template<typename _vec3, typename _cvec3>
	stl_triangle(_vec3 a, _vec3 b, _vec3 c, _cvec3 col = _cvec3(NAN)) {
		this->a = stl_vec3(a), this->b = stl_vec3(b), this->c = stl_vec3(c), n = vec3(0.);
		this->setColor(col);
	}
	template<typename _cvec3>
	void setColor(_cvec3 p) {
		if (isnan(p.sqr())) { col = 0; return; }
		uint16_t r = (uint16_t)(31.99f * clamp(p.x, 0.f, 1.f));
		uint16_t g = (uint16_t)(31.99f * clamp(p.y, 0.f, 1.f));
		uint16_t b = (uint16_t)(31.99f * clamp(p.z, 0.f, 1.f));
		col = (uint16_t)0b1000000000000000 | (r << 10) | (g << 5) | b;
	}
};

#pragma pack(pop)



constexpr uint8_t STL_CCW = 0x01;
constexpr uint8_t STL_CW = 0x02;

bool writeSTL(FILE* fp, stl_triangle data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = 0) {
	if (!fp) return false;

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

	// normal calculation
	if (correct_normal == STL_CCW || correct_normal == STL_CW) {
		for (unsigned i = 0; i < N; i++) {
			const stl_vec3* p = &data[i].a;
			stl_vec3 a = p[0], b = p[1], c = p[2];
			stl_vec3 u(b.x - a.x, b.y - a.y, b.z - a.z);
			stl_vec3 v(c.x - a.x, c.y - a.y, c.z - a.z);
			if (correct_normal == STL_CW) std::swap(u, v);
			stl_vec3 n(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x);
			double m = 1.0 / sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
			data[i].n = stl_vec3((float)(m*n.x), (float)(m*n.y), (float)(m*n.z));
		}
	}

	// write triangles
#if 0
	for (unsigned i = 0; i < N; i++) {
		if (fwrite(&data[i], 1, sizeof(stl_triangle), fp) != 50)
			return false;
	}
#else
	if (fwrite(&data[0], N, sizeof(stl_triangle), fp) != 50 * N)
		return false;
#endif

	fflush(fp);
	return true;
}

bool writeSTL(FILE* fp, const triangle_3d data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = STL_CCW) {
	stl_triangle* T = new stl_triangle[N];
	if (!T) return false;
	for (unsigned i = 0; i < N; i++)
		T[i] = stl_triangle(data[i]);
	bool res = writeSTL(fp, T, N, header, correct_normal);
	delete T; return res;
}

bool writeSTL(const char filename[], stl_triangle data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = 0) {
	FILE* fp = fopen(filename, "wb");
	if (!fp) return false;
	bool ok = writeSTL(fp, data, N, header, correct_normal);
	fclose(fp);
	return ok;
}
bool writeSTL(const char filename[], const triangle_3d data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = STL_CCW) {
	stl_triangle* T = new stl_triangle[N];
	if (!T) return false;
	for (unsigned i = 0; i < N; i++)
		T[i] = stl_triangle(data[i]);
	bool res = writeSTL(filename, T, N, header, correct_normal);
	delete T; return res;
}



// write colored STL
// vec3 ColorF(vec3)
template<typename trig, typename ColorFunction>
bool writeSTL_recolor(FILE* fp, const trig data[], unsigned N,
	const char header[80], const uint8_t correct_normal, ColorFunction ColorF) {
	stl_triangle* T = new stl_triangle[N];
	if (!T) return false;
	for (unsigned i = 0; i < N; i++) {
		T[i] = stl_triangle(data[i], ColorF((1. / 3.)*(data[i].A + data[i].B + data[i].C)));
	}
	bool res = writeSTL(fp, T, N, header, correct_normal);
	delete T; return res;
}

// convert triangle arrays
void convertTriangles(std::vector<stl_triangle> &res, const triangle_3d src[], unsigned N) {
	res.reserve(res.size() + N);
	for (unsigned i = 0; i < N; i++) {
		res.push_back(stl_triangle(src[i]));
	}
}

template<typename ColorFunction>  // vec3 ColorFunction(vec3 position)
void convertTriangles_color(std::vector<stl_triangle> &res, const triangle_3d src[], unsigned N, ColorFunction ColorF) {
	res.reserve(res.size() + N);
	for (unsigned i = 0; i < N; i++) {
		res.push_back(stl_triangle(src[i], ColorF(src[i].center())));
	}
}
template<typename ColorFunction>  // vec3 ColorFunction(vec3 unit_normal)
void convertTriangles_color_normal(std::vector<stl_triangle> &res, const triangle_3d src[], unsigned N,
	ColorFunction ColorF = [](vec3 n) { return 0.5*n + vec3(.5); }) {
	res.reserve(res.size() + N);
	for (unsigned i = 0; i < N; i++) {
		res.push_back(stl_triangle(src[i], ColorF(src[i].unit_normal())));
	}
}
template<typename ColorFunction>
bool writeSTL_recolor_normal(FILE* fp, const triangle_3d data[], unsigned N,
	const char header[80], ColorFunction ColorF = [](vec3 n) { return 0.5*n + vec3(.5); }) {
	stl_triangle* T = new stl_triangle[N];
	if (!T) return false;
	for (unsigned i = 0; i < N; i++) {
		T[i] = stl_triangle(data[i], ColorF(data[i].unit_normal()));
	}
	bool res = writeSTL(fp, T, N, header, STL_CCW);
	delete T; return res;
}




// discretize a 2d function to an array of STL triangles
// F(x, y)
// allocate trigs to 2*xD*yD

template<typename Fun>
int stl_fun2trigs(Fun F, stl_triangle* trigs, double x0, double x1, double y0, double y1, int xD, int yD, double z_min = -INFINITY, double z_max = INFINITY) {
	int CNT = 0;
	double dx = (x1 - x0) / xD, dy = (y1 - y0) / yD;

	// samples
	vec3 *S = new vec3[(xD + 1)*(yD + 1)];
	double minz = z_max, maxz = z_min;
	for (int j = 0; j <= yD; j++) {
		double y = y0 + j * dy;
		for (int i = 0; i <= xD; i++) {
			double x = x0 + i * dx;
			double z = F(x, y);
			z = clamp(z, z_min, z_max);
			S[j*(xD + 1) + i] = vec3(x, y, z);
			if (z > maxz) maxz = z;
			if (z < minz) minz = z;
		}
	}

	// draw triangles
	for (int j = 0; j < yD; j++) {
		vec3 *Px0 = &S[j*(xD + 1)], *Px1 = Px0 + xD + 1;
		for (int i = 0; i < xD; i++) {
			// read points
			vec3 p00 = Px0[i], p10 = Px0[i + 1];
			vec3 p01 = Px1[i], p11 = Px1[i + 1];
			// construct triangles
			stl_triangle T1, T2;
			if ((p01 - p10).sqr() < (p11 - p00).sqr()) {
				T1.a = p00, T1.b = p10, T1.c = p01, T1.n = ncross(p10 - p00, p01 - p00);
				T2.a = p11, T2.b = p01, T2.c = p10, T2.n = ncross(p01 - p11, p10 - p11);
			}
			else {
				T1.a = p00, T1.b = p10, T1.c = p11, T1.n = ncross(p10 - p00, p11 - p00);
				T2.a = p00, T2.b = p11, T2.c = p01, T2.n = ncross(p11 - p00, p01 - p00);
			}
			trigs[CNT++] = T1, trigs[CNT++] = T2;
		}
	}


	// set color
	double invdz = 1.0 / (maxz - minz);
	if (!(invdz > 0. && invdz < 1e100)) invdz = 0.;
	for (int i = 0; i < CNT; i++) {
		double z = (1. / 3.)*(trigs[i].a.z + trigs[i].b.z + trigs[i].c.z);
		if (isnan(z)) {
			if (isnan(trigs[i].a.z)) trigs[i].a.z = clamp(0.f, (float)minz, (float)maxz);
			if (isnan(trigs[i].b.z)) trigs[i].b.z = clamp(0.f, (float)minz, (float)maxz);
			if (isnan(trigs[i].c.z)) trigs[i].c.z = clamp(0.f, (float)minz, (float)maxz);
			trigs[i].setColor(vec3(0, 0.2, 0));  // dark green means NAN
		}
		else {
			vec3 col = ColorFunctions::Rainbow((z - minz) * invdz);
			trigs[i].setColor(col);
		}
	}

	delete S;
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
