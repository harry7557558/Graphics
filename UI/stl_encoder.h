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

#include "ui/color_functions/poly.h"

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
		this->__setColor(vec3f(NAN));
	}
	template<typename _trig, typename _cvec3>
	stl_triangle(_trig T, _cvec3 col) {
		a = stl_vec3(T[0]), b = stl_vec3(T[1]), c = stl_vec3(T[2]), n = stl_vec3(0., 0., 0.);
		this->__setColor(col);
	}
	template<typename _vec3>
	stl_triangle(_vec3 a, _vec3 b, _vec3 c) {
		this->a = stl_vec3(a), this->b = stl_vec3(b), this->c = stl_vec3(c), n = stl_vec3(0., 0., 0.);
		this->__setColor(vec3f(NAN));
	}
	template<typename _vec3, typename _cvec3>
	stl_triangle(_vec3 a, _vec3 b, _vec3 c, _cvec3 col) {
		this->a = stl_vec3(a), this->b = stl_vec3(b), this->c = stl_vec3(c), n = vec3(0.);
		this->__setColor(col);
	}
	template<typename _cvec3>
	void __setColor(_cvec3 p) {
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

template<typename triangle>
bool writeSTL(FILE* fp, const triangle data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = STL_CCW) {
	stl_triangle* T = new stl_triangle[N];
	if (!T) return false;
	for (unsigned i = 0; i < N; i++)
		T[i] = stl_triangle(data[i][0], data[i][1], data[i][2]);
	bool res = writeSTL(fp, T, N, header, correct_normal);
	delete T; return res;
}

bool writeSTL(const char filename[], stl_triangle data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = STL_CCW) {
	FILE* fp = fopen(filename, "wb");
	if (!fp) return false;
	bool ok = writeSTL(fp, data, N, header, correct_normal);
	fclose(fp);
	return ok;
}

template<typename triangle>
bool writeSTL(const char filename[], const triangle data[], unsigned N,
	const char header[80] = nullptr, const uint8_t correct_normal = STL_CCW) {
	stl_triangle* T = new stl_triangle[N];
	if (!T) return false;
	for (unsigned i = 0; i < N; i++)
		T[i] = stl_triangle(data[i][0], data[i][1], data[i][2]);
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





template<typename _Float, typename _Int>
bool WritePLY(const char* filename,
	const _Float *vertices, int VN, const _Int *trig_faces, int TN) {

	FILE* fp = fopen(filename, "wb");
	if (!fp) return false;

	// ply header
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", VN);
	fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
	fprintf(fp, "element face %d\n", TN);
	fprintf(fp, "property list uchar int vertex_indices\n");
	fprintf(fp, "end_header\n");

	// vertices
	for (int i = 0; i < 3 * VN; i++) {
		float v = (float)vertices[i];
		fwrite(&v, 4, 1, fp);
	}

	// faces
	for (int i = 0; i < 3 * TN; i++) {
		uint32_t v = (uint32_t)trig_faces[i];
		if (i % 3 == 0) fputc(3, fp);
		fwrite(&v, 4, 1, fp);
	}

	return fclose(fp) == 0;
}
