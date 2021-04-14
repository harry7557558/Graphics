// define structs and write files

#include "numerical/geometry.h"
#include <stdio.h>
#include <vector>
#include <algorithm>


struct ball {
	vec3f p;
	uint32_t col = 0xffffff;
};

struct stick {
	vec3f p1, p2;
	uint32_t col = 0xffffff;
};


// use sticks to connect any pair of balls with distance r
void connect_sticks(std::vector<ball> balls, std::vector<stick> &sticks, const float r, const float eps, uint32_t col) {
	// possibly O(N²)
	std::sort(balls.begin(), balls.end(), [](ball a, ball b) {
		return a.p.z < b.p.z;
	});
	for (int i = 0; i < (int)balls.size(); i++) {
		for (int j = i + 1; j < (int)balls.size(); j++) {
			if (abs(length(balls[j].p - balls[i].p) - 1.0f) < eps)
				sticks.push_back(stick{ balls[i].p, balls[j].p, col });
			if (balls[j].p.z - balls[i].p.z > 1.0f + eps) break;
		}
	}
}



void write_file(const char* filename,
	std::vector<ball> balls, std::vector<stick> sticks,
	const float r_ball, const float r_stick) {

	int BN = (int)balls.size();
	int SN = (int)sticks.size();

	// icosahedron
	const int B_VN = 12, B_FN = 20;
	const vec3f B_VS[B_VN] = { vec3f(0.0f,0.8506508f,0.5257311f), vec3f(-0.8506508f,0.5257311f,0.0f), vec3f(-0.5257311f,0.0f,0.8506508f), vec3f(0.0f,0.8506508f,-0.5257311f), vec3f(-0.5257311f,0.0f,-0.8506508f), vec3f(0.0f,-0.8506508f,-0.5257311f), vec3f(0.8506508f,-0.5257311f,0.0f), vec3f(0.0f,-0.8506508f,0.5257311f), vec3f(0.5257311f,0.0f,0.8506508f), vec3f(-0.8506508f,-0.5257311f,0.0f), vec3f(0.8506508f,0.5257311f,0.0f), vec3f(0.5257311f,0.0f,-0.8506508f), };
	const ivec3 B_FS[B_FN] = { ivec3(0,1,2), ivec3(3,1,0), ivec3(3,4,1), ivec3(5,6,7), ivec3(8,0,2), ivec3(8,2,7), ivec3(8,7,6), ivec3(9,1,4), ivec3(9,5,7), ivec3(9,2,1), ivec3(9,7,2), ivec3(9,4,5), ivec3(10,3,0), ivec3(10,0,8), ivec3(10,8,6), ivec3(11,4,3), ivec3(11,3,10), ivec3(11,6,5), ivec3(11,5,4), ivec3(11,10,6), };

	// hexagonal prism
	const int S_N = 6;

	std::vector<ball> vertice_list;  // position + color
	std::vector<ivec3> face_list;

	// balls (icosahedron)
	for (ball b : balls) {
		vec3f p = b.p;
		int fi0 = (int)vertice_list.size();
		for (int i = 0; i < B_VN; i++)
			vertice_list.push_back(ball{ p + r_ball * B_VS[i], b.col });
		for (int i = 0; i < B_FN; i++)
			face_list.push_back(ivec3(fi0) + B_FS[i]);
	}

	// sticks (hexagonal prism)
	for (stick s : sticks) {
		vec3f a = s.p1, b = s.p2, ab = b - a;
		vec3f u(r_stick, 0.0f, 0.0f), v(0.0f, r_stick, 0.0f);
		vec3f ut = normalize(cross(vec3f(0, 0, 1), ab));
		if (!isnan(ut.sqr())) {
			u = r_stick * ut;
			v = r_stick * normalize(cross(ab, ut));
		}
		int fi0 = (int)vertice_list.size();
		for (int i = 0; i < S_N; i++) {
			float t = i * float(2.0*PI) / S_N;
			vertice_list.push_back(ball{ a + cos(t)*u - sin(t)*v, s.col });
		}
		for (int i = 0; i < S_N; i++) {
			float t = (i + 0.5f) * float(2.0*PI) / S_N;
			vertice_list.push_back(ball{ b + cos(t)*u - sin(t)*v, s.col });
		}
		for (int i = 0; i < S_N; i++) {
			face_list.push_back(ivec3(fi0) + ivec3(i, i + S_N, (i + 1) % S_N));
			face_list.push_back(ivec3(fi0) + ivec3((i + 1) % S_N + S_N, (i + 1) % S_N, i + S_N));
		}
	}

	// ply file
	FILE* fp = fopen(filename, "wb");
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", B_VN * BN + 2 * S_N * SN);
	fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
	fprintf(fp, "property uchar blue\nproperty uchar green\nproperty uchar red\n");
	fprintf(fp, "element face %d\n", B_FN * BN + 2 * S_N * SN);
	fprintf(fp, "property list uchar int vertex_indices\n");
	fprintf(fp, "end_header\n");
	for (ball p : vertice_list) {
		fwrite(&p.p, 4, 3, fp);
		fwrite(&p.col, 1, 3, fp);
	}
	for (ivec3 p : face_list) {
		fputc(3, fp);
		fwrite(&p, 4, 3, fp);
	}
	fclose(fp);
}
