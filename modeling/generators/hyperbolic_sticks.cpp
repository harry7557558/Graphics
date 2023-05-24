// a model to test rasterization rendering

#include "ui/stl_encoder.h"

std::vector<triangle_3d> T;

void addStick(vec3 p, vec3 q, double r, int N) {
	vec3 d = normalize(q - p);
	vec3 u = ncross(d, vec3(1.23, 4.56, 7.89));
	vec3 v = ncross(d, u);
	for (int i = 0; i < N; i++) {
		double a0 = i * (2.*PI) / N, a1 = a0 + 2.*PI / N;
		vec3 w0 = u * cos(a0) + v * sin(a0), w1 = u * cos(a1) + v * sin(a1);
		vec3 p0 = p + r * w0, p1 = p + r * w1, q0 = q + r * w0, q1 = q + r * w1;
		T.push_back(triangle_3d{ q0, p0, q1 });
		T.push_back(triangle_3d{ p0, p1, q1 });
		T.push_back(triangle_3d{ p0, p, p1 });
		T.push_back(triangle_3d{ q1, q, q0 });
	}
}

int main(int argc, char* argv[]) {
	const int N = 50;
	const vec3 P0(2, -2, -3), P1(2, 2, 3);
	for (int i = 0; i < N; i++) {
		mat3 M = axis_angle(vec3(0, 0, 1), i*(2.*PI) / N);
		addStick(M*P0, M*P1, 0.1, 10);
	}

	writeSTL(argv[1], &T[0], T.size());
	return 0;
}
