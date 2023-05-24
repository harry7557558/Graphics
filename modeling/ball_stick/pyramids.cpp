#include "ball_stick.h"

std::vector<ball> balls;
std::vector<stick> sticks;

int main(int argc, char* argv[]) {

	// dimension of the pyramids
	const int N = 4;

	// triangle-based pyramid
	mat3f T_trig = mat3f(
		1.0f, 0.0f, 0.0f,
		0.5f, 0.866025404f, 0.0f,
		0.5f, 0.288675135f, 0.816496581f
	);
	for (int k = 0; k <= N; k++) {
		for (int j = 0; j + k <= N; j++) {
			for (int i = 0; i + j + k <= N; i++) {
				vec3f p = T_trig * vec3f(i, j, k);
				balls.push_back(ball{ p, 0xf0f8ff });
			}
		}
	}
	connect_sticks(balls, sticks, 1.0f, 0.01f, 0xffb050);

	// square-based pyramid
	mat3f T_square = mat3f(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.707106781f
	);
	int TP_N = balls.size();
	vec3f translate = vec3f(float(N + max(0.2*N, 1.0)), 0, 0);
	for (int k = 0; k <= N; k++) {
		for (int j = 0; j + k <= N; j++) {
			for (int i = 0; i + k <= N; i++) {
				vec3f p = T_square * vec3f(i, j, k);
				balls.push_back(ball{ p + translate, 0xfff8f0 });
			}
		}
	}
	connect_sticks(std::vector<ball>(balls.begin() + TP_N, balls.end()), sticks, 1.0f, 0.01f, 0x80b0ff);

	// write file
	printf("%d balls\n", (int)balls.size());
	printf("%d sticks\n", (int)sticks.size());
	write_file(argv[1], balls, sticks, 0.18f, 0.09f);
	return 0;
}
