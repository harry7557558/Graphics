#include "ball_stick.h"

std::vector<ball> Balls;
std::vector<stick> Sticks;


// R: controls the dimension of the tower
// B: controls the dimension of the hanging octahedron
void add_shape(int R, int B, int ROPE_LENGTH, bool HAS_UNDER_LAYERS, vec3f position, uint32_t color = 0xffffff) {

	std::vector<ball> balls;

	mat3f T = mat3f(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.707106781f
	);

	// bottom layer
	for (int i = -R; i <= R; i++) {
		for (int j = -R; j <= R; j++) {
			if (abs(i) < R - 2 && abs(j) < R - 2) continue;
			vec3f p = vec3f(i, j, 0);
			balls.push_back(ball{ p });
		}
	}

	// second layer
	for (int i = -R; i < R; i++) {
		for (int j = -R; j < R; j++) {
			if (i > -R + 1 && i < R - 2 && j > -R + 1 && j < R - 2) continue;
			vec3f p = T * vec3f(i, j, 1);
			balls.push_back(ball{ p });
		}
	}

	// more layers
	for (int k = R - 2; k > -R + 2; k--) {
		auto add_square = [&](vec3f p) {
			for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
				balls.push_back(ball{ p + vec3f(i, j, 0) });
		};
		add_square(T * vec3f(-R, -R, R - k));
		add_square(T * vec3f(-R, k - 1, R - k));
		add_square(T * vec3f(k - 1, -R, R - k));
		add_square(T * vec3f(k - 1, k - 1, R - k));
	}

	// top layers
	for (int k = 0; k <= 2; k++) {
		for (int i = 0; i + k <= 2; i++) {
			for (int j = 0; j + k <= 2; j++) {
				vec3f p = T * vec3f(-R + i, -R + j, 2 * R - 2 + k);
				balls.push_back(ball{ p });
			}
		}
	}

	// hang
	vec3f p0 = T * vec3f(-R + 2, -R + 2, 2 * R - 4);
	balls.push_back(ball{ p0 });
	for (int i = 1; i < ROPE_LENGTH; i++)
		balls.push_back(ball{ p0 = p0 - vec3f(0, 0, 1) });
	p0 = p0 - vec3f(0, 0, 1);
	std::vector<vec3f> octa;
	for (int k = -B; k <= B; k++) {
		for (int i = 0; i + abs(k) <= B; i++) {
			for (int j = 0; j + abs(k) <= B; j++) {
				vec3f p = T * (k < 0 ? vec3f(i - k, j - k, k) : vec3f(i, j, k));
				octa.push_back(p);
			}
		}
	}
	for (vec3f p : octa) {
		p = p - T * vec3f(0, 0, B) + p0;
		p = mat3f(
			cos(1.0f), -sin(1.0f), 0.0f,
			sin(1.0f), cos(1.0f), 0.0f,
			0.0f, 0.0f, 1.0f
		) * p;
		balls.push_back(ball{ p });
	}

	if (HAS_UNDER_LAYERS) {

		// negative two'th layer
		for (int i = -R; i < R; i++) {
			for (int j = -R; j < R; j++) {
				if (i > -R + 1 && i < R - 2 && j > -R + 1 && j < R - 2) continue;
				vec3f p = T * vec3f(i + 1, j + 1, -1);
				balls.push_back(ball{ p });
			}
		}

		// more under layers
		auto add_square = [&](vec3f p) {
			for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
				balls.push_back(ball{ p + vec3f(i, j, 0) });
		};
		for (int k = R - 2; k > -R + 2; k--) {
			add_square(T * vec3f(-k, -k, k - R));
			add_square(T * vec3f(-k, R - 1, k - R));
			add_square(T * vec3f(R - 1, -k, k - R));
			add_square(T * vec3f(R - 1, R - 1, k - R));
		}

		// base
		for (int i = R - 3; i <= R; i++) for (int j = R - 3; j <= R; j++)
			add_square(T * vec3f(i, j, -2 * R + 2));
	}

	for (int i = 0; i < (int)balls.size(); i++)
		balls[i] = ball{ balls[i].p + position, 0xffffff };
	Balls.insert(Balls.end(), balls.begin(), balls.end());
	connect_sticks(balls, Sticks, 1.0f, 0.01f, color);
}

int main(int argc, char* argv[]) {


	add_shape(6, 4, 3, true, vec3f(0.0f, 0.0f, 10.0f*0.707106781f), 0x027eb8);

	add_shape(4, 2, 2, true, vec3f(20.0f, 0.0f, 6.0f*0.707106781f), 0xb5b802);

	add_shape(5, 2, 1, false, vec3f(0.0f, 20.0f, 0.0f), 0x10a927);

	add_shape(7, 3, 2, false, vec3f(20.0f, 20.0f, 0.0f), 0xd314c9);


	printf("%d balls\n", (int)Balls.size());
	printf("%d sticks\n", (int)Sticks.size());
	write_file(argv[1], Balls, Sticks, 0.18f, 0.09f);
	return 0;
}
