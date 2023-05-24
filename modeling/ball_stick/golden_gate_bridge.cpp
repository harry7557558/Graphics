#include "ball_stick.h"

template<typename Fun>
float bisection_search(Fun f, float a, float b) {
	float fa = f(a), fb = f(b);
	for (int i = 0; i < 64; i++) {
		float c = 0.5f*(a + b);
		float fc = f(c);
		if (fc*fa > 0.0) a = c, fa = fc;
		else b = c, fb = fc;
		if (b - a < 1e-6) break;
	}
	return 0.5f*(a + b);
}


std::vector<ball> Balls;
std::vector<stick> Sticks;

int main(int argc, char* argv[]) {

	const float rt2_2 = 0.707106781f;
	mat3f T = mat3f(
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.5f, 0.5f, 0.707106781f
	);

	ivec3 P0 = ivec3(-10, -100, -20), P1 = ivec3(10, 100, 70);


	// towers
	std::vector<ball> balls;
	for (int k = P0.z; k <= P1.z; k++) {
		for (int j = P0.y; j <= P1.y; j++) {
			for (int i = P0.x; i <= P1.x; i++) {
				vec3f p0 = T * vec3f(i - k / 2, j - k / 2, k);

				vec3f p = p0;
				p.x = abs(p.x), p.y = abs(abs(p.y) - 50.0f);
				float tower = max(max(abs(p.x - 5.0f) - 1.5f, p.y - 2.0f), p.z - 66.0f*rt2_2);
				float cross = std::min({
					max(max(p.x - 5.0f, p.y - 1.5f), abs(p.z - 18.0f*rt2_2) - 2.0f*rt2_2),
					max(max(p.x - 5.0f, p.y - 1.5f), abs(p.z - 32.0f*rt2_2) - 2.0f*rt2_2),
					max(max(p.x - 5.0f, p.y - 1.5f), abs(p.z - 46.0f*rt2_2) - 2.0f*rt2_2),
					max(max(p.x - 5.0f, p.y - 1.5f), abs(p.z - 60.0f*rt2_2) - 2.0f*rt2_2)
					});
				float d = min(tower, cross);

				if (d < 0.0) balls.push_back(ball{ p0 });
			}
		}
	}
	connect_sticks(balls, Sticks, 1.0f, 0.01f, 0xf1481e);
	Balls.insert(Balls.end(), balls.begin(), balls.end());


	// road
	balls.clear();
	for (int k = P0.z; k <= P1.z; k++) {
		for (int j = P0.y; j <= P1.y; j++) {
			for (int i = P0.x; i <= P1.x; i++) {
				vec3f p0 = T * vec3f(i - k / 2, j - k / 2, k);

				vec3f p = p0;
				p.x = abs(p.x);
				float road = max(p.x - 4.0f, abs(p.z - 0.5f*rt2_2) - 1.0f*rt2_2);
				float d = road;

				if (d < 0.0) balls.push_back(ball{ p0 });
			}
		}
	}
	connect_sticks(balls, Sticks, 1.0f, 0.01f, 0xf1481e);
	Balls.insert(Balls.end(), balls.begin(), balls.end());


	// chain (willing to make all sticks the same length)
	vec2f C0 = vec2f(-48.5f, 63.0f*rt2_2);
	vec2f C1 = vec2f(48.5f, 63.0f*rt2_2);  // vec2f(-C0.x, C0.y)
	const int CN = 121;
	// catenary equation y=a*cosh(x/a)+b
	float l = (float)CN - 1.0f;
	float a = bisection_search([&](float x) {
		float y = 2.0f*x*sinh(C1.x / x);
		return y - l;
	}, 1e-6f, 1e+6f);
	float b = C0.y - a * cosh(C0.x / a);
	// initial positions
	vec2f cps[CN], cps_f[CN];
	for (int i = 0; i < CN; i++) {
		float x = bisection_search([&](float t) {
			float s0 = a * sinh(C0.x / a);
			float s1 = a * sinh(t / a);
			return s1 - s0 - (float(i) / float(CN - 1) * l);
		}, -C1.x, C1.x);
		float y = a * cosh(x / a) + b;
		cps[i] = vec2f(x, y);
	}

	balls.clear();
	for (int i = 1; i < CN - 1; i++) {
		vec2f p = cps[i];
		balls.push_back(ball{ vec3f(-4.5f, p.x, p.y) });
		balls.push_back(ball{ vec3f(4.5f, p.x, p.y) });
		p = cps[i] + vec2f(100.0f, 0.0f);
		if (abs(p.x) <= 100.0f)
			balls.push_back(ball{ vec3f(-4.5f, p.x, p.y) }),
			balls.push_back(ball{ vec3f(4.5f, p.x, p.y) });
		p = cps[i] - vec2f(100.0f, 0.0f);
		if (abs(p.x) <= 100.0f)
			balls.push_back(ball{ vec3f(-4.5f, p.x, p.y) }),
			balls.push_back(ball{ vec3f(4.5f, p.x, p.y) });
	}

	Balls.insert(Balls.end(), balls.begin(), balls.end());
	connect_sticks(balls, Sticks, 1.0f, 0.01f, 0xf1481e);

	// translation
	vec3f translate = vec3f(0.0f, 0.0f, 20.0f*rt2_2);
	translate = vec3f(0, 0, 0);
	for (int i = 0; i < (int)Balls.size(); i++)
		Balls[i].p += translate;
	for (int i = 0; i < (int)Sticks.size(); i++)
		Sticks[i].p1 += translate, Sticks[i].p2 += translate;

	// write file
	printf("%d balls\n", (int)Balls.size());
	printf("%d sticks\n", (int)Sticks.size());
	write_file(argv[1], Balls, Sticks, 0.18f, 0.09f);
	return 0;
}
