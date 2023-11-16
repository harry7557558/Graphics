// test this header
#include "delaunay_2d.h"


#include "numerical/geometry.h"
#include "numerical/random.h"


std::vector<vec2> testcase_01() {
	std::vector<vec2> res;
	uint32_t seed = 1;
	for (int i = 0; i < 1000; i++) {
		vec2 p = rand2_u(seed);
		res.push_back(p);
	}
	return res;
}

std::vector<vec2> testcase_02() {
	std::vector<vec2> res;
	for (int i = -20; i <= 20; i++) {
		for (int j = -20; j <= 20; j++) {
			double x = 0.06*i, y = 0.06*j;
			if (x*x + y * y - sin(5.*x)*cos(5.*y) < 1.0)
				res.push_back(vec2(x, y));
		}
	}
	return res;
}

std::vector<vec2> testcase_03() {
	std::vector<vec2> res;
	auto f = [](vec2 p) {
		return pow(p.sqr() - 1, 3.) - p.x*p.x * p.y*p.y*p.y;
	};
	auto gradf = [&](vec2 p) {
		const double eps = 0.0001;
		return vec2(f(p + vec2(eps, 0)) - f(p - vec2(eps, 0)), f(p + vec2(0, eps)) - f(p - vec2(0, eps))) / (2.0*eps);
	};
	uint32_t seed = 1;
	for (int i = -10; i <= 10; i++) {
		for (int j = -10; j <= 10; j++) {
			vec2 p = 0.15*vec2(i, j) + 0.01*rand2_u(seed);
			for (int u = 0; u < 4; u++) {
				double z = f(p); vec2 grad = gradf(p);
				vec2 dp = normalize(grad) * min(abs(z) / length(grad), 0.2);
				p -= dp + 0.0*rand01(seed)*rand2(seed);
				res.push_back(p);
			}
		}
	}
	return res;
}

std::vector<vec2> testcase_04() {
	std::vector<vec2> res;
	auto p = [](double t) {
		return vec2(sin(6.*t), cos(5.*t));
	};
	for (double t = 0; t < 2.*PI; t += 0.005) {
		res.push_back(p(t));
	}
	return res;
}

std::vector<vec2> testcase_05() {
	std::vector<vec2> res;
	auto p = [](double t) {
		return sin(0.01*t)*sin(0.01*t)*cossin(t);
	};
	for (double t = 0; t < 100.*PI; t += 0.1) {
		res.push_back(p(t));
	}
	return res;
}

std::vector<vec2> testcase_06() {
	std::vector<vec2> res;
	auto p = [](double t) {
		return 0.01*t*cossin(t);
	};
	for (double t = 0; t < 100.; t += 0.1) {
		res.push_back(p(t));
	}
	return res;
}

std::vector<vec2> testcase_07() {
	std::vector<vec2> res;
	auto fun = [](vec2 p) {
		double x = p.x, y = p.y, x2 = x * x, y2 = y * y, x4 = x2 * x2, y4 = y2 * y2;
		return x4 + y4 - 1.5*x2*y2 - 0.5;
	};
	vec2 p0 = vec2(-1.5, -1.5), dp = vec2(3, 3);
	double dx = 1.0; int N = 1;
	for (int i = 0; i < 10; i++) {
		for (int i = 1; i < N; i += 2) {
			for (int j = 1; j < N; j += 2) {
				double u = i * dx, v = j * dx;
				bool s[4] = {
					fun(p0 + vec2(u,v)*dp) < 0, fun(p0 + vec2(u + dx,v)*dp) < 0,
					fun(p0 + vec2(u,v + dx)*dp) < 0, fun(p0 + vec2(u + dx,v + dx)*dp) < 0
				};
				if (((int)s[0] + (int)s[1] + (int)s[2] + (int)s[3]) % 4)
					res.push_back(p0 + vec2(u, v)*dp);
			}
		}
		dx /= 2, N *= 2;
	}
	return res;
}

std::vector<vec2> testcase_08() {
	std::vector<vec2> res;
	int N = 10;
	double dx = 1.0 / N;
	uint32_t seed = 0;
	for (int i = -N; i <= N; i++) {
		for (int j = -N; j <= N; j++) {
			vec2 p = vec2(i, j)*dx;
			double xd = 0.95*dx * (rand01(seed) - 0.5);
			double yd = 0.95*dx * (rand01(seed) - 0.5);
			if (abs(i) != N) p.x += xd;
			if (abs(j) != N) p.y += yd;
			res.push_back(p);
		}
	}
	return res;
}




#include <chrono>

int main() {
	const double SC = 200.;
	freopen("D:\\.svg", "w", stdout);
	printf("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='%d' height='%d'>\n", 600, 600);
	printf("<g transform='translate(%d,%d) scale(%lg,%lg)'>", 300, 300, SC, -SC);
	printf("<line x1='-2' y1='0' x2='2' y2='0' style='stroke:gray;stroke-width:2px;vector-effect:non-scaling-stroke;'/>");
	printf("<line x1='0' y1='-2' x2='0' y2='2' style='stroke:gray;stroke-width:2px;vector-effect:non-scaling-stroke;'/>");

	std::vector<vec2> vertices = testcase_08();
	std::vector<ivec3> trigs;

	auto t0 = std::chrono::high_resolution_clock::now();
	Delaunay_2d<vec2>().delaunay(vertices, trigs);
	auto t1 = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "vn = %d\nfn = %d\n", (int)vertices.size(), (int)trigs.size());
	fprintf(stderr, "%lf ms\n", 1000.*std::chrono::duration<double>(t1 - t0).count());


	for (int i = 0, vn = vertices.size(); i < vn; i++) {
		printf("<circle cx='%lg' cy='%lg' r='%lg' />",
			vertices[i].x, vertices[i].y, 2. / SC);
	}

	printf("<g style='stroke-width:%lgpx;stroke:black;fill:none;'>", 1. / SC);
	for (int i = 0, tn = trigs.size(); i < tn; i += 1) {
		vec2 p0 = vertices[trigs[i].x], p1 = vertices[trigs[i].y], p2 = vertices[trigs[i].z];
		printf("<polygon points='%lg,%lg %lg,%lg %lg,%lg' />",
			p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);
		if (abs(det(p1 - p0, p2 - p0)) < 1e-10)
			fprintf(stderr, "degenerated triangle\n");
	}
	printf("</g>");

	printf("</g></svg>");
	return 0;
}

