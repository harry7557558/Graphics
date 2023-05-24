// reference https://github.com/darkskyapp/delaunay-fast
// to-do: understand how this algorithm works

#include <stdio.h>
#include <vector>
#include <algorithm>
#include "numerical/geometry.h"
#include "numerical/random.h"


const double epsilon = 1e-6;

triangle_2d supertriangle(const std::vector<vec2> &vertices) {
	// equilateral triangle
	vec2 n[3] = { vec2(0, -1), vec2(0.8660254037844386, 0.5), vec2(-0.8660254037844386, 0.5) };
	double d[3] = { -INFINITY, -INFINITY, -INFINITY };
	for (int i = 0; i < (int)vertices.size(); i++) {
		for (int u = 0; u < 3; u++) {
			double di = dot(n[u], vertices[i]);
			d[u] = max(d[u], di);
		}
	}
	vec2 p[3];
	for (int u = 0; u < 3; u++)
		p[u] = mat2(n[u], n[(u + 1) % 3]).transpose().inverse() * vec2(d[u], d[(u + 1) % 3]);
	vec2 c = (p[0] + p[1] + p[2]) / 3.;
	for (int u = 0; u < 3; u++) p[u] = c + (p[u] - c)*1.2;
	return triangle_2d(p[0], p[1], p[2]);
}

struct ccircle {
	int i, j, k;
	vec2 c; double r2;
};
ccircle circumcircle(const std::vector<vec2> &vertices, int i, int j, int k) {
	vec2 p1 = vertices[i], p2 = vertices[j], p3 = vertices[k];
	double fabsy1y2 = abs(p1.y - p2.y), fabsy2y3 = abs(p2.y - p3.y);

	vec2 c;

	if (fabsy1y2 < epsilon && fabsy2y3 < epsilon)
		throw "coincident points";

	if (fabsy1y2 < epsilon) {
		double m2 = -((p3.x - p2.x) / (p3.y - p2.y));
		vec2 mp2 = 0.5*(p2 + p3);
		c.x = 0.5*(p1.x + p2.x);
		c.y = m2 * (c.x - mp2.x) + mp2.y;
	}
	else if (fabsy2y3 < epsilon) {
		double m1 = -((p2.x - p1.x) / (p2.y - p1.y));
		vec2 mp1 = 0.5*(p1 + p2);
		c.x = 0.5*(p2.x + p3.x);
		c.y = m1 * (c.x - mp1.x) + mp1.y;
	}

	else {
		double m1 = -((p2.x - p1.x) / (p2.y - p1.y));
		double m2 = -((p3.x - p2.x) / (p3.y - p2.y));
		vec2 mp1 = 0.5*(p1 + p2), mp2 = 0.5*(p2 + p3);
		c.x = (m1 * mp1.x - m2 * mp2.x + mp2.y - mp1.y) / (m1 - m2);
		c.y = fabsy1y2 > fabsy2y3 ?
			m1 * (c.x - mp1.x) + mp1.y :
			m2 * (c.x - mp2.x) + mp2.y;
	}

	return ccircle{ i, j, k, c, (p2 - c).sqr() };
}

void delaunay(std::vector<vec2> vertices, std::vector<ivec3> &trigs) {
	trigs.clear();
	int N = (int)vertices.size();
	if (N < 3) return;

	if (N == 3) {
		// imagine you get an empty triangle list in this case
		trigs.push_back(ivec3(0, 1, 2));
		return;
	}

	std::vector<int> indices(N);
	for (int i = 0; i < N; i++) indices[i] = i;
	std::sort(indices.begin(), indices.end(), [&](int i, int j) {
		return vertices[i].x == vertices[j].x ? vertices[i].y > vertices[j].y : vertices[i].x > vertices[j].x;
	});

	triangle_2d st = supertriangle(vertices);
	vertices.push_back(st[0]); vertices.push_back(st[1]); vertices.push_back(st[2]);

	std::vector<ccircle> open, closed;
	open.push_back(circumcircle(vertices, N, N + 1, N + 2));

	for (int i = (int)indices.size(); i--;) {
		std::vector<ivec2> edges;
		int c = indices[i];

		for (int j = (int)open.size(); j--;) {
			// right of the circumcircle
			double dx = vertices[c].x - open[j].c.x;
			if (dx > 0. && dx*dx > open[j].r2) {
				closed.push_back(open[j]);
				open.erase(open.begin() + j);
				continue;
			}
			// outside the circumcircle
			double dy = vertices[c].y - open[j].c.y;
			if (dx * dx + dy * dy - open[j].r2 > epsilon) {
				continue;
			}
			// left of the circumcircle
			edges.push_back(ivec2(open[j].i, open[j].j));
			edges.push_back(ivec2(open[j].j, open[j].k));
			edges.push_back(ivec2(open[j].k, open[j].i));
			open.erase(open.begin() + j);
		}

		// remove duplicate edges
		for (int j = 0; j < (int)edges.size(); j++) {
			for (int i = j + 1; i < (int)edges.size(); i++) {
				if ((edges[j] == edges[i]) || (edges[j].yx() == edges[i])) {
					edges.erase(edges.begin() + i);
					edges.erase(edges.begin() + j);
					j--; break;
				}
			}
		}

		for (int j = edges.size(); j;) {
			ivec2 e = edges[--j];
			open.push_back(circumcircle(vertices, e.x, e.y, c));
		}
	}

	closed.insert(closed.end(), open.begin(), open.end());
	open.clear();

	trigs.clear();
	for (int i = closed.size(); i--;) {
		if (closed[i].i < N && closed[i].j < N && closed[i].k < N) {
			trigs.push_back(ivec3(closed[i].i, closed[i].j, closed[i].k));
		}
	}
}


#include <chrono>

std::vector<vec2> randomPoints(int N, uint32_t seed) {
	std::vector<vec2> res;
	for (int i = 0; i < N; i++) {
		vec2 p = rand2_u(seed);
		res.push_back(p);
	}
	return res;
}

int main() {
	freopen("D:\\.svg", "w", stdout);
	printf("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='%d' height='%d'>\n", 600, 600);
	printf("<g transform='translate(%d,%d) scale(%lg,%lg)'>", 300, 300, 200., -200.);
	printf("<line x1='-2' y1='0' x2='2' y2='0' style='stroke:gray;stroke-width:2px;vector-effect:non-scaling-stroke;'/>");
	printf("<line x1='0' y1='-2' x2='0' y2='2' style='stroke:gray;stroke-width:2px;vector-effect:non-scaling-stroke;'/>");

	std::vector<vec2> vertices = randomPoints(100000, 1);
	std::vector<ivec3> trigs;

	auto t0 = std::chrono::high_resolution_clock::now();
	delaunay(vertices, trigs);
	auto t1 = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "%lf ms\n", 1000.*std::chrono::duration<double>(t1 - t0).count());


	for (int i = 0, vn = vertices.size(); i < vn; i++) {
		printf("<circle cx='%lg' cy='%lg' r='%lg' />",
			vertices[i].x, vertices[i].y, 3 / 200.);
	}

	printf("<g style='stroke-width:0.005px;stroke:black;fill:none;'>");
	for (int i = 0, tn = trigs.size(); i < tn; i++) {
		printf("<polygon points='%lg,%lg %lg,%lg %lg,%lg' />",
			vertices[trigs[i].x].x, vertices[trigs[i].x].y,
			vertices[trigs[i].y].x, vertices[trigs[i].y].y,
			vertices[trigs[i].z].x, vertices[trigs[i].z].y);
	}
	printf("</g>");

	printf("</g></svg>");
	return 0;
}

