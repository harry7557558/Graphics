// delaunay triangulation in 2D

#include <vector>
#include <algorithm>

#include "numerical/geometry.h"



// duplicate points may lead to undefined behavior

class Delaunay_2d {

public:

	double epsilon, upsilon;

	triangle_2d supertriangle(const std::vector<vec2> &vertices) {
		vec2 pmin(INFINITY), pmax(-INFINITY);
		for (int i = vertices.size(); i--;) {
			vec2 p = vertices[i];
			pmin = pMin(p, pmin), pmax = pMax(p, pmax);
		}
		vec2 dp = pmax - pmin;
		double d = upsilon * max(dp.x, dp.y);
		vec2 mid = pmin + 0.5*dp;
		vec2 p1 = mid + vec2(0, 1)*d;
		vec2 p2 = mid + vec2(-0.866025, -0.5)*d;
		vec2 p3 = mid + vec2(0.866025, -0.5)*d;
		return triangle_2d(p1, p2, p3);
	}

	struct ccircle {
		int i, j, k;
		vec2 c; double r2;
	};
	ccircle circumcircle(const std::vector<vec2> &vertices, int i, int j, int k) {
		vec2 p1 = vertices[i], p2 = vertices[j], p3 = vertices[k];
		double fabsy1y2 = abs(p1.y - p2.y), fabsy2y3 = abs(p2.y - p3.y);

		vec2 c;

#if 0
		if (fabsy1y2 < epsilon && fabsy2y3 < epsilon)
			fprintf(stderr, "duplicate points\n");
#endif

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
				if (dx * dx + dy * dy > open[j].r2 + epsilon) {
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


	Delaunay_2d(double epsilon = (double)1e-10, double upsilon = (double)1e+4)
		: epsilon(epsilon), upsilon(upsilon) {}

};
