// failed attempt for adaptive parametric surface triangulation

#include "numerical/geometry.h"

// parametric equation, 0 < u,v < 1
vec3 fun(double u, double v) {
	//return vec3(cos(v)*cossin(u), sin(v));
	//return vec3((1. + cos(2.*v))*cossin(u), sin(2.*v));
	return vec3(u, v, u*v);
	return vec3(u + 0.2*v*v, v - 0.2*tan(u), tanh(1 - sin(3.*u)*cos(2.*v)));
}


// this will store discretized triangles
#include <vector>
std::vector<triangle> Trigs;
#include "ui/stl_encoder.h"
std::vector<stl_triangle> Shape;


// non-adaptive triangulation
void triangulate(int un, int vn) {
	vec3 *p = new vec3[(un + 1)*(vn + 1)];
	for (int ui = 0; ui <= un; ui++) {
		for (int vi = 0; vi <= vn; vi++) {
			double u = ui / double(un), v = vi / double(vn);
			p[ui*(vn + 1) + vi] = fun(u, v);
		}
	}
	Trigs.reserve(Trigs.size() + 2 * un*vn);
	for (int ui = 0; ui < un; ui++) {
		for (int vi = 0; vi < vn; vi++) {
			vec3 p00 = p[ui*(vn + 1) + vi];
			vec3 p01 = p[ui*(vn + 1) + (vi + 1)];
			vec3 p10 = p[(ui + 1)*(vn + 1) + vi];
			vec3 p11 = p[(ui + 1)*(vn + 1) + (vi + 1)];
			if ((p01 - p10).sqr() < (p00 - p11).sqr()) {
				Trigs.push_back(triangle{ p10, p00, p01 });
				Trigs.push_back(triangle{ p01, p11, p10 });
			}
			else {
				Trigs.push_back(triangle{ p11, p10, p00 });
				Trigs.push_back(triangle{ p00, p01, p11 });
			}
		}
	}
	delete p;
}




// adaptive subdivision
// mean to be OO when written into a header :^|

#include <map>
#include <functional>
std::map<vec2, vec3, std::function<bool(vec2, vec2)>> samples_map([](vec2 a, vec2 b) { return a.x<b.x ? true : a.x>b.x ? false : a.y < b.y; });

struct sample {
	vec2 uv;
	vec3 p;
	sample() {}
	sample(vec2 uv) :uv(uv) {
		auto d = samples_map.find(uv);
		if (d != samples_map.end()) p = d->second;
		else samples_map[uv] = p = fun(uv.x, uv.y);
	}
	sample(double u, double v) {
		uv = vec2(u, v);
		auto d = samples_map.find(uv);
		if (d != samples_map.end()) p = d->second;
		else samples_map[uv] = p = fun(u, v);
	}
};

void subdivide_quad(const sample s[4], int remain_recurse, double tol);
void subdivide_trig(const sample s[4], int remain_recurse, double tol);

bool isGoodEnough_line(vec3 a, vec3 m, vec3 b, double tol) {
	return (0.5*(a + b) - m).sqr() < tol*tol;
}

void triangulate_adaptive(int un, int vn, int max_depth, double tolerance) {
	sample s[4] = { sample(0,0), sample(1,0), sample(1,1), sample(0,1) };
	subdivide_quad(s, max_depth, tolerance);
}

void subdivide_quad(const sample s[4], int remain_recurse, double tol) {
	sample ss[5];
	sample s1[3] = { s[1], s[0], s[3] };
	sample s2[3] = { s[3], s[2], s[1] };
	subdivide_trig(s1, remain_recurse, tol);
	subdivide_trig(s2, remain_recurse, tol);
}


void subdivide_trig(const sample s[3], int remain_recurse, double tol) {

	// recursion limit exceeded
	if (remain_recurse == 0) {
		Trigs.push_back(triangle{ s[0].p, s[1].p, s[2].p });
		return;
	}

	// new samples - ss[6] will be 1./3.*(s[0]+s[1]+s[2])
	sample ss[7] = { s[0], s[1], s[2],
		sample(0.5*(s[0].uv + s[1].uv)), sample(0.5*(s[1].uv + s[2].uv)), sample(0.5*(s[2].uv + s[0].uv)),
		sample((1. / 3.)*(s[0].uv + s[1].uv + s[2].uv)) };

	// list of edges: a, b, mid
	const static int edgeList[3][3] = {
		{ 0,1, 3 }, { 1,2, 4 }, { 2,0, 5 }
	};
	// are these edges accurate enough?
	bool edge_ok[3] = { false, false, false };
	for (int i = 0; i < 3; i++)
		edge_ok[i] = isGoodEnough_line(s[edgeList[i][0]].p, ss[edgeList[i][2]].p, s[edgeList[i][1]].p, tol);

	// actions to different situations
	int situation = int(edge_ok[0]) + 2 * int(edge_ok[1]) + 4 * int(edge_ok[2]);
	// situation lookup table: sub-triangle construction (max 4 triangles, >10 means go recursive)
	const static int subtrig_table[8][12] = {
		{ 15,10,13, 13,11,14, 14,12,15, 13,14,15 },  // 000
		{ 11,16,10, 12,16,11, 10,16,12, -1 },  // 100
		{ 11,16,10, 12,16,11, 10,16,12, -1 },  // 010
		{ 12,15,11, 11,15,10, -1 },  // 110
		{ 11,16,10, 12,16,11, 10,16,12, -1 },  // 001
		{ 11,14,10, 10,14,12, -1 },  // 101
		{ 10,13,12, 12,13,11, -1 },  // 011
		{ 0,1,2, -1 },  // 111
	};

	// decode the lookup table and take actions
	auto subtrigs = subtrig_table[situation];
	for (int i = 0; i < 12 && subtrigs[i] != -1; i += 3) {
		// recursive subdivition
		if (subtrigs[i] >= 10) {
			sample st[3] = { ss[subtrigs[i] - 10], ss[subtrigs[i + 1] - 10], ss[subtrigs[i + 2] - 10] };
			subdivide_trig(st, remain_recurse - 1, tol);
		}
		// add triangle
		else {
			Trigs.push_back(triangle{ ss[subtrigs[i]].p, ss[subtrigs[i + 1]].p, ss[subtrigs[i + 2]].p });
		}
	}
}





int main(int argc, char* argv[]) {
	//triangulate(16, 16);
	triangulate_adaptive(1, 1, 20, 0.001);

	writeSTL(argv[1], &Trigs[0], Trigs.size(), nullptr, "bac");
	samples_map.clear(); Trigs.clear(); Shape.clear();
	return 0;
}

