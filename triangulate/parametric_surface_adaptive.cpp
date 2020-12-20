// incompleted attempt to adaptively triangulate parametric surfaces

#include "numerical/geometry.h"

// parametric equation, 0 < u,v < 1
#include "modeling/generators/parametric/surfaces.h"
vec3 fun(double u, double v) {
	//return vec3(cos(u)*cossin(v), sin(u));
	//return vec3(cos(u)*cossin(5.*pow(v, 10.)), sin(u));
	//return vec3((1. + cos(2.*u))*cossin(v), sin(2.*u));
	//return vec3((1.0 + cos(3.*u))*cossin(5.*v), sin(3.*u));
	//return vec3((1. + cos(2.*u))*cossin(v), sin(2.*u) - exp(-1000.*(1. - 2.*u)*(1. - 2.*u)));
	//return vec3(v, u, u*v);
	return vec3(v + 0.2*u*u, u - 0.2*tan(v), tanh(1 - sin(3.*v)*cos(2.*u)));
	//auto S = &ParamSurfaces[21]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // uneven torus
	//auto S = &ParamSurfaces[23]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // deformed torus (star)
	//auto S = &ParamSurfaces[29]; return 0.5*S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // twisted torus
	//auto S = &ParamSurfaces[30]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // toric heart
	//auto S = &ParamSurfaces[32]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // egg
	//auto S = &ParamSurfaces[33]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // droplet
	//auto S = &ParamSurfaces[34]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // bowling
	//auto S = &ParamSurfaces[35]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // gourd
	//auto S = &ParamSurfaces[36]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // pumpkin
	//auto S = &ParamSurfaces[37]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // pepper without stem
	//auto S = &ParamSurfaces[38]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // ball cactus
	//auto S = &ParamSurfaces[39]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));  // ice cream
	//auto S = &ParamSurfaces[1]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[4]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[6]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[12]; return 0.3*S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[18]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[19]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[25]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
	//auto S = &ParamSurfaces[27]; return S->P(mix(S->u0, S->u1, u), mix(S->v0, S->v1, v));
}


// this will store discretized triangles
#include <vector>
std::vector<triangle> Trigs;
#include "ui/stl_encoder.h"
std::vector<stl_triangle> Shape;

void drawDot(vec3 p, double r, vec3 col) {
	return;
	for (int i = 0; i < 8; i++) {
		vec3 v = vec3(0, 0, i >= 4 ? -1. : 1.);
		vec3 e1 = vec3(cossin(.5*PI*i));
		vec3 e2 = vec3(cossin(.5*PI*(i + v.z)));
		Shape.push_back(stl_triangle(p + r * v, p + r * e2, p + r * e1, col));
	}
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

void subdivide_quad(const sample s0[4], int remain_recurse, double tol);

bool isGoodEnough_line(vec3 a, vec3 m, vec3 b, double tol) {
	return (0.5*(a + b) - m).sqr() < tol*tol;
	//return (a == m && b == m) || ((0.5*(a + b) - m).sqr() < tol*tol && length(cross(normalize(m - a), normalize(b - m))) < 10.*tol);
}

void triangulate_adaptive(int un, int vn, int max_depth, double tolerance) {
	sample s[4] = { sample(0,0), sample(1,0), sample(1,1), sample(0,1) };
	subdivide_quad(s, max_depth, tolerance);

	printf("%d\n", samples_map.size());
	for (auto it = samples_map.begin(); it != samples_map.end(); it++) {
		drawDot(it->second, tolerance, vec3(1, 1, 0));
	}
}

void subdivide_quad(const sample s0[4], int remain_recurse, double tol) {

	// recursion limit exceeded - construct triangles
	if (!(remain_recurse > 0)) {
		if ((s0[0].p - s0[2].p).sqr() < (s0[1].p - s0[3].p).sqr())
			Shape.push_back(stl_triangle(s0[3].p, s0[0].p, s0[2].p, vec3(1, 0, 0))),
			Shape.push_back(stl_triangle(s0[2].p, s0[0].p, s0[1].p, vec3(1, 0, 0)));
		else
			Shape.push_back(stl_triangle(s0[3].p, s0[0].p, s0[1].p, vec3(1, 0, 0))),
			Shape.push_back(stl_triangle(s0[1].p, s0[2].p, s0[3].p, vec3(1, 0, 0)));
		return;
	}

	// all samples needed
	sample ss[9] = {
		s0[0], s0[1], s0[2], s0[3],
		sample(0.5*(s0[0].uv + s0[1].uv)), sample(0.5*(s0[1].uv + s0[2].uv)), sample(0.5*(s0[2].uv + s0[3].uv)), sample(0.5*(s0[3].uv + s0[0].uv)),
		sample(0.25*(s0[0].uv + s0[1].uv + s0[2].uv + s0[3].uv))
	};

	// list of edges in the diagram - line segments connecting two points
	const static int edges[12][2] = {
		{0,4}, {4,1}, {1,5}, {5,2}, {2,6}, {6,3}, {3,7}, {7,0}, {7,8}, {8,5}, {4,8}, {8,6}
	};
	// list of "double edges" - three points in a line
	const static int double_edges[6][3] = {
		{0,4,1}, {1,5,2}, {2,6,3}, {3,7,0}, {7,8,5}, {4,8,6}
	};
	const static int double_edges_edge[6][2] = {
		{0,1}, {2,3}, {4,5}, {6,7}, {8,9}, {10,11}
	};

	// check the "double edges" to see if their errors are small enough
	bool isAccurateEnough[6];
	for (int i = 0; i < 6; i++)
		isAccurateEnough[i] = isGoodEnough_line(ss[double_edges[i][0]].p, ss[double_edges[i][1]].p, ss[double_edges[i][2]].p, tol);

	// calculate the current situation for lookup table
	int situation_index = 0;
	for (int i = 0; i < 6; i++)
		situation_index |= (int(!isAccurateEnough[i]) << i);
	// lookup table: square construction, max 4 squares
	//  - 0: accurate enough; 1: not accurate enough
	//  - all accurate enough: no sub-sampling
	//  - only one side not accurate enough: sub-sample two attached squares
	//  - two adjacent sides not accurate enough: sub-sample three attached squares
	//  - two opposite sides not accurate enough: divide into two long rectangles
	//  - any "crossing" side not accurate enough: sub-sample all four squares
	const static int subsquare_table[64][16] = {
		{ -1 },  // 0000
		{ 0,4,8,7, 4,1,5,8, -1 },  // 1000
		{ 4,1,5,8, 8,5,2,6, -1 },  // 0100
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, -1 },  // 1100
		{ 8,5,2,6, 7,8,6,3, -1 },  // 0010
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 1010
		{ 4,1,5,8, 8,5,2,6, 7,8,6,3, -1 },  // 0110
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 1110
		{ 7,8,6,3, 0,4,8,7, -1 },  // 0001
		{ 7,8,6,3, 0,4,8,7, 4,1,5,8, -1 },  // 1001
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 0101
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 1101
		{ 8,5,2,6, 7,8,6,3, 0,4,8,7, -1 },  // 0011
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 1011
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 0111
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 1111
		{-1},{-1},{-1},{-1},{-1},  // all { 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 }
		{ 0,4,6,3, 4,1,2,6, -1 },  // 101010
		{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},  // subdivide all
		{ 0,1,5,7, 7,5,2,3, -1 },  // 010101
		{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1}  // subdivide all
	};
	// squares spaces to fill
	const static int fillsquare_table[64][24] = {
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 0000
		{ 8,5,2,6, 7,8,6,3, -1 },  // 1000
		{ 7,8,6,3, 0,4,8,7, -1 },  // 0100
		{ 7,8,6,3, -1},  // 1100
		{ 0,4,8,7, 4,1,5,8, -1 },  // 0010
		{ -1 },  // 1010
		{ 0,4,8,7, -1 },  // 0110
		{ -1 },  // 1110
		{ 4,1,5,8, 8,5,2,6, -1 },  // 0001
		{ 8,5,2,6, -1 },  // 1001
		{ -1 },  // 0101
		{ -1 },  // 1101
		{ 4,1,5,8, -1 },  // 0011
		{ -1 },  // 1011
		{ -1 },  // 0111
		{ -1 },  // 1111
		{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},
		{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},
		{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},{-1},
	};

	// "decompress" the table
	if (situation_index >= 16 && subsquare_table[situation_index][0] == -1) {
		situation_index = 15;  // subdivide all squares
	}
	// prevent termination condition satisfication caused by coincidence
	// (some coincidence cases still passes)
	if (situation_index == 0 && !(
		isGoodEnough_line(ss[0].p, ss[8].p, ss[2].p, 1.4142*tol)
		&& isGoodEnough_line(ss[1].p, ss[8].p, ss[3].p, 1.4142*tol))) {
		situation_index = 15;
	}

	// take actions to the situation

	// subdivide squares with too large error
	auto subsquares = subsquare_table[situation_index];
	for (int c = 0; c < 16 && subsquares[c] != -1; c += 4) {
		sample sd[4] = { ss[subsquares[c]], ss[subsquares[c + 1]], ss[subsquares[c + 2]], ss[subsquares[c + 3]] };
		subdivide_quad(sd, remain_recurse - 1, tol);
	}

	// fill empty spaces
	auto trigs = fillsquare_table[situation_index];
	for (int c = 0; c < 16 && trigs[c] != -1; c += 4) {
		vec3 p[4] = { ss[trigs[c]].p, ss[trigs[c + 1]].p, ss[trigs[c + 2]].p, ss[trigs[c + 3]].p };
		if ((p[0] - p[2]).sqr() < (p[1] - p[3]).sqr())
			Shape.push_back(stl_triangle(p[3], p[0], p[2], vec3(.9, .8, .7))),
			Shape.push_back(stl_triangle(p[2], p[0], p[1], vec3(.9, .8, .7)));
		else
			Shape.push_back(stl_triangle(p[0], p[1], p[3], vec3(.9, .8, .7))),
			Shape.push_back(stl_triangle(p[3], p[1], p[2], vec3(.9, .8, .7)));
	}

}






int main(int argc, char* argv[]) {
	triangulate_adaptive(1, 1, 16, 0.01);

	//writeSTL(argv[1], &Trigs[0], Trigs.size(), nullptr, "bac");
	writeSTL(argv[1], &Shape[0], Shape.size(), nullptr, "cba");
	samples_map.clear(); Trigs.clear(); Shape.clear();
	return 0;
}

