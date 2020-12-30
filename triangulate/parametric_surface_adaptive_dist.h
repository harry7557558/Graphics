#pragma once

// A header for adaptive parametric surface triangulation controlled by distance-based error.
// Not perfect, sub-samples are still missed; For temporary use.

// To use:

// std::vector<triangle> T = AdaptiveParametricSurfaceTriangulator_dist(P).triangulate_adaptive(u0, u1, v0, v1, un, vn, max_depth, tolerance, isUClose=false, isVClose=false);

// Parameters:

// @P: parametric equation, vec3 P(double u, double v)
// @u0, @u1, @v0, @v1 defines the parameter interval
// @un, @vn, @max_depth:
//   The triangulation routine first divides the parametric space into un×vn patches,
//     then perform recursive subdivision on patches;
//   Each subdivision divides the patch into 2x2 sub-patches, where @max_depth is the maximum recursive call depth;
// @tolerance:
//   A parameter use to determine if the local error is small enough;
//   The recursion routine stops subdividing if: for samples (uv1,p1) and (uv2,p2), the distance between 0.5*(p1+p2) and P(0.5*(uv1+uv2)) is less than @tolerance;
// @isUClose, @isVClose:
//   Parameters used in triangulation routine, indicate if the parameter is periodic;
// @isUClose: indicates whether for arbitrary v in [v0,v1], P(u0,v)==P(u1,v)
// @isVClose: indicates whether for arbitrary u in [u0,u1], P(u,v0)==P(u,v1)



#ifndef __INC_GEOMETRY_H
#include "numerical/geometry.h"
#endif


#include <vector>
#include <map>
#include <functional>


template<typename Fun>
class AdaptiveParametricSurfaceTriangulator_dist {

	// parametric surface sample struct
	struct sample {
		vec2 uv;
		vec3 p;
	};

	// data structure to store samples for surface reconstruction and to void duplicated samples
	std::map<vec2, vec3, std::function<bool(vec2, vec2)>> *samples_map;
	sample sample_m(vec2 uv) {
		sample s; s.uv = uv;
		auto d = samples_map->find(uv);
		if (d != samples_map->end()) s.p = d->second;
		else samples_map->operator[](uv) = s.p = fun(uv.x, uv.y);
		return s;
	}

	// surface patchs
	struct quadpatch {
		sample s[4];
		quadpatch() {}
		quadpatch(const sample ss[4]) {
			s[0] = ss[0], s[1] = ss[1], s[2] = ss[2], s[3] = ss[3];
		}
	};
	std::vector<quadpatch> Patches;

	// give three sample points along a line segment, test if it is accurate enough
	// sometimes this returns true due to coincidence
	bool isGoodEnough_line(vec3 a, vec3 m, vec3 b, double tol) {
		return (0.5*(a + b) - m).sqr() < tol*tol;
	}

	// parametric equation, P=fun(u,v)
	Fun fun;

	// store triangles
	std::vector<triangle_3d> Trigs;

public:

	// constructor and destructor
	AdaptiveParametricSurfaceTriangulator_dist(const Fun func) {
		samples_map = new std::map<vec2, vec3, std::function<bool(vec2, vec2)>>([](vec2 a, vec2 b) { return a.x<b.x ? true : a.x>b.x ? false : a.y < b.y; });
		fun = func;
	}
	~AdaptiveParametricSurfaceTriangulator_dist() {
		delete samples_map; samples_map = 0;
	}

	// user call function
	std::vector<triangle_3d> triangulate_adaptive(double u0, double u1, double v0, double v1, int un, int vn, int max_depth, double tolerance, bool isUClose = false, bool isVClose = false);

private:

	// recursive part in subdividing a patch
	void subdivide_quad(const sample s0[4], int remain_recurse, double tol);

	// recursive part in triangulating patches
	void addQuadPatch(const sample s[4]);
	void addTriPatch(const sample s[3]);
};


template<typename Fun>
std::vector<triangle_3d> AdaptiveParametricSurfaceTriangulator_dist<Fun>::triangulate_adaptive(
	double u0, double u1, double v0, double v1, int un, int vn, int max_depth, double tolerance, bool isUClose, bool isVClose) {
	samples_map->clear();
	Trigs.clear();

	// divide parametric space into un×vn patches
	for (int ui = 0; ui < un; ui++) {
		for (int vi = 0; vi < vn; vi++) {
			// try to avoid floating point inaccuracy
			double ut0 = (double)ui / (double)un, ut1 = (double)(ui + 1) / (double)un;
			double vt0 = (double)vi / (double)vn, vt1 = (double)(vi + 1) / (double)vn;
			double u_0 = u0 * (1. - ut0) + u1 * ut0;
			double u_1 = u0 * (1. - ut1) + u1 * ut1;
			double v_0 = v0 * (1. - vt0) + v1 * vt0;
			double v_1 = v0 * (1. - vt1) + v1 * vt1;
			// subdivide patch adaptively
			sample s[4] = { sample_m(vec2(u_0,v_0)), sample_m(vec2(u_1,v_0)), sample_m(vec2(u_1,v_1)), sample_m(vec2(u_0,v_1)) };
			subdivide_quad(s, max_depth, tolerance);
		}
	}

	// if the parameter forms a loop, repeat samples at the boundary
	std::vector<sample> toAdd;
	if (isUClose)
		for (auto it = samples_map->begin(); it != samples_map->end(); it++) {
			if (it->first.x == u0) toAdd.push_back(sample{ vec2(u1,it->first.y), it->second });
			if (it->first.x == u1) toAdd.push_back(sample{ vec2(u0,it->first.y), it->second });
		}
	if (isVClose)
		for (auto it = samples_map->begin(); it != samples_map->end(); it++) {
			if (it->first.y == v0) toAdd.push_back(sample{ vec2(it->first.x,v1), it->second });
			if (it->first.y == v1) toAdd.push_back(sample{ vec2(it->first.x,v0), it->second });
		}
	for (auto it = toAdd.begin(); it != toAdd.end(); it++)
		samples_map->operator[](it->uv) = it->p;

	// triangulate patches
	int PN = Patches.size();
	for (int i = 0; i < PN; i++) {
		addQuadPatch(Patches[i].s);
	}

	return Trigs;
}

template<typename Fun>
void AdaptiveParametricSurfaceTriangulator_dist<Fun>::subdivide_quad(const sample s0[4], int remain_recurse, double tol) {

	// recursion limit exceeded - construct triangles
	if (!(remain_recurse > 0)) {
		Patches.push_back(quadpatch(s0));
		return;
	}

	// all samples needed
	sample ss[9] = {
		s0[0], s0[1], s0[2], s0[3],
		sample_m(0.5*(s0[0].uv + s0[1].uv)), sample_m(0.5*(s0[1].uv + s0[2].uv)), sample_m(0.5*(s0[2].uv + s0[3].uv)), sample_m(0.5*(s0[3].uv + s0[0].uv)),
		sample_m(0.25*(s0[0].uv + s0[1].uv + s0[2].uv + s0[3].uv))
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
		sample s[4] = { ss[trigs[c]], ss[trigs[c + 1]], ss[trigs[c + 2]], ss[trigs[c + 3]] };
		Patches.push_back(quadpatch(s));
	}

}

template<typename Fun>
void AdaptiveParametricSurfaceTriangulator_dist<Fun>::addQuadPatch(const sample s[4]) {

	// check edges to see if they are subdivided
	vec2 edges[4] = {
		0.5*(s[0].uv + s[1].uv), 0.5*(s[1].uv + s[2].uv), 0.5*(s[2].uv + s[3].uv), 0.5*(s[3].uv + s[0].uv)
	};
	std::map<vec2, vec3>::iterator edge_ip[4];
	bool hasMid[4];
	for (int i = 0; i < 4; i++) {
		edge_ip[i] = samples_map->find(edges[i]);
		hasMid[i] = (edge_ip[i] != samples_map->end());
	}
	sample ss[9] = { s[0], s[1], s[2], s[3] };
	for (int i = 0; i < 4; i++) {
		ss[i + 4] = hasMid[i] ? sample{ edge_ip[i]->first, edge_ip[i]->second } : sample{ vec2(NAN), vec3(NAN) };
	}

	// calculate the index of the situation
	int situation_index = int(hasMid[0]) + 2 * int(hasMid[1]) + 4 * int(hasMid[2]) + 8 * int(hasMid[3]);
	if (situation_index == 0) {
		// no edge splitted, normal triangulation
		if ((s[0].p - s[2].p).sqr() > (s[1].p - s[3].p).sqr()) {
			if (s[3].p != s[0].p && s[3].p != s[1].p && s[0].p != s[1].p)
				Trigs.push_back(triangle_3d(s[3].p, s[0].p, s[1].p));
			if (s[1].p != s[2].p && s[1].p != s[3].p && s[2].p != s[3].p)
				Trigs.push_back(triangle_3d(s[1].p, s[2].p, s[3].p));
		}
		else {
			if (s[2].p != s[3].p && s[2].p != s[0].p && s[3].p != s[0].p)
				Trigs.push_back(triangle_3d(s[2].p, s[3].p, s[0].p));
			if (s[0].p != s[1].p && s[0].p != s[2].p && s[1].p != s[2].p)
				Trigs.push_back(triangle_3d(s[0].p, s[1].p, s[2].p));
		}
		return;
	}
	if (situation_index == 15) {
		// take an additional sample at the center
		ss[8] = sample_m(0.25*(s[0].uv + s[1].uv + s[2].uv + s[3].uv));
	}

	// lookup tables to generate triangles and quads
	const static int triangle_table[16][12] = {
		{ -1 },  // 0000
		{ 0,4,3, 3,4,2, 2,4,1, -1 },  // 1000
		{ 1,5,0, 0,5,3, 3,5,2, -1 },  // 0100
		{ 0,4,3, 4,1,5, 5,2,3, 4,5,3 },  // 1100
		{ 2,6,1, 1,6,0, 0,6,3, -1 },  // 0010
		{ -1 },  // 1010
		{ 1,5,0, 5,2,6, 6,3,0, 5,6,0 },  // 0110
		{ 1,5,4, 4,5,6, 6,5,2, -1 },  // 1110
		{ 3,7,2, 2,7,1, 1,7,0, -1 },  // 0001
		{ 3,7,2, 7,0,4, 4,1,2, 7,4,2 },  // 1001
		{ -1 },  // 0101
		{ 0,4,7, 7,4,5, 5,4,1, -1 },  // 1101
		{ 2,6,1, 6,3,7, 7,0,1, 6,7,1 },  // 0011
		{ 3,7,6, 6,7,4, 4,7,0, -1 },  // 1011
		{ 2,6,5, 5,6,7, 7,6,3, -1 },  // 0111
		{ -1 },  // 1111
	};
	const static int quad_table[16][16] = {
		{ -1 },  // 0000
		{ -1 },  // 1000
		{ -1 },  // 0100
		{ -1 },  // 1100
		{ -1 },  // 0010
		{ 0,4,6,3, 4,1,2,6, -1 },  // 1010
		{ -1 },  // 0110
		{ 0,4,6,3, -1},  // 1110
		{ -1 },  // 0001
		{ -1 },  // 1001
		{ 0,1,5,7, 7,5,2,3, -1 },  // 0101
		{ 7,5,2,3, -1 },  // 1101
		{ -1 },  // 0011
		{ 4,1,2,6, -1 },  // 1011
		{ 0,1,5,7, -1 },  // 0111
		{ 0,4,8,7, 4,1,5,8, 8,5,2,6, 7,8,6,3 },  // 1111
	};

	// take actions to the situation
	auto triangles = triangle_table[situation_index];
	for (int i = 0; i < 12 && triangles[i] != -1; i += 3) {
		sample st[3] = { ss[triangles[i]], ss[triangles[i + 1]], ss[triangles[i + 2]] };
		addTriPatch(st);
	}
	auto quads = quad_table[situation_index];
	for (int i = 0; i < 16 && quads[i] != -1; i += 4) {
		sample st[4] = { ss[quads[i]], ss[quads[i + 1]], ss[quads[i + 2]], ss[quads[i + 3]] };
		addQuadPatch(st);
	}

}

template<typename Fun>
void AdaptiveParametricSurfaceTriangulator_dist<Fun>::addTriPatch(const sample s[3]) {
	// similar to addQuadPatch() except there are only three vertices
	// not widely tested, possibly has bug

	// check edges to see if they are subdivided
	vec2 edges[3] = {
		0.5*(s[0].uv + s[1].uv), 0.5*(s[1].uv + s[2].uv), 0.5*(s[2].uv + s[0].uv)
	};
	std::map<vec2, vec3>::iterator edge_ip[3];
	bool hasMid[3];
	for (int i = 0; i < 3; i++) {
		edge_ip[i] = samples_map->find(edges[i]);
		hasMid[i] = (edge_ip[i] != samples_map->end());
	}
	sample ss[6] = { s[0], s[1], s[2] };
	for (int i = 0; i < 3; i++) {
		ss[i + 3] = hasMid[i] ? sample{ edge_ip[i]->first, edge_ip[i]->second } : sample{ vec2(NAN), vec3(NAN) };
	}

	// calculate the index of the situation
	int situation_index = int(hasMid[0]) + 2 * int(hasMid[1]) + 4 * int(hasMid[2]);
	if (situation_index == 0) {
		Trigs.push_back(triangle_3d(s[0].p, s[1].p, s[2].p));
		return;
	}

	// lookup table to generate triangles
	const static int triangle_table[8][12] = {
		{ -1 },  // 000
		{ 0,3,2, 2,3,1, -1},  // 100
		{ 1,4,0, 0,4,2, -1},  // 010
		{ 3,1,4, -1},  // 110
		{ 2,5,1, 1,5,0, -1},  // 001
		{ 0,3,5, -1},  // 101
		{ 2,5,4, -1},  // 011
		{ 0,3,5, 1,4,3, 2,5,4, 3,4,5 },  // 111
	};
	const static int quad_table[8][4] = {
		{ -1 },  // 000
		{ -1 },  // 100
		{ -1 },  // 010
		{ 0,3,4,2 },  // 110
		{ -1 },  // 001
		{ 3,1,2,5 },  // 101
		{ 0,1,4,5 },  // 011
		{ -1 },  // 111
	};

	// take actions to the situation
	auto triangles = triangle_table[situation_index];
	for (int i = 0; i < 12 && triangles[i] != -1; i += 3) {
		sample st[3] = { ss[triangles[i]], ss[triangles[i + 1]], ss[triangles[i + 2]] };
		addTriPatch(st);
	}
	auto quads = quad_table[situation_index];
	for (int i = 0; i < 4 && quads[i] != -1; i += 4) {
		sample st[4] = { ss[quads[i]], ss[quads[i + 1]], ss[quads[i + 2]], ss[quads[i + 3]] };
		addQuadPatch(st);
	}
}

