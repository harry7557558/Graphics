// generate random closed surface (triangulated)

#pragma GCC optimize "Ofast"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "numerical\random.h"



struct segment {
	vec3 p, q;
	vec3 dir() const { return q - p; };
};
struct triangle {
	vec3 a, b, c;
	vec3 normal() const { return cross(b - a, c - a); }
	vec3 unitnormal() const { return normalize(cross(b - a, c - a)); }
	double area() const { return 0.5*length(cross(b - a, c - a)); }
};


// =================== Mesh DS with Continuity ===================

//                                V[2]
//    
//                               XXX
//                             XX   X
//                           X       XX
//           F[2]          XX         XX        F[0]
//                 E[2]  XX             XX E[0]
//                     X                  X                cross(v[0]-v[1],v[2]-v[1]) outward
//                   XX      v     v       X               (different from struct triangle)
//                XX            ^           XX
//              X                             XX
//           XX                                 XXX
//         X                E[1]       XXXXXXXXXX XX
//        XX XX X XXXX XXXXXXXXXXXXXXXX               V[0]
//    V[1]
//                        F[1]

//             V[1]
//
//              X
//              X
//             X                not sure how it can be helpful
//              X               but please, v[0] < v[1]
//          ~  X  ~
//             X
//    F[1]     X    F[0]
//             X
//             X
//            X
//            X
//            X
//   
//           V[0]

// this is not a pleasant coding experience...
// maybe i can do better...


// closed triangular mesh
// one day, I may move it ↓ to a separate header file

struct mesh {
	struct vertex : public vec3 {
		int e = -1;  // index of any connected edge
		vertex() {}
		vertex(vec3 p, int e = -1) :e(e) { x = p.x, y = p.y, z = p.z; }
		void setP(vec3 p) { x = p.x, y = p.y, z = p.z; }
	};
	struct edge {
		int v[2] = { -1, -1 };  // index of vertexes in increasing order
		int f[2] = { -1, -1 };  // index of triangular faces; order matters
		// The first face should have the same vertex order and the second one has reversed order
		// Eg. if v is {0,1}, then {0,1,2} should be the first face and {1,0,3} should be the second one
		// In other words, the first triangle is on the "right" side of the edge
		edge() {}
		edge(int v0, int v1, int f0 = -1, int f1 = -1) { v[0] = v0, v[1] = v1, f[0] = f0, f[1] = f1; }
		bool existVertex(int id) const { return v[0] == id || v[1] == id; }
		bool existFace(int id) const { return f[0] == id || f[1] == id; }
		int anotherVertex(int vi) const { return v[0] == vi ? v[1] : v[1] == vi ? v[0] : -1; }
		int anotherFace(int fi) const { return f[0] == fi ? f[1] : f[1] == fi ? f[0] : -1; }
		// call autoSwap() after
		void swap() { std::swap(v[0], v[1]), std::swap(f[0], f[1]); }
		void autoSwap() { if (v[0] > v[1]) std::swap(v[0], v[1]), std::swap(f[0], f[1]); }
		void replaceVertex(int v0, int v1) { for (int i = 0; i < 2; i++) if (v[i] == v0) v[i] = v1; }
		void replaceFace(int f0, int f1) { for (int i = 0; i < 2; i++) if (f[i] == f0) f[i] = f1; }
	};
	struct face {
		int v[3] = { -1, -1, -1 };  // index of vertexes
		int e[3] = { -1, -1, -1 };  // index of edges
		int f[3] = { -1, -1, -1 };  // index of triangular faces
		// cross(f0-f1,f2-f1) should face outward
		face() {}
		face(int v0, int v1, int v2, int e0 = -1, int e1 = -1, int e2 = -1, int f0 = -1, int f1 = -1, int f2 = -1) { v[0] = v0, v[1] = v1, v[2] = v2, e[0] = e0, e[1] = e1, e[2] = e2, f[0] = f0, f[1] = f1, f[2] = f2; }
		bool existVertex(int id) const { return v[0] == id || v[1] == id || v[2] == id; }
		bool existEdge(int id) const { return e[0] == id || e[1] == id || e[2] == id; }
		bool existFace(int id) const { return f[0] == id || f[1] == id || f[2] == id; }
		int searchVertex(int id) const { return v[0] == id ? 0 : v[1] == id ? 1 : v[2] == id ? 2 : -1; }
		int searchEdge(int id) const { return e[0] == id ? 0 : e[1] == id ? 1 : e[2] == id ? 2 : -1; }
		int searchFace(int id) const { return f[0] == id ? 0 : f[1] == id ? 1 : f[2] == id ? 2 : -1; }
		int oppositeVertex(int a, int b) const { for (int i = 0; i < 3; i++) if (v[i] != a && v[i] != b) return v[i]; return -1; }
		//int oppositeEdge(int a) const { for (int i = 0; i < 3; i++) if (!Es[e[i]].existVertex(a)) return e[i]; return -1; }
		void forward() {
			int t = v[2]; v[2] = v[1], v[1] = v[0], v[0] = t;
			t = e[2]; e[2] = e[1], e[1] = e[0], e[0] = t;
			t = f[2]; f[2] = f[1], f[1] = f[0], f[0] = t;
		}
		void replaceVertex(int v0, int v1) { for (int i = 0; i < 3; i++) if (v[i] == v0) v[i] = v1; }
		void replaceEdge(int e0, int e1) { for (int i = 0; i < 3; i++) if (e[i] == e0) e[i] = e1; }
		void replaceFace(int f0, int f1) { for (int i = 0; i < 3; i++) if (f[i] == f0) f[i] = f1; }
	};
	std::vector<vertex> Vs;
	std::vector<edge> Es;
	std::vector<face> Fs;
	vec3 getVertex(int vid) const { return Vs[vid]; }
	segment getEdge(int eid) const { return segment{ Vs[Es[eid].v[0]], Vs[Es[eid].v[1]] }; }
	triangle getFace(int fid) const { return triangle{ Vs[Fs[fid].v[1]], Vs[Fs[fid].v[0]], Vs[Fs[fid].v[2]] }; }
	vec3 getFaceNormal(int fid) const { return cross(Vs[Fs[fid].v[0]] - Vs[Fs[fid].v[1]], Vs[Fs[fid].v[2]] - Vs[Fs[fid].v[1]]); }
	double getFaceAngle(int eid) const {
		// return the angle between two faces, between 0 and 2π, >π for convex mesh
		edge e = Es[eid]; vec3 p = Vs[e.v[0]], q = Vs[e.v[1]], d = q - p;
		vec3 e1 = Vs[Fs[e.f[0]].oppositeVertex(e.v[0], e.v[1])] - p;
		vec3 e2 = Vs[Fs[e.f[1]].oppositeVertex(e.v[0], e.v[1])] - p;
		vec3 n = normalize(d);
		vec3 u = normalize(e1 - dot(e1, n)*n);
		vec3 v = e2 - dot(e2, n)*n; v = normalize(v - dot(v, u)*u);
		if (dot(n, cross(u, v)) < 0) v = -v;
		vec2 a = vec2(dot(e1, u), dot(e1, v));
		vec2 b = vec2(dot(e2, u), dot(e2, v));
		double t = atan2(det(b, a), dot(b, a));
		if (t < 0) t += 2 * PI;
		//printf("%lf %lf\n", acos(dot(normalize(cross(e1, d)), normalize(cross(d, e2)))) + PI, t);
		return t;
	}
	int getOppositeVertex(int fid, int v1, int v2) const { for (int i = 0; i < 3; i++) if (Fs[fid].v[i] != v1 && Fs[fid].v[i] != v2) return Fs[fid].v[i]; return -1; }
	int getOppositeEdge(int fid, int v) const { for (int i = 0; i < 3; i++) if (!Es[Fs[fid].e[i]].existVertex(v)) return Fs[fid].e[i]; return -1; }
	int getOppositeFace(int eid, int v) const { for (int i = 0; i < 2; i++) if (!Fs[Es[eid].f[i]].existVertex(v)) return Es[eid].f[i]; return -1; }
	void replaceVertex(edge &e, int v0, int v1);
	void exportTriangles(std::vector<triangle> &T) const {  // export STL
		for (int i = 0, l = Fs.size(); i < l; i++) {
			T.push_back(triangle{ Vs[Fs[i].v[1]], Vs[Fs[i].v[0]], Vs[Fs[i].v[2]] });
		}
	}

	// use these functions for initialization
	// call sanity_check() to debug
	// note that addEdge() and addFace() do not check repeated elements
	mesh() {}
	int addVertex(vec3 p, int e = -1) {
		Vs.push_back(vertex(p, e));
		return Vs.size() - 1;
	}
	int addEdge(int v1, int v2, int swap = false) {
		if (swap) std::swap(v1, v2);
		edge E; E.v[0] = v1, E.v[1] = v2;
		Vs[v1].e = Vs[v2].e = Es.size();
		Es.push_back(E);
		return Es.size() - 1;
	}
	int addFace(int v0, int v1, int v2, int e0 = -1, int e1 = -1, int e2 = -1) {
		face F; F.v[0] = v0, F.v[1] = v1, F.v[2] = v2; F.e[0] = e0, F.e[1] = e1, F.e[2] = e2;
		Fs.push_back(F);
		return Fs.size() - 1;
	}
	void setEdgeFaces(int id, int f1, int f2, bool swap = false) {
		if (swap) std::swap(f1, f2);
		Es[id].f[0] = f1, Es[id].f[1] = f2;
	}
	void setFaceEdges(int id, int e1, int e2, int e3) {
		Fs[id].e[0] = e1, Fs[id].e[1] = e2, Fs[id].e[2] = e3;
	}
	void setFaceFaces(int id, int f1, int f2, int f3) {
		Fs[id].f[0] = f1, Fs[id].f[1] = f2, Fs[id].f[2] = f3;
	}
	void sanity_check(bool write_stdout = true) const {
#ifdef _DEBUG
		// write message to stdout/stderr
		if (write_stdout) fprintf(stdout, "Sanity Check [%p]\n", (void*)this);
		int ErrorCount = 0;
#define printError(message, ...) do{ fprintf(stderr, "[%d]: ", __LINE__); fprintf(stderr, message, ##__VA_ARGS__); ErrorCount++; }while(0)
#define checkRepetition(ctr) do{ if (ctr[0]==ctr[1]||ctr[0]==ctr[2]||ctr[1]==ctr[2]) printError("Repeated element {%d,%d,%d}\n", ctr[0], ctr[1], ctr[2]); } while(0)
		// check Euler formula
		int Vn = Vs.size(), En = Es.size(), Fn = Fs.size();
		if (Vn - En + Fn != 2) printError("V-E+F = %d-%d+%d = %d\n", Vn, En, Fn, Vn - En + Fn);
		// check vertexes
		for (int v = 0; v < Vn; v++) {
			if (Vs[v].e < 0 || Vs[v].e >= En) printError("Vs[%d].e==%d\n", v, Vs[v].e);  // invalid index number
			else if (!Es[Vs[v].e].existVertex(v)) {  // indexed segment do not have this vertex
				printError("!Es[Vs[%d].e].existVertex(%d)\n", v, v);
			}
		}
		// check edges
		for (int e = 0; e < En; e++) {
			edge Ei = Es[e];
			int count0 = ErrorCount;
			if (Ei.v[0] >= Ei.v[1]) {  // make sure the vertexes of edges is in increasing order
				printError("E[%d].v=={%d,%d}\n", e, Ei.v[0], Ei.v[1]);
			}
			for (int d = 0; d < 2; d++) {
				if (Ei.v[d] < 0 || Ei.v[d] >= Vn) printError("Es[%d].v[%d]==%d\n", e, d, Ei.v[d]);  // invalid index number
				if (Ei.f[d] < 0 || Ei.f[d] >= Fn) printError("Es[%d].f[%d]==%d\n", e, d, Ei.f[d]);  // invalid index number
				else if (!Fs[Ei.f[d]].existEdge(e)) {  // the edge mentioned a face, but the face do not connect this edge
					printError("!Vs[Es[%d].f[%d]].existEdge(%d)\n", e, d, e);
				}
			}
			if (ErrorCount == count0) {
				// check the order of the faces listed by the edge
				// To fix: make sure the face on the "right" side of the edge appears first
				// and check the direction of the triangle
				face f = Fs[Ei.f[0]]; int d = f.searchVertex(Ei.v[0]);
				if (d == -1 || Ei.v[1] != f.v[(d + 1) % 3]) {
					printError("Incorrect edge-triangle order: E[%d].f[0], {%d,%d}, {%d,%d,%d}\n", e, Ei.v[0], Ei.v[1], f.v[0], f.v[1], f.v[2]);
				}
				f = Fs[Ei.f[1]]; d = f.searchVertex(Ei.v[0]);
				if (Ei.v[1] != f.v[(d + 2) % 3]) {
					printError("Incorrect edge-triangle order: E[%d].f[1], {%d,%d}, {%d,%d,%d}\n", e, Ei.v[0], Ei.v[1], f.v[0], f.v[1], f.v[2]);
				}
			}
		}
		// check faces
		for (int f = 0; f < Fn; f++) {
			face Fi = Fs[f];
			for (int d = 0; d < 3; d++) {
				if (Fi.v[d] < 0 || Fi.v[d] >= Vn) printError("Fs[%d].v[%d]==%d\n", f, d, Fi.v[d]);  // invalid index number
				else checkRepetition(Fi.v);
				if (Fi.e[d] < 0 || Fi.e[d] >= En) printError("Fs[%d].e[%d]==%d\n", f, d, Fi.e[d]);  // invalid index number
				else if (!Es[Fi.e[d]].existFace(f)) {  // mentioned edge do not connect to face
					printError("!Es[Fs[%d].e[%d]].existFace(%d)\n", f, d, f);
				}
				else if (!Es[Fi.e[d]].existVertex(Fi.v[d])) {  // edges do not correstpond to vertex; check the order of parameters in setFaceEdges() and addFace()
					printError("!Es[Fs[%d].e[%d]].existVertex(Fs[%d].v[%d])\n", f, d, f, d);
				}
				else if (!Es[Fi.e[d]].existVertex(Fi.v[(d + 2) % 3])) {  // vertex/edge shifting (reference the ascii diagram above)
					printError("!Es[Fs[%d+1].e[%d]].existVertex(Fs[%d].v[%d])\n", f, d, f, d);
				}
				else checkRepetition(Fi.e);
				if (Fi.f[d] < 0 || Fi.f[d] >= Fn || Fi.f[d] == f) printError("Fs[%d].f[%d]==%d\n", f, d, Fi.f[d]);  // invalid index number
				else if (!Fs[Fi.f[d]].existFace(f)) {  // mentioned face do not connect to face
					printError("!Fs[Fs[%d].f[%d]].existFace(%d)\n", f, d, f);
				}
				else if (!Fs[Fi.f[d]].existVertex(Fi.v[d])) {  // faces do not correstpond to vertex; order error
					printError("!Fs[Fs[%d].f[%d]].existVertex(Fs[%d].v[%d])\n", f, d, f, d);
				}
				else checkRepetition(Fi.f);
			}
		}
		// some insane meshes may still pass despite the length of this checker
		if (ErrorCount) fprintf(stdout, "%d errors found.\n", ErrorCount);
		if (write_stdout) fprintf(stdout, "Complete. [%p]\n", (void*)this);
#undef printError
#endif
	}

	// mesh manipulation; call sanity_check() after
	void splitEdge(int id, vec3 p) {  // why so loooong......
		edge E = Es[id];
		int v1 = E.v[0];
		int v3 = E.v[1];
		int v2 = Fs[E.f[0]].oppositeVertex(v1, v3);
		int v4 = Fs[E.f[1]].oppositeVertex(v1, v3);
		int v = addVertex(p);
		int e01 = getOppositeEdge(E.f[0], E.v[1]);
		int e02 = getOppositeEdge(E.f[0], E.v[0]);
		int e03 = getOppositeEdge(E.f[1], E.v[0]);
		int e04 = getOppositeEdge(E.f[1], E.v[1]);
		int e1 = id; Es[e1] = edge(v1, v);
		int e2 = addEdge(v2, v);
		int e3 = addEdge(v3, v);
		int e4 = addEdge(v4, v);
		int f1 = E.f[0]; Fs[f1] = face(v1, v, v2, e01, e1, e2);
		int f2 = E.f[1]; Fs[f2] = face(v2, v, v3, e02, e2, e3);
		int f3 = addFace(v3, v, v4, e03, e3, e4);
		int f4 = addFace(v4, v, v1, e04, e4, e1);
		Es[e01].replaceFace(E.f[0], f1); Es[e01].autoSwap();
		Es[e02].replaceFace(E.f[0], f2); Es[e02].autoSwap();
		Es[e03].replaceFace(E.f[1], f3); Es[e03].autoSwap();
		Es[e04].replaceFace(E.f[1], f4); Es[e04].autoSwap();
		Es[e1].f[0] = f1, Es[e1].f[1] = f4;
		Es[e2].f[0] = f2, Es[e2].f[1] = f1;
		Es[e3].f[0] = f3, Es[e3].f[1] = f2;
		Es[e4].f[0] = f4, Es[e4].f[1] = f3;
		Fs[f1].f[0] = getOppositeFace(e01, v), Fs[f1].f[1] = f4, Fs[f1].f[2] = f2;
		Fs[f2].f[0] = getOppositeFace(e02, v), Fs[f2].f[1] = f1, Fs[f2].f[2] = f3;
		Fs[f3].f[0] = getOppositeFace(e03, v), Fs[f3].f[1] = f2, Fs[f3].f[2] = f4;
		Fs[f4].f[0] = getOppositeFace(e04, v), Fs[f4].f[1] = f3, Fs[f4].f[2] = f1;
		Fs[Fs[f1].f[0]].replaceFace(E.f[0], f1);
		Fs[Fs[f2].f[0]].replaceFace(E.f[0], f2);
		Fs[Fs[f3].f[0]].replaceFace(E.f[1], f3);
		Fs[Fs[f4].f[0]].replaceFace(E.f[1], f4);
		this->sanity_check();
	}
	void flipEdge(int id) {
		edge E = Es[id]; int F0 = E.f[0], F1 = E.f[1];
		//while (Fs[E.f[0]].v[0] != E.v[0]) Fs[E.f[0]].forward();
		//while (Fs[E.f[1]].v[0] != E.v[1]) Fs[E.f[1]].forward();
		int v0 = E.v[0];
		int v2 = E.v[1];
		int v1 = getOppositeVertex(E.f[0], v0, v2);
		int v3 = getOppositeVertex(E.f[1], v0, v2);
		int e0 = getOppositeEdge(E.f[0], v2);
		int e1 = getOppositeEdge(E.f[0], v0);
		int e2 = getOppositeEdge(E.f[1], v0);
		int e3 = getOppositeEdge(E.f[1], v2);
		int f0 = Es[e0].anotherFace(E.f[0]);
		int f1 = Es[e1].anotherFace(E.f[0]);
		int f2 = Es[e2].anotherFace(E.f[1]);
		int f3 = Es[e3].anotherFace(E.f[1]);
		if (f1 == f2 || f0 == f3) return;  // happens when the mesh intersects itself
		Es[id] = edge(v1, v3, F0, F1);
		Fs[F0] = face(v2, v1, v3, e2, e1, id, f2, f1, F1);
		Fs[F1] = face(v3, v1, v0, e3, id, e0, f3, F0, f0);
		Es[e0].replaceFace(F0, F1);
		Es[e2].replaceFace(F1, F0);
		Fs[f0].replaceFace(F0, F1);
		Fs[f2].replaceFace(F1, F0);
		Es[id].autoSwap();
		Vs[Es[id].v[0]].e = Vs[Es[id].v[1]].e = id;
		Vs[v0].e = e0, Vs[v2].e = e2;
		//this->sanity_check();
	}

	// loop subdivision
	void listNeighborVertices(int vid, std::vector<int> &cvs) {
		// list index of vertexes directly connected to the given vertex
		int ei0 = Vs[vid].e, ei = ei0;
		edge e;
		do {
			e = Es[ei];
			if (e.v[0] != vid) e.swap();
			cvs.push_back(e.v[1]);
			ei = getOppositeEdge(e.f[1], e.v[1]);
		} while (ei != ei0);
	}
	void LoopSubdivide() {
		int Vn = Vs.size(), En = Es.size(), Fn = Fs.size();
		// calculate the new positions of even vertices
		std::vector<vec3> Vs_new; Vs_new.reserve(Vn);
		std::vector<int> cvs;
		for (int i = 0; i < Vn; i++) {
			cvs.clear();
			listNeighborVertices(i, cvs);
			int N = cvs.size();
			double beta = 0.625 - pow(0.375 + 0.25*cos(2 * PI / N), 2.);
			//double beta = N == 3 ? 0.5625 : 0.375;
			vec3 S(0.0); for (int i = 0; i < N; i++) S += Vs[cvs[i]];
			Vs_new.push_back(vertex(Vs[i] * (1. - beta) + (S / N) * beta));
		}
		// calculate and add the new positions of odd vertices
		for (int i = 0; i < En; i++) {
			edge e = Es[i];
			vec3 a = Vs[e.v[0]], b = Vs[e.v[1]];
			vec3 c = Vs[Fs[e.f[0]].oppositeVertex(e.v[0], e.v[1])];
			vec3 d = Vs[Fs[e.f[1]].oppositeVertex(e.v[0], e.v[1])];
			addVertex(0.375*(a + b) + 0.125*(c + d));
		}
		for (int i = 0; i < Vn; i++) Vs[i].setP(Vs_new[i]);
		// create arrays of new edges and faces
		std::vector<edge> Es_new; Es_new.reserve(2 * En + 3 * Fn);  // splitted-edge pair(2), splitted-face pair(3)
		for (int i = 0; i < En; i++) {
			Es_new.push_back(edge(Es[i].v[0], Vn + i));
			Es_new.push_back(edge(Es[i].v[1], Vn + i));
		}
		for (int i = 0; i < Fn; i++) {
			Es_new.push_back(edge(Fs[i].e[2] + Vn, Fs[i].e[0] + Vn, 4 * i + 3, 4 * i));
			Es_new.push_back(edge(Fs[i].e[0] + Vn, Fs[i].e[1] + Vn, 4 * i + 3, 4 * i + 1));
			Es_new.push_back(edge(Fs[i].e[1] + Vn, Fs[i].e[2] + Vn, 4 * i + 3, 4 * i + 2));
		}
		std::vector<face> Fs_new; Fs_new.reserve(4 * Fn);  // splitted-face pair(4: 3 border + 1 center)
		for (int i = 0; i < Fn; i++) {
			face f = Fs[i];
			Fs_new.push_back(face(f.e[2] + Vn, f.v[2], f.e[0] + Vn, 2 * En + 3 * i, -1, -1, 4 * i + 3, -1, -1));
			Fs_new.push_back(face(f.e[0] + Vn, f.v[0], f.e[1] + Vn, 2 * En + 3 * i + 1, -1, -1, 4 * i + 3, -1, -1));
			Fs_new.push_back(face(f.e[1] + Vn, f.v[1], f.e[2] + Vn, 2 * En + 3 * i + 2, -1, -1, 4 * i + 3, -1, -1));
			Fs_new.push_back(face(f.e[0] + Vn, f.e[1] + Vn, f.e[2] + Vn, 2 * En + 3 * i, 2 * En + 3 * i + 1, 2 * En + 3 * i + 2, 4 * i, 4 * i + 1, 4 * i + 2));
		}
		// dirty connectivity code
		for (int i = 0; i < En; i++) {
			edge e = Es[i]; int v0 = e.v[0], v1 = e.v[1], f0 = e.f[0], f1 = e.f[1];
			int fi = 4 * f0 + (Fs[f0].searchVertex(v0) + 1) % 3;
			int fj = 4 * f1 + (Fs[f1].searchVertex(v0) + 1) % 3;
			int vi = Fs_new[fi].searchVertex(v0);
			//if (vi == -1 || Fs_new[fi].e[(vi + 1) % 3] != -1) throw(__LINE__);
			Fs_new[fi].e[(vi + 1) % 3] = 2 * i, Fs_new[fi].f[(vi + 1) % 3] = fj;
			vi = Fs_new[fj].searchVertex(v0);
			//if (vi == -1 || Fs_new[fj].e[vi] != -1) throw(__LINE__);
			Fs_new[fj].e[vi] = 2 * i, Fs_new[fj].f[vi] = fi;
			Es_new[2 * i] = edge(v0, Vn + i, fi, fj);
			fi = 4 * f1 + (Fs[f1].searchVertex(v1) + 1) % 3;
			fj = 4 * f0 + (Fs[f0].searchVertex(v1) + 1) % 3;
			vi = Fs_new[fi].searchVertex(v1);
			//if (vi == -1 || Fs_new[fi].e[(vi + 1) % 3] != -1) throw(__LINE__);
			Fs_new[fi].e[(vi + 1) % 3] = 2 * i + 1, Fs_new[fi].f[(vi + 1) % 3] = fj;
			vi = Fs_new[fj].searchVertex(v1);
			//if (vi == -1 || Fs_new[fj].e[vi] != -1) throw(__LINE__);
			Fs_new[fj].e[vi] = 2 * i + 1, Fs_new[fj].f[vi] = fi;
			Es_new[2 * i + 1] = edge(v1, Vn + i, fi, fj);
		}
		// almost done
		Es = Es_new, Fs = Fs_new;
		for (int i = 0; i < 3 * Fn; i++) Es[i + 2 * En].autoSwap();
		for (int i = 0, En = Es.size(); i < En; i++) Vs[Es[i].v[0]].e = Vs[Es[i].v[1]].e = i;
		this->sanity_check();
	}
};


// given four tropical vertices and two polar vertices, construct a "sane" octahedral mesh
mesh constructOctahedron(vec3 S[6]) {
	mesh M;
	for (int i = 0; i < 6; i++) M.addVertex(S[i]);
	for (int i = 0; i < 4; i++) {
		M.addEdge(i, (i + 1) % 4, i == 3);  // 0-3
	}
	for (int i = 0; i < 4; i++) {
		M.addFace(i, 4, (i + 1) % 4);  // 0-3
		M.addEdge(i, 4);  // 4-7
	}
	for (int i = 0; i < 4; i++) {
		M.addFace((i + 1) % 4, 5, i);  // 4-7
		M.addEdge(i, 5);  // 8-11
	}
	for (int i = 0; i < 4; i++) {  // connectivity
		M.setEdgeFaces(i, i + 4, i, i == 3);
		M.setEdgeFaces(i + 4, i, (i + 3) % 4);
		M.setEdgeFaces(i + 8, (i + 3) % 4 + 4, i + 4);
		M.setFaceEdges(i, i, i + 4, (i + 1) % 4 + 4);
		M.setFaceEdges(i + 4, i, (i + 1) % 4 + 8, i + 8);
		M.setFaceFaces(i, i + 4, (i + 3) % 4, (i + 1) % 4);
		M.setFaceFaces(i + 4, i, (i + 1) % 4 + 4, (i + 3) % 4 + 4);
	}
	M.sanity_check(false);
	return M;
}


// generate a grain-shaped polyhedron with N (even) faces
mesh randomPolyhedron(int N = 16) {
	// generate a random planar quadrilateral
	double a[4];
	do {
		for (int i = 0; i < 4; i++) a[i] = randf(0, 2 * PI);
		std::sort(a, a + 4);
	} while (a[1] - a[0] < .7 || a[2] - a[1] < .7 || a[3] - a[2] < .7 || a[0] + 2 * PI - a[3] < .7);
	vec3 S[6];
	for (int i = 0; i < 4; i++) S[i] = vec3(cos(a[i]), sin(a[i]), 0);

	// construct an octahedron
	vec3 n;
	do { n = rand3_c(); } while (n.z < 0.3);
	S[4] = randf(0.5, 1.5)*n;
	S[5] = randf(-1.5, -0.5)*n;
	for (int i = 0; i < 6; i++) S[i] += rand3_u(0.15);  // add noise
	mesh M = constructOctahedron(S);

	// grow the octahedron into a polyhedron
	for (int Q = 8; Q < N; Q += 2) {
		// find the edge that is best to be expanded
		double ma = -INFINITY; segment s; int mi = -1;
		for (int i = 0, l = M.Es.size(); i < l; i++) {
			double t0 = M.getFaceAngle(i), t = 0.5*(t0 - PI);
			double a = M.getFace(M.Es[i].f[0]).area() + M.getFace(M.Es[i].f[1]).area();  // projection area
			//printf("%lf %lf %lf\n", a, degree(t), a*cos(t) / t0);
			a = a * cos(t) / t0;  // small angles have more weight
			if (a > ma) ma = a, s = M.getEdge(i), mi = i;
		}
		// expand edge
		n = M.getFace(M.Es[mi].f[0]).unitnormal() + M.getFace(M.Es[mi].f[1]).unitnormal();
		double t = randf(0.2, 0.8); vec3 p = mix(M.Vs[M.Es[mi].v[0]], M.Vs[M.Es[mi].v[1]], t);
		p += n * (randf(0.2, 0.7)*exp(-0.05*Q)) + rand3_u(0.1);
		M.splitEdge(mi, p);
		M.sanity_check(false);
	}

	// translate polyhedron to the center and apply a random rotation
	double V = 0; vec3 C(0.0);
	for (int i = 0, l = M.Fs.size(); i < l; i++) {
		triangle t = M.getFace(i);
		double dV = 1. / 6. * det(t.a, t.b, t.c);
		C += dV * 0.25*(t.a + t.b + t.c), V += dV;
	}
	C /= V;
	mat3 R = randRotation();
	for (int i = 0, l = M.Vs.size(); i < l; i++) M.Vs[i].setP(R * (M.Vs[i] - C));
	M.sanity_check(false);
	return M;
}

// doesn't work very well
mesh randomDistortion(int D = 4) {
	vec3 S[6] = { vec3(1,0,0), vec3(0,1,0), vec3(-1,0,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1) };
	mesh M = constructOctahedron(S);
	while (D--) M.LoopSubdivide();
	int L = M.Vs.size(); auto P = &M.Vs[0];
	for (int i = 0; i < L; i++) P[i].setP(normalize(P[i]));

	const int TN = 4;
	int T[TN]; for (int i = 0; i < TN; i++) T[i] = i;
	for (int i = 0; i < TN - 2; i++) {
		int j = randi(i, TN);
		std::swap(T[i], T[j]);
	}
	for (int U = 0; U < TN; U++) {
		switch (T[U]) {
		case 0: {
			// elongation
			double s; do { s = exp(randf_n(0.8)) - 1; } while (s > 1.0 || s < -0.7);
			vec3 d = rand3(), sd = s * d;
			for (int i = 0; i < L; i++) P[i] += dot(P[i], sd)*d;
			break;
		}
		case 1: {
			// bending
			double m = randf(0.3, 0.6), f = randf(1.5, 4.0), t = randf(0, 2 * PI);
			mat3 R = randRotation();
			for (int i = 0; i < L; i++) {
				P[i].setP(R*(P[i] + vec3(m * sin(f * P[i].z + t), 0, 0)));
			}
			break;
		}
		case 2: {
			// twisting
			double k = randf(-1.2, 1.2);
			mat3 R = randRotation();
			for (int i = 0; i < L; i++) {
				double a = k * P[i].z, c = cos(a), s = sin(a);
				P[i].setP(R * (mat3(c, -s, 0, s, c, 0, 0, 0, 1)*P[i]));
			}
			break;
		}
		case 3: {
			// displacement
			vec3 m = rand3()*rand3_f(0.2, 0.8), d = rand3_f(0, 2 * PI), f = rand3_f(0.8, 1.6);
			for (int i = 0; i < L; i++) {
				P[i] += m * vec3(sin(f.x*(P[i].x + d.x)), sin(f.y*(P[i].y + d.y)), sin(f.z*(P[i].z + d.z)));
			}
			break;
		}
		}
	}
	// contraction for heavily elongated shapes
	for (int i = 0; i < L; i++) P[i] /= sqrt(P[i].sqr() + 0.5);
	M.sanity_check();
	// flip heavily elongated edges
	for (int i = 0, En = M.Es.size(); i < En; i++) {
		if (0.0*M.getFaceAngle(i) < PI) {
			double l = length(M.Vs[M.Es[i].v[0]] - M.Vs[M.Es[i].v[1]]);
			triangle t1 = M.getFace(M.Es[i].f[0]);
			triangle t2 = M.getFace(M.Es[i].f[1]);
			double tl1[3] = { length(t1.b - t1.a), length(t1.c - t1.a), length(t1.c - t1.b) };
			double tl2[3] = { length(t2.b - t2.a), length(t2.c - t2.a), length(t2.c - t2.b) };
			if (l > .999*max(max(tl1[0], tl1[1]), tl1[2]) && l > .999*max(max(tl2[0], tl2[1]), tl2[2])) {
				if (l > 1.5*min(min(tl1[0], tl1[1]), tl1[2]) && l > 1.5*min(min(tl2[0], tl2[1]), tl2[2])) {
					M.flipEdge(i);
				}
			}
		}
	}
	M.sanity_check();
	return M;
}


// random 16-faced polyhedron
void randomGrain(std::vector<triangle> &T) {
	mesh M = randomPolyhedron(16);
	M.exportTriangles(T);
}
// random 4096-faced polyhedron
void randomPebble(std::vector<triangle> &T) {
	mesh M = randomPolyhedron(16);
	for (int i = 0; i < 4; i++) M.LoopSubdivide();
	M.exportTriangles(T);
}
// random 8192-faced polyhedron
void randomStone(std::vector<triangle> &T) {
	mesh M = randomPolyhedron(128);
	for (int i = 0; i < 3; i++) M.LoopSubdivide();
	M.exportTriangles(T);
}
// random
void randomAbstract(std::vector<triangle> &T) {
	mesh M = randomDistortion();
	M.exportTriangles(T);
}


bool writeSTL(triangle* T, int N, const char* filename) {
	FILE* fp = fopen(filename, "wb");
	if (!fp) return false;

	// stl header
	char s[80]; for (int i = 0; i < 80; i++) s[i] = 0;
	sprintf(s, "%d triangles", N);
	fwrite(s, 1, 80, fp);
	fwrite(&N, 4, 1, fp);

	// triangles
	auto writev = [&](vec3 p) {
		float f = (float)p.x; fwrite(&f, sizeof(float), 1, fp);
		f = (float)p.y; fwrite(&f, sizeof(float), 1, fp);
		f = (float)p.z; fwrite(&f, sizeof(float), 1, fp);
	};
	for (int i = 0; i < N; i++) {
		triangle t = T[i];
		writev(normalize(cross(t.c - t.a, t.b - t.a)));
		writev(t.a); writev(t.b); writev(t.c);
		fputc(0, fp); fputc(0, fp);
	}

	fclose(fp); return true;
}

int main() {
	std::vector<triangle> STL;
#if 0
	//const int NI = 5, NJ = 5, NK = 5;
	const int NI = 1, NJ = 1, NK = 1;
	for (int I = 0; I < NI; I++) {
		for (int J = 0; J < NJ; J++) {
			for (int K = 0; K < NK; K++) {
				_SRAND((I*NI + J)*NJ + K);
				std::vector<triangle> T;
				randomStone(T);
				vec3 tr(2 * I + 1 - NI, 2 * J + 1 - NJ, 2 * K + 1 - NK);
				for (int i = 0, l = T.size(); i < l; i++) STL.push_back(triangle{ T[i].a + tr, T[i].b + tr, T[i].c + tr });
			}
		}
	}
	writeSTL(&STL[0], STL.size(), "D:\\test.stl");
#else
	for (int i = 0; i < 100; i++) {
		_SRAND(i);
		std::vector<triangle> T;
		randomStone(T);
		char filename[64];
		sprintf(filename, "D:\\test%d%d.stl", i / 10, i % 10);
		writeSTL(&T[0], T.size(), filename);
	}
#endif
	return 0;
}

