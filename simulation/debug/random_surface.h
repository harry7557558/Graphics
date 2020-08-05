// generate random closed surface (triangulated)

#pragma GCC optimize "Ofast"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "D:\Coding\Github\Graphics\fitting\numerical\random.h"



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

//                                V3
//    
//                               XXX
//                             XX   X
//                           X       XX
//           F3            XX         XX        F3
//                  E3   XX             XX E2
//                     X                  X
//                   XX      v     v       X
//                XX            ^           XX
//              X                             XX
//           XX                                 XXX
//         X                E1         XXXXXXXXXX XX
//        XX XX X XXXX XXXXXXXXXXXXXXXX               V2
//    V1
//                        F1

// this is an unpleasant coding experience...
// maybe i can do better...


struct mesh {
	struct vertex : public vec3 {
		int e = -1;  // index of any connected edge
		vertex() {}
		vertex(vec3 p) { x = p.x, y = p.y, z = p.z; }
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
		// call autoSwap() after
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
		int oppositeVertex(int a, int b) const { for (int i = 0; i < 3; i++) if (v[i] != a && v[i] != b) return v[i]; }
		//int oppositeEdge(int a) const { for (int i = 0; i < 3; i++) if (!Es[e[i]].existVertex(a)) return e[i]; }
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
	vec3 getFaceNormal(int fid) const {
		return cross(Vs[Fs[fid].v[0]] - Vs[Fs[fid].v[1]], Vs[Fs[fid].v[2]] - Vs[Fs[fid].v[1]]);
	}
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
	int getOppositeVertex(int fid, int v1, int v2) const { for (int i = 0; i < 3; i++) if (Fs[fid].v[i] != v1 && Fs[fid].v[i] != v2) return Fs[fid].v[i]; }
	int getOppositeEdge(int fid, int v) const { for (int i = 0; i < 3; i++) if (!Es[Fs[fid].e[i]].existVertex(v)) return Fs[fid].e[i]; }
	int getOppositeFace(int eid, int v) const { for (int i = 0; i < 2; i++) if (!Fs[Es[eid].f[i]].existVertex(v)) return Es[eid].f[i]; }
	void replaceVertex(edge &e, int v0, int v1);
	void listTriangles(std::vector<triangle> &T) const {  // export STL
		for (int i = 0, l = Fs.size(); i < l; i++) {
			T.push_back(triangle{ Vs[Fs[i].v[1]], Vs[Fs[i].v[0]], Vs[Fs[i].v[2]] });
		}
	}

	// use these functions for initialization
	// call sanity_check() to debug
	// note that addEdge() and addFace() do not check repeated elements
	mesh() {}
	int addVertex(vec3 p) {
		Vs.push_back(p);
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
		// check Euler formula
		int Vn = Vs.size(), En = Es.size(), Fn = Fs.size();
		if (Vn - En + Fn != 2) fprintf(stderr, "[%d]: V-E+F = %d-%d+%d = %d\n", __LINE__, Vn, En, Fn, Vn - En + Fn);
		// check vertexes
		for (int v = 0; v < Vn; v++) {
			if (Vs[v].e < 0 || Vs[v].e >= En) fprintf(stderr, "[%d]: Vs[%d].e==%d\n", __LINE__, v, Vs[v].e);  // invalid index number
			else if (!Es[Vs[v].e].existVertex(v)) {  // indexed segment do not have this vertex
				fprintf(stderr, "[%d]: !Es[Vs[%d].e].existVertex(%d)\n", __LINE__, v, v);
			}
		}
		// check edges
		for (int e = 0; e < En; e++) {
			edge Ei = Es[e];
			int notok = 0;
			if (Ei.v[0] >= Ei.v[1]) {  // make sure the vertexes of edges is in increasing order
				notok += fprintf(stderr, "[%d]: E[%d].v=={%d,%d}\n", __LINE__, e, Ei.v[0], Ei.v[1]);
			}
			for (int d = 0; d < 2; d++) {
				if (Ei.v[d] < 0 || Ei.v[d] >= Vn) notok += fprintf(stderr, "[%d]: Es[%d].v[%d]==%d\n", __LINE__, e, d, Ei.v[d]);  // invalid index number
				if (Ei.f[d] < 0 || Ei.f[d] >= Fn) notok += fprintf(stderr, "[%d]: Es[%d].f[%d]==%d\n", __LINE__, e, d, Ei.f[d]);  // invalid index number
				else if (!Fs[Ei.f[d]].existEdge(e)) {  // the edge mentioned a face, but the face do not connect this edge
					notok += fprintf(stderr, "[%d]: !Vs[Es[%d].f[%d]].existEdge(%d)\n", __LINE__, e, d, e);
				}
			}
			if (!notok) {
				// check the order of the faces listed by the edge
				// To fix: make sure the face on the "right" side of the edge appears first
				face f = Fs[Ei.f[0]]; int d = Ei.v[0] == f.v[0] ? 0 : Ei.v[0] == f.v[1] ? 1 : 2;
				if (Ei.v[1] != f.v[(d + 1) % 3]) fprintf(stderr, "[%d]: Incorrect edge-triangle order: E[%d].f[0], {%d,%d}, {%d,%d,%d}\n", __LINE__, e, Ei.v[0], Ei.v[1], f.v[0], f.v[1], f.v[2]);
				f = Fs[Ei.f[1]]; d = Ei.v[0] == f.v[0] ? 0 : Ei.v[0] == f.v[1] ? 1 : 2;
				if (Ei.v[1] != f.v[(d + 2) % 3]) fprintf(stderr, "[%d]: Incorrect edge-triangle order: E[%d].f[1], {%d,%d}, {%d,%d,%d}\n", __LINE__, e, Ei.v[0], Ei.v[1], f.v[0], f.v[1], f.v[2]);
			}
		}
		// check faces
		for (int f = 0; f < Fn; f++) {
			face Fi = Fs[f];
			for (int d = 0; d < 3; d++) {
				if (Fi.v[d] < 0 || Fi.v[d] >= Vn) fprintf(stderr, "[%d]: Fs[%d].v[%d]==%d\n", __LINE__, f, d, Fi.v[d]);  // invalid index number
				if (Fi.e[d] < 0 || Fi.e[d] >= En) fprintf(stderr, "[%d]: Fs[%d].e[%d]==%d\n", __LINE__, f, d, Fi.e[d]);  // invalid index number
				else if (!Es[Fi.e[d]].existFace(f)) {  // mentioned edge do not connect to face
					fprintf(stderr, "[%d]: !Es[Fs[%d].e[%d]].existFace(%d)\n", __LINE__, f, d, f);
				}
				else if (!Es[Fi.e[d]].existVertex(Fi.v[d])) {  // edges do not correstpond to vertex; check the order of parameters in setFaceEdges() and addFace()
					fprintf(stderr, "[%d]: !Es[Fs[%d].e[%d]].existVertex(Fs[%d].v[%d])\n", __LINE__, f, d, f, d);
				}
				if (Fi.f[d] < 0 || Fi.f[d] >= Fn) fprintf(stderr, "[%d]: Fs[%d].f[%d]==%d\n", __LINE__, f, d, Fi.f[d]);  // invalid index number
				else if (!Fs[Fi.f[d]].existFace(f)) {  // mentioned face do not connect to face
					fprintf(stderr, "[%d]: !Fs[Fs[%d].f[%d]].existFace(%d)\n", __LINE__, f, d, f);
				}
				else if (!Fs[Fi.f[d]].existVertex(Fi.v[d])) {  // faces do not correstpond to vertex; order error
					fprintf(stderr, "[%d]: !Fs[Fs[%d].f[%d]].existVertex(Fs[%d].v[%d])\n", __LINE__, f, d, f, d);
				}
			}
		}
		// some insane meshes may still pass despite the length of this checker
		if (write_stdout) fprintf(stdout, "Complete. [%p]\n", (void*)this);
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
	}
};


void randomPolyhedron(std::vector<triangle> &T) {
	mesh M;

	// generate random planar quadrilateral
	double a[4];
	do {
		for (int i = 0; i < 4; i++) a[i] = randf(0, 2 * PI);
		std::sort(a, a + 4);
	} while (a[1] - a[0] < .7 || a[2] - a[1] < .7 || a[3] - a[2] < .7 || a[0] + 2 * PI - a[3] < .7);
	for (int i = 0; i < 4; i++) {
		M.addVertex(vec3(cos(a[i]), sin(a[i]), 0));  // 0-3
	}

	// construct an octahedron
	vec3 n;
	do { n = rand3_c(); } while (n.z < 0.3);
	M.addVertex(randf(0.5, 1.5)*n);  // 4
	M.addVertex(randf(-1.5, -0.5)*n);  // 5
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
	// add noise
	for (int i = 0; i < 6; i++) M.Vs[i] += rand3_u(0.15);
	M.sanity_check(false);

	// grow the octahedron into a 16-faced polyhedron
	for (int Q = 0; Q < 4; Q++) {
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
		p += n * (randf(0.15, 0.5)*(1.0 - 0.1*Q)) + rand3_u(0.1);
		M.splitEdge(mi, p);
		M.sanity_check(false);
	}

	// translate to the center and apply a random rotation
	double V = 0; vec3 C(0.0);
	for (int i = 0, l = M.Fs.size(); i < l; i++) {
		triangle t = M.getFace(i);
		double dV = 1. / 6. * det(t.a, t.b, t.c);
		C += dV * 0.25*(t.a + t.b + t.c), V += dV;
	}
	C /= V;
	mat3 R = axis_angle(rand3(), randf(0, 2 * PI));
	for (int i = 0, l = M.Vs.size(); i < l; i++) M.Vs[i] = R * (M.Vs[i] - C);

	// export triangular faces
	M.listTriangles(T);
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
	const int NI = 5, NJ = 5, NK = 5;
	for (int I = 0; I < NI; I++) {
		for (int J = 0; J < NJ; J++) {
			for (int K = 0; K < NK; K++) {
				_SRAND((I*NI + J)*NJ + K);
				std::vector<triangle> T;
				randomPolyhedron(T);
				vec3 tr(2 * I + 1 - NI, 2 * J + 1 - NJ, 2 * K + 1 - NK);
				for (int i = 0, l = T.size(); i < l; i++) STL.push_back(triangle{ T[i].a + tr, T[i].b + tr, T[i].c + tr });
			}
		}
	}
	writeSTL(&STL[0], STL.size(), "D:\\test.stl");
	return 0;
}

