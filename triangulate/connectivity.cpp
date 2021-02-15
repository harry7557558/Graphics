// try to restore connectivity of STL files


#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"

void readSTL(const char* filename, std::vector<triangle_3d_f> &trigs);


#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;




struct triangle_mesh {
	struct edge {
		int v[2] = { -1, -1 };  // vertices
	};
	struct face {
		int v[3] = { -1, -1, -1 };  // vertice
		int e[3] = { -1, -1, -1 };  // edges
	};

	std::vector<vec3f> vertice;
	std::vector<edge> edges;
	std::vector<face> faces;

	void fromSTL(std::vector<triangle_3d_f> trigs);
	void toSTL(std::vector<triangle_3d_f> &trigs);

	std::vector<triangle_mesh> split_disconnected() const;
	void loop_subdivide();

	// based on divergence theorem, assume ccw outward normal
	float calc_volume() const {
		float V = 0.;
		for (int i = 0; i < (int)faces.size(); i++) {
			const int *v = faces[i].v;
			float dV = det(vertice[v[0]], vertice[v[1]], vertice[v[2]]) * 0.166666667f;
			V += dV;
		}
		return V;
	}

};


// restore the continuity of a discrete STL model
void triangle_mesh::fromSTL(std::vector<triangle_3d_f> trigs) {

	int FN = (int)trigs.size();
	int VN = 0;
	int EN = 0;
	vertice.clear(), edges.clear(), faces.clear();

	// restore vertice
	{
		struct vec3_id {
			vec3f p;
			int id;
		};
		vec3_id *vtx = new vec3_id[3 * FN];
		for (int i = 0; i < FN; i++) {
			for (int u = 0; u < 3; u++)
				vtx[3 * i + u] = vec3_id{ trigs[i][u], 3 * i + u };
		}

		auto t0 = NTime::now();
		std::sort(vtx, vtx + 3 * FN, [](vec3_id a, vec3_id b) {
			return a.p.x < b.p.x ? true : a.p.x > b.p.x ? false : a.p.y < b.p.y ? true : a.p.y > b.p.y ? false : a.p.z < b.p.z;
		});
		printf("Sort vertice: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());


		vertice.clear();
		faces.resize(FN);
		vec3f previous_p = vec3f(NAN);
		for (int i = 0; i < 3 * FN; i++) {
			if (vtx[i].p != previous_p) {
				previous_p = vtx[i].p;
				vertice.push_back(vtx[i].p);
				VN++;
			}
			faces[vtx[i].id / 3].v[vtx[i].id % 3] = VN - 1;
		}
		delete vtx;
	}

	// restore edges
	{
		struct edge_id {
			int v0, v1;
			int id;
		};
		edge_id *egs = new edge_id[3 * FN];
		for (int i = 0; i < FN; i++) {
			for (int u = 0; u < 3; u++) {
				int v0 = min(faces[i].v[u], faces[i].v[(u + 1) % 3]);
				int v1 = max(faces[i].v[u], faces[i].v[(u + 1) % 3]);
				egs[3 * i + u] = edge_id{ v0, v1, 3 * i + u };
			}
		}

		auto t0 = NTime::now();
		std::sort(egs, egs + 3 * FN, [](edge_id a, edge_id b) {
			return a.v0 < b.v0 ? true : a.v0 > b.v0 ? false : a.v1 < b.v1;
		});
		printf("Sort edges: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		edges.clear();
		int prev_v0 = -1, prev_v1 = -1;
		for (int i = 0; i < 3 * FN; i++) {
			if (!(prev_v0 == egs[i].v0 && prev_v1 == egs[i].v1)) {
				prev_v0 = egs[i].v0, prev_v1 = egs[i].v1;
				edges.push_back(edge{ prev_v0, prev_v1 });
				EN++;
			}
			faces[egs[i].id / 3].e[egs[i].id % 3] = EN - 1;
		}
		delete egs;
	}

}


// connected mesh to discrete STL model
void triangle_mesh::toSTL(std::vector<triangle_3d_f> &trigs) {
	int FN = (int)faces.size();
	trigs.reserve(trigs.size() + FN);
	for (int i = 0; i < FN; i++) {
		face f = faces[i];
		trigs.push_back(triangle_3d_f(vertice[f.v[0]], vertice[f.v[1]], vertice[f.v[2]]));
	}
}





// split the mesh into disconnected sub-meshes
std::vector<triangle_mesh> triangle_mesh::split_disconnected() const {

	std::vector<triangle_mesh> MS;
	int VN = (int)vertice.size(), EN = (int)edges.size(), FN = (int)faces.size();

	// data structure for working with mesh connectivity
	class disjoint_set {
		uint8_t *rank;
	public:
		int *parent;
		const int inf = 0x3fffffff;
		disjoint_set(int N) {
			parent = new int[N];
			rank = new uint8_t[N];
			for (int i = 0; i < N; i++) {
				parent[i] = -inf;
				rank[i] = 0;
			}
		}
		~disjoint_set() {
			delete parent; parent = 0;
			delete rank; rank = 0;
		}
		int findRepresentative(int i) {
			if (parent[i] < 0) return i;
			else {
				int ans = findRepresentative(parent[i]);
				parent[i] = ans;
				return ans;
			}
		}
		int representative_ID(int i) {
			while (parent[i] >= 0) i = parent[i];
			return -1 - parent[i];
		}
		bool unionSet(int i, int j) {
			int i_rep = findRepresentative(i);
			int j_rep = findRepresentative(j);
			if (i_rep == j_rep) return false;
			if (rank[i_rep] < rank[j_rep])
				parent[i_rep] = parent[i] = j_rep;
			else if (rank[i_rep] > rank[j_rep])
				parent[j_rep] = parent[j] = i_rep;
			else parent[j_rep] = parent[j] = i_rep, rank[i_rep]++;
			return true;
		}
	};

	// disjoint set
	disjoint_set djs(VN);
	for (int i = 0; i < EN; i++) {
		djs.unionSet(edges[i].v[0], edges[i].v[1]);
	}

	// identify vertice sets in the disjoint set
	for (int i = 0; i < VN; i++) {
		djs.findRepresentative(i);
	}
	int MS_N = 0;
	for (int i = 0; i < VN; i++) {
		if (djs.parent[i] == -djs.inf) {
			djs.parent[i] = -(++MS_N);
		}
	}
	for (int i = 0; i < MS_N; i++) {
		MS.push_back(triangle_mesh());
	}

	// map current mesh vertice ID to mesh vertice IDs in the result
	std::vector<ivec2> vertice_map;
	vertice_map.resize(VN);
	for (int i = 0; i < VN; i++) {
		int id = djs.representative_ID(i);
		vertice_map[i] = ivec2(id, (int)MS[id].vertice.size());
		MS[id].vertice.push_back(vertice[i]);
	}

	// map edges
	std::vector<ivec2> edge_map;
	edge_map.resize(EN);
	for (int i = 0; i < EN; i++) {
		ivec2 v0 = vertice_map[edges[i].v[0]], v1 = vertice_map[edges[i].v[1]];
		if (v0.x != v1.x) throw("bug");
		int id = v0.x;
		edge e; e.v[0] = min(v0.y, v1.y), e.v[1] = max(v0.y, v1.y);
		edge_map[i] = ivec2(id, (int)MS[id].edges.size());
		MS[id].edges.push_back(e);
	}

	// add faces
	for (int i = 0; i < FN; i++) {
		ivec2 v0 = vertice_map[faces[i].v[0]], v1 = vertice_map[faces[i].v[1]], v2 = vertice_map[faces[i].v[2]];
		if (v0.x != v1.x || v0.x != v2.x) throw("bug");
		int id = v0.x;
		ivec2 e0 = edge_map[faces[i].e[0]], e1 = edge_map[faces[i].e[1]], e2 = edge_map[faces[i].e[2]];
		if (e0.x != id || e1.x != id || e2.x != id) throw("bug");
		face f;
		f.v[0] = v0.y, f.v[1] = v1.y, f.v[2] = v2.y;
		f.e[0] = e0.y, f.e[1] = e1.y, f.e[2] = e2.y;
		MS[id].faces.push_back(f);
	}

	return MS;
}

// visualize disconnected components using colors
void visualize_disconnected_stl(const char* filename, const std::vector<triangle_mesh> &MS) {
	std::vector<stl_triangle> trigs;
	int MS_N = (int)MS.size();

	// calculate colors based on volume
	std::vector<float> Vs;
	float maxV = -INFINITY, minV = INFINITY;
	for (int i = 0; i < MS_N; i++) {
		float V = MS[i].calc_volume();
		maxV = max(maxV, V);
		minV = min(minV, V);
		Vs.push_back(V);
	}
	std::vector<vec3f> cols;
	for (int i = 0; i < MS_N; i++) {
		cols.push_back(ColorFunctions<vec3f, float>::LightTemperatureMap((Vs[i] - minV) / max(maxV - minV, 1e-6f)));
	}

	// add triangles
	for (int Mi = 0; Mi < MS_N; Mi++) {
		for (int i = 0; i < (int)MS[Mi].faces.size(); i++) {
			const int *f = MS[Mi].faces[i].v;
			trigs.push_back(stl_triangle(MS[Mi].vertice[f[0]], MS[Mi].vertice[f[1]], MS[Mi].vertice[f[2]], cols[Mi]));
		}
	}
	writeSTL(filename, &trigs[0], (int)trigs.size(), nullptr, STL_CCW);

}




// loop subdivision, non-standard version does not work well for non-closed surface
void triangle_mesh::loop_subdivide() {

	int VN = (int)vertice.size(), EN = (int)edges.size(), FN = (int)faces.size();

	// compute even vertices
	struct even_vertex {
		int N = 0;  // number of neighbors
		vec3f pos = vec3f(0.);  // sum of neighbor vertices OR final vertex position
		void addNeighbor(vec3f p) { N++, pos += p; }
	};
	even_vertex *EVs = new even_vertex[VN];
	for (edge e : edges) {
		EVs[e.v[0]].addNeighbor(vertice[e.v[1]]);
		EVs[e.v[1]].addNeighbor(vertice[e.v[0]]);
	}
	// calculate vertice positions
	auto beta_f = [](int n) { return (0.625f - powf(0.375f + 0.25f*cosf(2.f*(float)PI / n), 2.f)) / n; };
	const int BETA_N = 64; float beta_table[BETA_N];  // lookup table
	for (int n = 2; n < BETA_N; n++) beta_table[n] = beta_f(n);
	beta_table[0] = beta_table[1] = NAN;  // should be a bug
	beta_table[2] = 0.125f;  // boundary??
	for (int i = 0; i < VN; i++) {
		int N = EVs[i].N;
		float beta = N < BETA_N ? beta_table[N] : beta_f(N);
		EVs[i].pos = vertice[i] * (1.f - N * beta) + EVs[i].pos * beta;
	}

	// compute odd vertices
	struct odd_vertex {  // same as even_vertex
		int N = 0;  // number of neighbouring faces, should be 1 or 2 but may be larger in certain cases
		vec3f pos = vec3f(0.);  // sum of opposite face vertices OR final vertex position
		void addNeighbor(vec3f p) { N++, pos += p; }
	};
	odd_vertex *OVs = new odd_vertex[EN];
	for (face f : faces) {
		OVs[f.e[0]].addNeighbor(vertice[f.v[2]]);
		OVs[f.e[1]].addNeighbor(vertice[f.v[0]]);
		OVs[f.e[2]].addNeighbor(vertice[f.v[1]]);
	}
	// calculate vertice positions
	for (int i = 0; i < EN; i++) {
		int N = OVs[i].N;
#if 0
		// doesn't really work because it does not correctly detect boundary
		if (N == 1)
			OVs[i].pos = 0.5f*(vertice[edges[i].v[0]] + vertice[edges[i].v[1]]);
		else if (N >= 2)
			OVs[i].pos = 0.375f*(vertice[edges[i].v[0]] + vertice[edges[i].v[1]]) + 0.25f*(OVs[i].pos / N);
		else
			OVs[i].pos = vec3f(NAN);  // should be a bug
#else
		OVs[i].pos = 0.375f*(vertice[edges[i].v[0]] + vertice[edges[i].v[1]]) + 0.25f*(OVs[i].pos / N);
#endif
	}

	// recreate mesh
	for (int i = 0; i < VN; i++) vertice[i] = EVs[i].pos;
	vertice.resize(VN + EN);
	for (int i = 0; i < EN; i++) vertice[VN + i] = OVs[i].pos;
	std::vector<edge> edges_new;
	edges_new.resize(2 * EN + 3 * FN);
	for (int i = 0; i < EN; i++) {
		edge e;
		e.v[0] = edges[i].v[0], e.v[1] = VN + i;
		edges_new[2 * i] = e;
		e.v[0] = edges[i].v[1];
		edges_new[2 * i + 1] = e;
	}
	faces.resize(4 * FN);
	for (int i = 0; i < FN; i++) {
		edge e;
		for (int u = 0; u < 3; u++) {
			int u1 = (u + 1) % 3, u0 = (u + 2) % 3;
			face *f = &faces[(u + 1)*FN + i];
			f->v[0] = faces[i].v[u];
			f->e[0] = 2 * faces[i].e[u];
			if (edges_new[f->e[0]].v[0] != faces[i].v[u] && edges_new[f->e[0]].v[1] != faces[i].v[u]) f->e[0]++;
			f->v[1] = VN + faces[i].e[u];
			f->e[1] = 2 * EN + 3 * i + u0;
			f->v[2] = VN + faces[i].e[u0];
			f->e[2] = 2 * faces[i].e[u0];
			if (edges_new[f->e[2]].v[0] != faces[i].v[u] && edges_new[f->e[2]].v[1] != faces[i].v[u]) f->e[2]++;

		}
		for (int u = 0; u < 3; u++) {
			int u1 = (u + 1) % 3;
			e.v[0] = VN + faces[i].e[u], e.v[1] = VN + faces[i].e[u1];
			edges_new[2 * EN + 3 * i + u] = e;
		}
		for (int u = 0; u < 3; u++) {
			faces[i].v[u] = VN + faces[i].e[u];
			faces[i].e[u] = 2 * EN + 3 * i + u;
		}
	}
	edges = edges_new;
}


int main(int argc, char* argv[]) {
	std::vector<triangle_3d_f> trigs;

	auto time_start = NTime::now();

	auto t0 = NTime::now();
	readSTL(argv[1], trigs);
	printf("Load file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

	t0 = NTime::now();
	triangle_mesh M;
	M.fromSTL(trigs);
	printf("Restore connectivity: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

	// find disconnected components
	if (0) {
		t0 = NTime::now();
		std::vector<triangle_mesh> MS = M.split_disconnected();
		printf("Disjoint set: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		t0 = NTime::now();
		visualize_disconnected_stl("D:\\disconnected.stl", MS);
		printf("Write file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
	}

	// loop subdivision
	else {
		std::vector<triangle_3d_f> trigs;

		t0 = NTime::now();
		for (int i = 0; i < 1; i++) {
			//M.toSTL(trigs);
			M.loop_subdivide();
		}
		printf("Loop subdivision: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		t0 = NTime::now();
		M.toSTL(trigs);
		writeSTL("D:\\subdivide.stl", &trigs[0], (int)trigs.size(), nullptr, STL_CCW);
		printf("Write file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
	}

	printf("Total time elapsed: %.1lfms\n", 1000.*fsec(NTime::now() - time_start).count());

	return 0;
}


void readSTL(const char* filename, std::vector<triangle_3d_f> &trigs) {
	FILE *fp = fopen(filename, "rb");
	char header[80]; fread(header, 1, 80, fp);
	int N; fread(&N, sizeof(int), 1, fp);
	trigs.resize(N);
	for (int i = 0; i < N; i++) {
		float f[12];
		fread(f, sizeof(float), 12, fp);
		trigs[i] = triangle_3d_f{ vec3f(f[3], f[4], f[5]), vec3f(f[6], f[7], f[8]), vec3f(f[9], f[10], f[11]) };
		uint16_t col; fread(&col, 2, 1, fp);
	}
}

