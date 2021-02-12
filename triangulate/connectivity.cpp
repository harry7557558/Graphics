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
		//int f[2] = { -1, -1 };  // faces, more than 2 in certain situation
	};
	struct face {
		int v[3] = { -1, -1, -1 };  // vertice
		int e[3] = { -1, -1, -1 };  // edges
		//int f[3] = { -1, -1, -1 };  // neighborhood faces
	};

	std::vector<vec3f> vertice;
	std::vector<edge> edges;
	std::vector<face> faces;

	void fromSTL(std::vector<triangle_3d_f> trigs);
	std::vector<triangle_mesh> split_disconnected() const;

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

	// To-do: restore edges.f and faces.f

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

	t0 = NTime::now();
	std::vector<triangle_mesh> MS = M.split_disconnected();
	printf("Disjoint set: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

	t0 = NTime::now();
	visualize_disconnected_stl("D:\\connect.stl", MS);
	printf("Write file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

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

