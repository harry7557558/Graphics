// try to restore connectivity of STL files
// generated by reconstructing the isosurface of a scalar field using marching cube


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

	void fromSTL(std::vector<triangle_3d_f> trigs, float epsilon);
	void toSTL(std::vector<triangle_3d_f> &trigs);
	bool writePLY(const char* filename);

	std::vector<triangle_mesh> split_disjoint() const;
	void loop_subdivide();
	void reduce_edge(float k);

	void smooth_laplacian(float k);
	void smooth_laplacian_weighted(float k, bool constrained);
	void smooth_taubin(float k);
	void smooth_taubin(float k, int N, bool constrained);
	void smooth_taubin_weighted(float k, bool constrained);

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

	// for debugging purpose
	int count_duplicate_vertice() const {
		std::vector<vec3f> v = vertice;
		std::sort(v.begin(), v.end(), [](vec3f a, vec3f b) {
			return a.z == b.z ? a.y == b.y ? a.x < b.x : a.y < b.y : a.z < b.z;
		});
		int dup_count = 0;
		for (int i = 0; i + 1 < (int)v.size(); i++) {
			if (v[i] == v[i + 1]) dup_count++;
		}
		return dup_count;
	}
};






/* ================ CONNECTIVITY RESTORATION ================ */

// data structure for working with mesh connectivity
class disjoint_set {
	uint8_t *rank;
public:
	int *parent;
	const int inf = 0x7fffffff;
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


// restore the continuity of a discrete STL model
void triangle_mesh::fromSTL(std::vector<triangle_3d_f> trigs, float epsilon = 0.) {

	int FN = (int)trigs.size();
	int VN = 0;
	int EN = 0;
	vertice.clear(), edges.clear(), faces.clear();

	// restore vertice
	struct vec3_id {
		vec3f p;
		int id;
	};
	vec3_id *vtx = new vec3_id[3 * FN];
	for (int i = 0; i < FN; i++) {
		for (int u = 0; u < 3; u++)
			vtx[3 * i + u] = vec3_id{ trigs[i][u], 3 * i + u };
	}
	vertice.clear();
	faces.resize(FN);

	if (!(epsilon > 0.)) {

		auto t0 = NTime::now();
		std::sort(vtx, vtx + 3 * FN, [](vec3_id a, vec3_id b) {
			return a.p.x < b.p.x ? true : a.p.x > b.p.x ? false : a.p.y < b.p.y ? true : a.p.y > b.p.y ? false : a.p.z < b.p.z;
		});
		printf("Sort vertice: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		vec3f previous_p = vec3f(NAN);
		for (int i = 0; i < 3 * FN; i++) {
			if (vtx[i].p != previous_p) {
				previous_p = vtx[i].p;
				vertice.push_back(vtx[i].p);
				VN++;
			}
			faces[vtx[i].id / 3].v[vtx[i].id % 3] = VN - 1;
		}
	}

	else {
		disjoint_set dsj(3 * FN);

		// apply a random rotation to avoid worst case runtime
		if (1) {
			const mat3f R(
				0.627040324915f, 0.170877213400f, 0.760014084653f,
				-0.607716612443f, -0.503066180808f, 0.614495676705f,
				0.487340691808f, -0.847186753714f, -0.211597860196f);
			for (int i = 0; i < 3 * FN; i++) vtx[i].p = R * vtx[i].p;
		}

		// three level sorting
		auto t0 = NTime::now();
		std::sort(vtx, vtx + 3 * FN, [](vec3_id a, vec3_id b) { return a.p.z < b.p.z; });
		printf("First level vertice sort: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
		t0 = NTime::now();
		for (int i = 0; i < 3 * FN;) {
			int j = i + 1;
			while (j < 3 * FN && vtx[j].p.z - vtx[j - 1].p.z < epsilon) j++;
			std::sort(vtx + i, vtx + j, [](vec3_id a, vec3_id b) { return a.p.y < b.p.y; });
			for (int u = i; u < j;) {
				int v = u + 1;
				while (v < j && vtx[v].p.y - vtx[v - 1].p.y < epsilon) v++;
				std::sort(vtx + u, vtx + v, [](vec3_id a, vec3_id b) { return a.p.x < b.p.x; });
				for (int m = u; m < v;) {
					int n = m + 1;
					while (n < v && vtx[n].p.x - vtx[n - 1].p.x < epsilon) n++;
					//printf("%d\n", n - m);  // mostly 6
					if (0) {  // O(N)
						for (int t = m; t + 1 < n; t++) dsj.unionSet(vtx[t].id, vtx[t + 1].id);
					}
					else {  // O(N²), more accurate and not much slower
						for (int t1 = m; t1 < n; t1++) for (int t2 = m; t2 < t1; t2++) {
							if ((vtx[t2].p - vtx[t1].p).sqr() < epsilon*epsilon) dsj.unionSet(vtx[t1].id, vtx[t2].id);
							//int i1 = vtx[t1].id, i2 = vtx[t2].id;
							//if ((trigs[i1 / 3][i1 % 3] - trigs[i2 / 3][i2 % 3]).sqr() < epsilon*epsilon) dsj.unionSet(i1, i2);
						}
					}
					m = n;
				}
				u = v;
			}
			i = j;
		}
		printf("Further vertice sort: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		// pull points out from the disjoint set
		int unique_count = 0;
		int *vertice_map = new int[3 * FN];
		for (int i = 0; i < 3 * FN; i++)
			if (dsj.findRepresentative(i) == i) {
				vertice_map[i] = unique_count++;
				vertice.push_back(trigs[i / 3][i % 3]);
			}
		for (int i = 0; i < 3 * FN; i++)
			vertice_map[i] = vertice_map[dsj.findRepresentative(i)];
		for (int i = 0; i < FN; i++) for (int u = 0; u < 3; u++) {
			faces[i].v[u] = vertice_map[3 * i + u];
		}
		delete vertice_map;
	}

	// debug
	//printf("%d\n", this->count_duplicate_vertice());

	delete vtx;

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


// write binary PLY file
bool triangle_mesh::writePLY(const char* filename) {
	FILE* fp = fopen(filename, "wb");
	if (!fp) return false;

	// must be little endian
	uint32_t d = 1;
	if (!(*(char*)&d)) return false;

	// write header
	int VN = (int)vertice.size(), FN = (int)faces.size();
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", VN);
	fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
	fprintf(fp, "element face %d\n", FN);
	fprintf(fp, "property list uchar int vertex_indices\n");
	fprintf(fp, "end_header\n");

	// write vertice
	if (fwrite((float*)&vertice[0], sizeof(float), 3 * VN, fp) != 3 * VN) return false;

	// write faces
	for (int i = 0; i < FN; i++) {
		fputc(3, fp);
		if (fwrite(faces[i].v, sizeof(int), 3, fp) != 3) return false;
	}

	if (fclose(fp)) return false;
	return true;
}






/* ================ SPLIT DISJOINT COMPONENTS ================ */


// split the mesh into disjoint sub-meshes
std::vector<triangle_mesh> triangle_mesh::split_disjoint() const {

	std::vector<triangle_mesh> MS;
	int VN = (int)vertice.size(), EN = (int)edges.size(), FN = (int)faces.size();

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

// visualize disjoint components using colors
void visualize_disjoint_ply(const char* filename, const std::vector<triangle_mesh> &MS) {
	std::vector<stl_triangle> trigs;
	int MS_N = (int)MS.size();

	// calculate colors based on volume
	std::vector<float> Vs;
	float maxV = -INFINITY, minV = INFINITY;
	for (int i = 0; i < MS_N; i++) {
		float V = abs(MS[i].calc_volume());
		maxV = max(maxV, V);
		minV = min(minV, V);
		//Vs.push_back(cbrt(V));
		Vs.push_back(V);
	}
	std::vector<vec3f> cols;
	for (int i = 0; i < MS_N; i++) {
		cols.push_back(ColorFunctions<vec3f, float>::LightTemperatureMap((Vs[i] - minV) / max(maxV - minV, 1e-6f)));
	}

	int VN = 0, FN = 0;
	for (int i = 0; i < MS_N; i++) {
		VN += (int)MS[i].vertice.size();
		FN += (int)MS[i].faces.size();
	}

	// ply header
	FILE* fp = fopen(filename, "wb");
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", VN);
	fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
	fprintf(fp, "property uchar red\nproperty uchar green\nproperty uchar blue\n");
	fprintf(fp, "element face %d\n", FN);
	fprintf(fp, "property list uchar int vertex_indices\n");
	fprintf(fp, "end_header\n");

	// write vertice
	for (int Mi = 0; Mi < MS_N; Mi++) {
		uint8_t col[3];
		for (int u = 0; u < 3; u++) col[u] = (uint8_t)(255 * ((float*)&cols[Mi])[u] + 0.5);
		for (int i = 0; i < (int)MS[Mi].vertice.size(); i++) {
			vec3f p = MS[Mi].vertice[i];
			fwrite(&p, sizeof(vec3f), 1, fp);
			fwrite(col, 1, 3, fp);
		}
	}

	// write faces
	int sum_vn = 0;
	for (int Mi = 0; Mi < MS_N; Mi++) {
		for (int i = 0; i < (int)MS[Mi].faces.size(); i++) {
			const triangle_mesh::face *f = &MS[Mi].faces[i];
			fputc(3, fp);
			for (int u = 0; u < 3; u++) {
				int d = sum_vn + f->v[u];
				fwrite(&d, 4, 1, fp);
			}
		}
		sum_vn += MS[Mi].vertice.size();
	}

	fclose(fp);
}







/* ================ LOOP SUBDIVISION ================ */


// loop subdivision
void triangle_mesh::loop_subdivide() {

	int VN = (int)vertice.size(), EN = (int)edges.size(), FN = (int)faces.size();

	// compute odd vertices
	struct odd_vertex {
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
		if (N == 1)
			OVs[i].pos = 0.5f*(vertice[edges[i].v[0]] + vertice[edges[i].v[1]]);
		else if (N >= 2)
			OVs[i].pos = 0.375f*(vertice[edges[i].v[0]] + vertice[edges[i].v[1]]) + 0.25f*(OVs[i].pos / (float)N);
		else
			OVs[i].pos = vec3f(NAN);  // should be a bug
	}

	// compute even vertices
	struct even_vertex {
		int N = 0;  // number of neighbors
		bool isBoundary = false;
		vec3f pos = vec3f(0.);  // sum of neighbor vertices OR final vertex position
		void addNeighbor(vec3f p, bool isBoundary) {
			if (isBoundary) {
				if (!this->isBoundary) N = 1, pos = p;
				else N++, pos += p;
				this->isBoundary = true;
			}
			else {
				if (!this->isBoundary) N++, pos += p;
			}
		}
	};
	even_vertex *EVs = new even_vertex[VN];
	for (int i = 0; i < (int)edges.size(); i++) {
		edge e = edges[i];
		EVs[e.v[0]].addNeighbor(vertice[e.v[1]], OVs[i].N == 1);
		EVs[e.v[1]].addNeighbor(vertice[e.v[0]], OVs[i].N == 1);
	}
	// calculate vertice positions
	auto beta_f = [](int n) { return (0.625f - powf(0.375f + 0.25f*cosf(2.f*(float)PI / n), 2.f)) / n; };
	const int BETA_N = 64; float beta_table[BETA_N] = { NAN, NAN };  // lookup table
	for (int n = 2; n < BETA_N; n++) beta_table[n] = beta_f(n);
	for (int i = 0; i < VN; i++) {
		int N = EVs[i].N;
		if (EVs[i].isBoundary) {
			EVs[i].pos = 0.75f * vertice[i] + 0.25f * (EVs[i].pos / (float)N);
		}
		else {
			float beta = N < BETA_N ? beta_table[N] : beta_f(N);
			EVs[i].pos = vertice[i] * (1.f - N * beta) + EVs[i].pos * beta;
		}
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








/* ================ MESH OPTIMIZATION ================ */


// reduce edges with length less that k*average(edge_length)
void triangle_mesh::reduce_edge(float k) {
	int VN = (int)vertice.size(), EN = (int)edges.size(), FN = (int)faces.size();

	// calculate the threshold edge length
	float suml = 0.0f;
	for (int i = 0; i < EN; i++)
		suml += length(vertice[edges[i].v[1]] - vertice[edges[i].v[0]]);
	k = k * (suml / EN);

	// merge close points
	disjoint_set dsj(VN);
	for (int i = 0; i < EN; i++) {
		if (length(vertice[edges[i].v[1]] - vertice[edges[i].v[0]]) < k) {
			dsj.unionSet(edges[i].v[0], edges[i].v[1]);
		}
	}
	vec3f *new_pos = new vec3f[VN];
	int *sumcount = new int[VN];
	for (int i = 0; i < VN; i++) new_pos[i] = vec3f(0.), sumcount[i] = 0;
	for (int i = 0; i < VN; i++) {
		int pi = dsj.findRepresentative(i);
		new_pos[pi] += vertice[i], sumcount[pi]++;
	}
	for (int i = 0; i < VN; i++) new_pos[i] /= (float)sumcount[i];
	delete sumcount;

	// update vertice
	int unique_count = 0;
	int *vertice_map = new int[VN];
	std::vector<vec3f> vertice_new;
	for (int i = 0; i < VN; i++)
		if (dsj.findRepresentative(i) == i) {
			vertice_map[i] = unique_count++;
			vertice_new.push_back(new_pos[i]);
		}
	for (int i = 0; i < VN; i++)
		vertice_map[i] = vertice_map[dsj.findRepresentative(i)];
	delete new_pos;
	vertice = vertice_new;
	//printf("%d\n", this->count_duplicate_vertice());
	vertice_new.clear(); vertice_new.shrink_to_fit();  // release memory
	for (int i = 0; i < EN; i++) {
		edges[i].v[0] = vertice_map[edges[i].v[0]];
		edges[i].v[1] = vertice_map[edges[i].v[1]];
	}
	for (int i = 0; i < FN; i++) {
		faces[i].v[0] = vertice_map[faces[i].v[0]];
		faces[i].v[1] = vertice_map[faces[i].v[1]];
		faces[i].v[2] = vertice_map[faces[i].v[2]];
	}
	delete vertice_map;

	// update faces
	std::vector<face> faces_new;
	for (int i = 0; i < FN; i++)
		if (faces[i].v[0] != faces[i].v[1] && faces[i].v[1] != faces[i].v[2] && faces[i].v[2] != faces[i].v[0])
			faces_new.push_back(faces[i]);
	faces = faces_new;
	faces_new.clear(); faces_new.shrink_to_fit();
	FN = (int)faces.size();

	// update edges (connectivity restoration)
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
	edges.clear(); EN = 0;
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

#if 1
	// remove duplicate triangles
	auto hashFace = [](face a) {
		auto hash32 = [](uint32_t x)->uint32_t {
			x = ((x >> 16) ^ x) * 0x45d9f3b;
			x = ((x >> 16) ^ x) * 0x45d9f3b;
			x = (x >> 16) ^ x;
			return x;
		};
		return hash32(a.v[0]) ^ hash32(a.v[1]) ^ hash32(a.v[2]);
	};
	auto isSameFace = [](face a, face b) {
		std::sort(a.v, a.v + 3);
		std::sort(b.v, b.v + 3);
		return a.v[0] == b.v[0] && a.v[1] == b.v[1] && a.v[2] == b.v[2];
	};
	t0 = NTime::now();
	std::sort(faces.begin(), faces.end(), [&](face a, face b) {
		return hashFace(a) < hashFace(b);  // faster
		std::sort(a.v, a.v + 3); std::sort(b.v, b.v + 3);
		return a.v[0] == b.v[0] ? a.v[1] == b.v[1] ? a.v[2] < b.v[2] : a.v[1] < b.v[1] : a.v[0] < b.v[0];
	});
	printf("Sort faces: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
	faces_new.clear(); faces_new.reserve(FN);
	for (int i = 0; i < FN;) {
		int j = i + 1;
		while (j < FN && hashFace(faces[j]) == hashFace(faces[i])) j++;
		for (int u = i; u < j; u++) {
			bool duplicate = false;
			for (int v = i; v < j; v++) if (v != u) {
				if (isSameFace(faces[u], faces[v])) duplicate = true;
			}
			if (!duplicate) faces_new.push_back(faces[u]);
		}
		i = j;
	}
	faces = faces_new;
#endif

}








/* ================ MESH SMOOTHING ================ */

// http://graphics.stanford.edu/courses/cs468-12-spring/LectureSlides/06_smoothing.pdf
// http://www.faculty.jacobs-university.de/llinsen/teaching/320491/Lecture13.pdf


#include <memory>

// Laplacian smoothing
void triangle_mesh::smooth_laplacian(float k) {
	int VN = (int)vertice.size();
	int *N = new int[VN];
	vec3f *Pj = new vec3f[VN];
	std::uninitialized_fill_n(N, VN, 0);
	std::uninitialized_fill_n((float*)&Pj[0], 3 * VN, 0.0f);

	int EN = (int)edges.size();
	for (int i = 0; i < EN; i++) {
		int v0 = edges[i].v[0], v1 = edges[i].v[1];
		Pj[v0] += vertice[v1], N[v0]++;
		Pj[v1] += vertice[v0], N[v1]++;
	}

	for (int i = 0; i < VN; i++) {
		Pj[i] *= 1.0f / (float)N[i];
		vec3f dp = Pj[i] - vertice[i];
		vertice[i] += k * dp;
	}

	delete N;
	delete Pj;
}

// cotangent weighted Laplacian smoothing, doesn't work
// currently identical to ordinary Laplacian smoothing except it provides an constrained option
void triangle_mesh::smooth_laplacian_weighted(float k, bool constrained = false) {
	//throw("DEBUGGING - DOESN'T WORK");

	int EN = (int)edges.size();
	float *W = new float[EN]; int *N = new int[EN];
	for (int i = 0; i < EN; i++) W[i] = 0., N[i] = 0;
	int FN = (int)faces.size();
	for (int i = 0; i < FN; i++) {
		const int *e = faces[i].e, *v = faces[i].v;
		for (int u = 0; u < 3; u++) {
#if 0
			// DOESN'T WORK
			vec3f a = vertice[v[(u + 1) % 3]] - vertice[u];
			vec3f b = vertice[v[(u + 2) % 3]] - vertice[u];
			float cot = dot(a, b) / length(cross(a, b));  // possible to become negative
			if (isnan(cot)) cot = 0.0f;
			W[e[(u + 1) % 3]] += cot;
#endif
			N[e[(u + 1) % 3]] += 1;
		}
	}
	for (int i = 0; i < EN; i++) W[i] *= 1.0f / float(N[i]);

	for (int i = 0; i < EN; i++) W[i] = 1.0;  // DOESN'T WORK

	int VN = (int)vertice.size();
	float *Ws = new float[VN];
	vec3f *Pj = new vec3f[VN];
	for (int i = 0; i < VN; i++) Ws[i] = 0., Pj[i] = vec3f(0.);

	for (int i = 0; i < EN; i++) {
		int v0 = edges[i].v[0], v1 = edges[i].v[1];
		Pj[v0] += W[v1] * vertice[v1], Ws[v0] += W[v1];
		Pj[v1] += W[v0] * vertice[v0], Ws[v1] += W[v0];
	}

	if (constrained) {
		// prevent the shifting of boundary points
		bool *cnst = new bool[VN];
		for (int i = 0; i < VN; i++) cnst[i] = false;
		for (int i = 0; i < EN; i++) if (N[i] & 1) {
			cnst[edges[i].v[0]] = cnst[edges[i].v[1]] = true;
		}
		for (int i = 0; i < VN; i++) {
			if (cnst[i]) continue;
			Pj[i] *= 1.0f / Ws[i];
			vec3f dp = Pj[i] - vertice[i];
			vertice[i] += k * dp;
		}
		delete cnst;
	}
	else {
		for (int i = 0; i < VN; i++) {
			Pj[i] *= 1.0f / Ws[i];
			vec3f dp = Pj[i] - vertice[i];
			vertice[i] += k * dp;
		}
	}

	delete N;
	delete Ws;
	delete Pj;
}

// Taubin smoothing, not recommended for k>0.8
void triangle_mesh::smooth_taubin(float k) {
	smooth_laplacian(k);
	smooth_laplacian(-k);
}
void triangle_mesh::smooth_taubin_weighted(float k, bool constrained = false) {
	smooth_laplacian_weighted(k, constrained);
	smooth_laplacian_weighted(-k, constrained);
}

// smooth N times
void triangle_mesh::smooth_taubin(float k, int N, bool constrained) {
	int VN = (int)vertice.size(), EN = (int)edges.size(), FN = (int)faces.size();

	// calculate k for each vertice
	float *K = new float[VN];
	std::uninitialized_fill_n(K, VN, k);
	if (constrained) {
		// do not change boundary vertice
		bool *face_count = new bool[EN];
		std::uninitialized_fill_n(face_count, EN, false);
		for (int i = 0; i < FN; i++)
			for (int u = 0; u < 3; u++) face_count[faces[i].e[u]] ^= true;
		for (int i = 0; i < EN; i++) if (face_count[i])
			K[edges[i].v[0]] = K[edges[i].v[1]] = 0.;
		delete face_count;
	}

	// calculate the number of neighbors for each vertice
	int *NN = new int[VN];
	std::uninitialized_fill_n(NN, VN, 0);
	for (int i = 0; i < EN; i++)
		NN[edges[i].v[0]]++, NN[edges[i].v[1]]++;

	// smoothing
	vec3f *Pj = new vec3f[VN];
	for (int Ni = 0; Ni < N; Ni++) {
		// calculate the sum of neighbours for each vertex
		std::uninitialized_fill_n((float*)&Pj[0], 3 * VN, 0.0f);
		for (int i = 0; i < EN; i++) {
			int v0 = edges[i].v[0], v1 = edges[i].v[1];
			Pj[v0] += vertice[v1], Pj[v1] += vertice[v0];
		}
		// update vertice position
		for (int i = 0; i < VN; i++) {
			Pj[i] *= 1.0f / (float)NN[i];
			vertice[i] += K[i] * (Pj[i] - vertice[i]);
		}
		// calculate the sum of neighbours for each vertex
		std::uninitialized_fill_n((float*)&Pj[0], 3 * VN, 0.0f);
		for (int i = 0; i < EN; i++) {
			int v0 = edges[i].v[0], v1 = edges[i].v[1];
			Pj[v0] += vertice[v1], Pj[v1] += vertice[v0];
		}
		// update vertice position
		for (int i = 0; i < VN; i++) {
			Pj[i] *= 1.0f / (float)NN[i];
			vertice[i] -= K[i] * (Pj[i] - vertice[i]);
		}
	}

	// clean up
	delete K; delete NN; delete Pj;
}






/* ================ MAIN ================ */

int main(int argc, char* argv[]) {

	auto time_start = NTime::now();

	// load file
	auto t0 = NTime::now();
	std::vector<triangle_3d_f> trigs;
	readSTL(argv[1], trigs);
	printf("Load file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

	// calculate an appropriate epsilon
	float sum_el = 0.;
	for (int i = 0; i < (int)trigs.size(); i++) {
		for (int u = 0; u < 3; u++) sum_el += length(trigs[i][(u + 1) % 3] - trigs[i][u]);
	}
	float epsilon = 1e-3 * sum_el / float(3 * trigs.size());

	// restore mesh connectivity
	t0 = NTime::now();
	triangle_mesh M;
	M.fromSTL(trigs, epsilon);
	printf("Restore connectivity: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

	if (!M.writePLY("D:\\connected.ply")) printf("Error\n");

	// find disconnected components
	if (0) {
		t0 = NTime::now();
		std::vector<triangle_mesh> MS = M.split_disjoint();
		printf("Disjoint set: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		t0 = NTime::now();
		visualize_disjoint_ply("D:\\disconnected.ply", MS);
		printf("Write file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
	}

	// loop subdivision
	if (0) {
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

	// reduce small edges
	if (0) {
		t0 = NTime::now();
		M.reduce_edge(0.5);
		printf("Edge reduction: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		t0 = NTime::now();
		std::vector<triangle_mesh> MS = M.split_disjoint();
		printf("Disjoint set: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		t0 = NTime::now();
		visualize_disjoint_ply("D:\\reduced.ply", MS);
		printf("Write file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
	}

	// mesh smoothing
	if (1) {
		t0 = NTime::now();
#if 0
		for (int i = 0; i < 100; i++) {
			//M.smooth_laplacian_weighted(0.8f, true);
			M.smooth_taubin(0.8f);
			//M.smooth_taubin_weighted(0.8f, true);
			if (i % 10 == 9) {
				for (int i = 0; i < (int)trigs.size(); i++) for (int u = 0; u < 3; u++) trigs[i][u] -= vec3f(0, 2, 0);
				trigs.clear();
				M.toSTL(trigs);
			}
		}
#else
		trigs.clear();
		M.smooth_taubin(0.8f, 100, true);
		M.toSTL(trigs);
#endif
		printf("Mesh smoothing: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());

		t0 = NTime::now();
		writeSTL("D:\\smoothing.stl", &trigs[0], (int)trigs.size(), nullptr, STL_CCW);
		printf("Write file: %.1lfms\n", 1000.*fsec(NTime::now() - t0).count());
	}

	printf("Total time elapsed: %.1lfms\n", 1000.*fsec(NTime::now() - time_start).count());

	return 0;
}


void readSTL(const char* filename, std::vector<triangle_3d_f> &trigs) {
	FILE *fp = fopen(filename, "rb");
	char header[80]; fread(header, 1, 80, fp);
	int N; fread(&N, sizeof(int), 1, fp);
	trigs.reserve(N);
	for (int i = 0; i < N; i++) {
		float f[12];
		fread(f, sizeof(float), 12, fp);
		triangle_3d_f t = triangle_3d_f{ vec3f(f[3], f[4], f[5]), vec3f(f[6], f[7], f[8]), vec3f(f[9], f[10], f[11]) };
		if (t[0] != t[1] && t[0] != t[2] && t[1] != t[2]) trigs.push_back(t);
		uint16_t col; fread(&col, 2, 1, fp);
	}
}

