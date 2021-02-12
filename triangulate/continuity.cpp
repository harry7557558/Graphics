// try to restore continuity of STL files


#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

#include "numerical/geometry.h"

void readSTL(const char* filename, std::vector<triangle_3d_f> &trigs);



// https://github.com/Harry7557558/Graphics/blob/master/modeling/generators/random_surface.h#L24-L61
// :(

struct edge {
	int v[2] = { -1, -1 };  // vertices
	int f[2] = { -1, -1 };  // faces
};
struct face {
	int v[3] = { -1, -1, -1 };  // vertice
	int e[3] = { -1, -1, -1 };  // edges
	int f[3] = { -1, -1, -1 };  // neighborhood faces
};

void restoreContinuity(std::vector<triangle_3d_f> trigs,
	std::vector<vec3f> &vertice, std::vector<face> &faces) {

	int FN = (int)trigs.size();
	int VN = 0;
	int EN = 0;

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
		std::sort(vtx, vtx + 3 * FN, [](vec3_id a, vec3_id b) {
			return a.p.x < b.p.x ? true : a.p.x > b.p.x ? false : a.p.y < b.p.y ? true : a.p.y > b.p.y ? false : a.p.z < b.p.z;
		});
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
	printf("%d\n", VN);


	// restore edges

}




int main(int argc, char* argv[]) {
	std::vector<triangle_3d_f> trigs;
	readSTL(argv[1], trigs);

	std::vector<vec3f> vertice;
	std::vector<face> faces;
	restoreContinuity(trigs, vertice, faces);

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

