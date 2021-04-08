// test delaunay triangulation

#include <stdint.h>
#include "numerical/geometry.h"
#include "numerical/random.h"

#include "triangulate/delaunay_2d.h"

#define STB_IMAGE_IMPLEMENTATION
#include ".libraries/stb_image.h"

typedef uint32_t COLOR;

vec3f color2vec(COLOR col) {
	uint8_t *p = (uint8_t*)&col;
	return vec3f(p[0], p[1], p[2]);
}

int main() {

	int W, H;
	COLOR *img = (COLOR*)stbi_load("test_images/lenna.png", &W, &H, nullptr, 4);
	if (W != 512 || H != 512) return 0 * printf("Error\n");

	int XN = 32, YN = 32;
	int XD = W / XN, YD = H / YN;

	std::vector<ivec2> points;
	uint32_t seed = 0;
	for (int i = 0; i <= XN; i++) {
		for (int j = 0; j <= YN; j++) {
			ivec2 p = ivec2(i*XD, j*YD);
			int xd = int(XD * (rand01(seed) - 0.5));
			int yd = int(YD * (rand01(seed) - 0.5));
			if (i != 0 && i != XN) p.x += xd;
			if (j != 0 && j != YN) p.y += yd;
			points.push_back(vec2(p));
		}
	}

	std::vector<ivec3> trigs;
	Delaunay_2d<ivec2>().delaunay(points, trigs);

	FILE* fp = fopen("delaunay-lenna.svg", "wb");
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='%d' height='%d'>\n", W, H);
	fprintf(fp, "<g style='stroke-width:%lgpx;stroke:white;'>\n", 0.01*XD);
	for (ivec3 t : trigs) {
		vec3f col = vec3f(0.0);
		int pixel_count = 0;
		int x0 = std::min({ points[t.x].x, points[t.y].x, points[t.z].x });
		int x1 = std::max({ points[t.x].x, points[t.y].x, points[t.z].x });
		int y0 = std::min({ points[t.x].y, points[t.y].y, points[t.z].y });
		int y1 = std::max({ points[t.x].y, points[t.y].y, points[t.z].y });
		for (int i = x0; i < x1; i++) for (int j = y0; j < y1; j++) {
			ivec2 p(i, j),
				pa = points[t.x] - p,
				pb = points[t.y] - p,
				pc = points[t.z] - p;
			if (((det(pa, pb) < 0) + (det(pb, pc) < 0) + (det(pc, pa) < 0)) % 3 == 0) {
				col += color2vec(img[j*W + i]);
				pixel_count += 1;
			}
		}
		col /= (float)pixel_count;
		fprintf(fp, "<polygon points='%d %d %d %d %d %d' fill='rgb(%d,%d,%d)'/>\n",
			points[t.x].x, points[t.x].y, points[t.y].x, points[t.y].y, points[t.z].x, points[t.z].y,
			(int)col.x, (int)col.y, (int)col.z);
	}
	fprintf(fp, "</g></svg>\n");
	fclose(fp);
}
