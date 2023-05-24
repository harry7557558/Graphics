// visualize image brightness as a 3d plot, created for fun

#define STB_IMAGE_IMPLEMENTATION
#include ".libraries/stb_image.h"

#include "ui/stl_encoder.h"


int main(int argc, char* argv[]) {
	// read image
	typedef unsigned char byte;
	typedef struct { byte r, g, b, a; } rgba;
	auto togray = [](rgba col) {
		return 0.3*col.r + 0.59*col.g + 0.11*col.b;
	};
	int W, H;
	rgba* img = (rgba*)stbi_load(argv[1], &W, &H, nullptr, 4);

	// encode stl
	int TN = 2 * (W - 1) * (H - 1);
	stl_triangle *stl = new stl_triangle[TN];
	const vec3 sc = vec3(vec2(5. / max(W, H)), 1. / 255.);
	int Td = 0;
	for (int i = 0; i < H - 1; i++) {
		for (int j = 0; j < W - 1; j++) {
			rgba c00 = img[i*W + j];
			rgba c01 = img[i*W + (j + 1)];
			rgba c10 = img[(i + 1)*W + j];
			rgba c11 = img[(i + 1)*W + (j + 1)];
			vec3 p00 = sc * vec3(i, j, togray(c00));
			vec3 p01 = sc * vec3(i, j + 1, togray(c01));
			vec3 p10 = sc * vec3(i + 1, j, togray(c10));
			vec3 p11 = sc * vec3(i + 1, j + 1, togray(c11));
			vec3 col = (1. / 255.) * vec3(c00.r, c00.g, c00.b);
			stl[Td++] = stl_triangle(p01, p00, p10, col);
			stl[Td++] = stl_triangle(p10, p11, p01, col);
		}
	}

	// output
	writeSTL(argv[2], stl, TN, nullptr, STL_CCW);
	delete stl;
	free(img);
	return 0;
}
