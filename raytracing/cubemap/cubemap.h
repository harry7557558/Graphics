#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "libraries/stb_image.h"

class Cubemap {
	ivec2 size[6];
	vec3f *pixels[6];
public:
	~Cubemap() {
		for (int i = 0; i < 6; i++) {
			size[i] = vec2(0);
			if (pixels[i]) {
				delete pixels[i];
				pixels[i] = nullptr;
			}
		}
	}
	Cubemap() {
		for (int i = 0; i < 6; i++)
			size[i] = ivec2(0, 0), pixels[i] = nullptr;
	}
	Cubemap(
		const char* up,
		const char* left,
		const char* front,
		const char* right,
		const char* back,
		const char* down
	) {
		for (int i = 0; i < 6; i++)
			size[i] = ivec2(0, 0), pixels[i] = nullptr;
		auto loadImage = [](const char* filename, ivec2 &size, vec3f* *pixels) {
			uint8_t* rgb = stbi_load(filename, &size.x, &size.y, nullptr, 3);
			if (rgb == 0) {
				fprintf(stderr, "Failed loading image: %s\n", filename);
				size = vec2(0), *pixels = nullptr;
				return;
			}
			*pixels = new vec3f[size.x*size.y];
			for (int j = 0; j < size.y; j++) {
				for (int i = 0; i < size.x; i++) {
					uint8_t *col = &rgb[3 * (j*size.x + i)];
					(*pixels)[(size.y - 1 - j)*size.x + i] = vec3f(col[0], col[1], col[2]) / 255.0;
				}
			}
		};
		loadImage(up, size[0], &pixels[0]);
		loadImage(left, size[1], &pixels[1]);
		loadImage(front, size[2], &pixels[2]);
		loadImage(right, size[3], &pixels[3]);
		loadImage(back, size[4], &pixels[4]);
		loadImage(down, size[5], &pixels[5]);
	}
	Cubemap& operator=(const Cubemap &other) {
		for (int i = 0; i < 6; i++) {
			this->size[i] = other.size[i];
			if (other.pixels[i]) {
				int n = this->size[i].x*this->size[i].y;
				this->pixels[i] = new vec3f[n];
				for (int t = 0; t < n; t++) this->pixels[i][t] = other.pixels[i][t];
			}
			else this->pixels[i] = nullptr;
		}
		return *this;
	}
	Cubemap(const Cubemap &other) {
		*this = other;
	}

	vec3f sample_face(int i, vec2f uv);
	vec3f sample_face_bilinear(int i, vec2f uv);
	vec3f sample(vec3f rd);
};


vec3f Cubemap::sample_face(int i, vec2f uv) {
	uv = clamp(vec2f(0.5f) + 0.5f*uv, 0.0f, 1.0f) * vec2f(size[i] - ivec2(1));
	int x = round(uv.x);
	int y = round(uv.y);
	return pixels[i][y*size[i].x + x];
}

vec3f Cubemap::sample_face_bilinear(int i, vec2f uv) {
	// idk why this is faster than sample_face()
	uv = vec2f(0.5) + 0.5f*uv;
	float y = uv.y * size[i].y - 0.5f, x = uv.x * size[i].x - 0.5f;
	int j0 = clamp(int(y), 0, size[i].y - 1), i0 = clamp(int(x), 0, size[i].x - 1);
	float jf = clamp(y - j0, 0.0f, 1.0f), if_ = clamp(x - i0, 0.0f, 1.0f);
	int j1 = clamp(j0 + 1, 0, size[i].y - 1), i1 = clamp(i0 + 1, 0, size[i].x - 1);
	return mix(
		mix(pixels[i][j0*size[i].x + i0], pixels[i][j0*size[i].x + i1], if_),
		mix(pixels[i][j1*size[i].x + i0], pixels[i][j1*size[i].x + i1], if_),
		jf);
}


vec3f Cubemap::sample(vec3f rd) {
	float dist[6] = { 1.0f / rd.z, -1.0f / rd.x, 1.0f / rd.y, 1.0f / rd.x, -1.0f / rd.y, -1.0f / rd.z };
	int min_i = -1; float min_d = INFINITY;
	for (int i = 0; i < 6; i++) {
		if (dist[i] > 0.0 && dist[i] < min_d)
			min_d = dist[i], min_i = i;
	}
	if (min_i == 0) return sample_face_bilinear(0, (rd*min_d).yx());
	if (min_i == 1) return sample_face_bilinear(1, (rd*min_d).yz());
	if (min_i == 2) return sample_face_bilinear(2, (rd*min_d).xz());
	if (min_i == 3) return sample_face_bilinear(3, (rd*min_d).yz() * vec2f(-1.0f, 1.0f));
	if (min_i == 4) return sample_face_bilinear(4, (rd*min_d).xz() * vec2f(-1.0f, 1.0f));
	if (min_i == 5) return sample_face_bilinear(5, (rd*min_d).yx() * vec2f(1.0f, -1.0f));
	return vec3f(max(rd.z, 0.0f));
}
