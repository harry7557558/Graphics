// test delaunay triangulation

#include <stdint.h>
#include "numerical/geometry.h"
#include "numerical/random.h"

#include "triangulate/delaunay_2d.h"

#define STB_IMAGE_IMPLEMENTATION
#include ".libraries/stb_image.h"


typedef uint32_t COLOR;

int W, H;
COLOR img[262144];
vec3f samples[512][512];  // x, y

void load_image_to_samples();
void load_samples_to_image();

vec3f color2vec(COLOR col) {
	uint8_t *p = (uint8_t*)&col;
	return vec3f(p[0], p[1], p[2]);
}
COLOR vec2color(vec3f col) {
	COLOR res; uint8_t *p = (uint8_t*)&res;
	p[0] = (uint8_t)col.x, p[1] = (uint8_t)col.y, p[2] = (uint8_t)col.z;
	return res;
}

vec3f get_pixel(vec2 p) {
	p = p * vec2(511, 511);
	ivec2 pi = ivec2(p);
	vec2 pf = p - vec2(pi);
	if (pi.x < 0) pi.x = 0, pf.x = 0.0;
	if (pi.y < 0) pi.y = 0, pf.y = 0.0;
	if (pi.x >= 511) pi.x = 510, pf.x = 1.0;
	if (pi.y >= 511) pi.y = 511, pf.y = 1.0;
	return samples[pi.x][pi.y];
}


int main() {

	COLOR* img_load = (COLOR*)stbi_load("test_images/peppers.png", &W, &H, nullptr, 4);
	if (W != 512 || H != 512) return 0 * printf("Error\n");
	for (int i = 0; i < W*H; i++) img[i] = img_load[i];
	load_image_to_samples();
	load_samples_to_image();

	std::vector<ivec2> points;
	uint32_t seed = 0;

	int XN = 16, YN = 16;
	int XD = W / XN, YD = H / YN;

	std::vector<ivec2> squares;
	for (int i = 0; i < XN; i++) for (int j = 0; j < YN; j++)
		squares.push_back(ivec2(i*XD, j*YD));

	while (max(XD, YD) > 8) {
		std::vector<ivec2> squares_old = squares;
		squares.clear();
		for (ivec2 s : squares_old) {
#if 0
			// calculate varience
			vec3f avr(0.0);
			for (int u = 0; u < XD; u++) for (int v = 0; v < YD; v++)
				avr += samples[s.x + u][s.y + v] / float(XD * YD);
			vec3f var(0.0);
			for (int u = 0; u < XD; u++) for (int v = 0; v < YD; v++) {
				vec3f dif = samples[s.x + u][s.y + v] - avr;
				var += dif * dif / float(XD*YD);
			}
			// subdivide or not
			if (std::max({ sqrt(var.x), sqrt(var.y), sqrt(var.z) }) < 20) {
#else
			// calculate maximum difference
			vec3f maxcol(0.0), mincol(256.0);
			for (int u = 0; u < XD; u++) for (int v = 0; v < YD; v++) {
				vec3f col = samples[s.x + u][s.y + v];
				maxcol = pMax(maxcol, col), mincol = pMin(mincol, col);
			}
			vec3f colrange = maxcol - mincol;
			// subdivide or not
			if (std::max({ colrange.x, colrange.y, colrange.z }) < 64) {
#endif
				ivec2 dp = ivec2(randi(1, XD), randi(1, YD));
				if (s.x == 0) dp.x = 0; if (s.x + XD == 512) dp.x = XD - 1;
				if (s.y == 0) dp.y = 0; if (s.y + YD == 512) dp.y = YD - 1;
				points.push_back(s + dp);
			}
			else {
				for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++)
					squares.push_back(s + ivec2(u, v)*ivec2(XD, YD) / 2);
			}
		}
		XD /= 2, YD /= 2;
	}

	for (ivec2 s : squares) {
		ivec2 dp = ivec2(randi(1, XD), randi(1, YD));
		if (s.x == 0) dp.x = 0; if (s.x + XD == 512) dp.x = XD - 1;
		if (s.y == 0) dp.y = 0; if (s.y + YD == 512) dp.y = YD - 1;
		points.push_back(s + dp);
	}

	std::vector<ivec3> trigs;
	Delaunay_2d<ivec2>().delaunay(points, trigs);

	FILE* fp = fopen("delaunay-adaptive-peppers.svg", "wb");
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
		fprintf(fp, "<polygon points='%d %d %d %d %d %d' fill='#%x'/>\n",
			points[t.x].x, points[t.x].y, points[t.y].x, points[t.y].y, points[t.z].x, points[t.z].y,
			vec2color(vec3f(col.z, col.y, col.x)));
	}
	fprintf(fp, "</g>\n");
	//fprintf(fp, "<script>for(var trigs=document.getElementsByTagName(\"polygon\"),i=0;i<trigs.length;i++)trigs[i].setAttribute(\"style\",\"fill:white;stroke-width:0.5px;stroke:black;\");</script>\n");  // definitely not work
	fprintf(fp, "</svg>\n");
	fclose(fp);
}




// convert to pixels to samples and apply a Gaussian blur
void load_image_to_samples() {
	const int R = 5;
	/*const float kernel[3][3] = {
		{ 0.077847, 0.123317, 0.077847 },
		{ 0.123317, 0.195346, 0.123317 },
		{ 0.077847, 0.123317, 0.077847 }
	};*/
	const float kernel[5][5] = {
		{ 0.023528, 0.033969, 0.038393, 0.033969, 0.023528 },
		{ 0.033969, 0.049045, 0.055432, 0.049045, 0.033969 },
		{ 0.038393, 0.055432, 0.062651, 0.055432, 0.038393 },
		{ 0.033969, 0.049045, 0.055432, 0.049045, 0.033969 },
		{ 0.023528, 0.033969, 0.038393, 0.033969, 0.023528 }
	};
	for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) {
		vec3f sum(0.0); float sum_weight = 0;
		for (int u = 0; u < R; u++) for (int v = 0; v < R; v++) {
			int x = i + u - R / 2, y = j + v - R / 2;
			if (x >= 0 && x < W && y >= 0 && y < H) {
				sum += color2vec(img[y*W + x]) * kernel[u][v];
				sum_weight += kernel[u][v];
			}
		}
		samples[i][j] = sum / sum_weight;
	}
}

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

// convert samples to pixels to be writed
void load_samples_to_image() {
	for (int i = 0; i < W; i++) for (int j = 0; j < H; j++) {
		img[j*W + i] = vec2color(samples[i][j]) | 0xff000000;
	}
#ifdef STB_IMAGE_WRITE_IMPLEMENTATION
	stbi_write_png("test.png", W, H, 4, img, 4 * W);
#endif
}

