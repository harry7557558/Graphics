#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libraries/stb_image_write.h"

#include "numerical/geometry.h"
#include "UI/color_functions/poly.h"

// dimension of base grid
#define M 12
// dimension of interpolated grid
#define N 1024
// parameters are from 0 to 1
double fun(double x, double y) {
	//double val = pow(x - 0.1, 2.0) + pow(y - 0.5, 2.0) - 0.3;
	//double val = pow(x, 6.0) + pow(y, 6.0) - pow(0.5, 6.0);
	double val = x * x / 0.3 + y * y / 0.8 - 1.0 + 0.5*sin(2.0*PI*x) + 0.2*cos(3.0*PI*y);
	return clamp(val, -1.0, 1.0);
}

// sample values
double G0[M][M];
double G1_ref[N][N];  // reference

// interpolated values
double G1[N][N];

// take samples
void init() {
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			G0[j][i] = fun((i + 0.5) / M, (j + 0.5) / M);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			G1_ref[j][i] = fun((i + 0.5) / N, (j + 0.5) / N);
}


// interpolation functions

double ClosestInterpolation(double x, double y) {
	int j1 = int(y*M), i1 = int(x*M);
	return G0[j1][i1];
}

double BilinearInterpolation(double x, double y) {
	y = y * M - 0.5, x = x * M - 0.5;
	int j0 = clamp(int(y), 0, M - 1), i0 = clamp(int(x), 0, M - 1);
	double jf = clamp(y - j0, 0.0, 1.0), if_ = clamp(x - i0, 0.0, 1.0);
	int j1 = clamp(j0 + 1, 0, M - 1), i1 = clamp(i0 + 1, 0, M - 1);
	return mix(
		mix(G0[j0][i0], G0[j0][i1], if_),
		mix(G0[j1][i0], G0[j1][i1], if_),
		jf);
}

double CatmullRomInterpolation(double x, double y) {
	auto interp = [](double s0, double s1, double s2, double s3, double t)->double {
		double t2 = t * t, t3 = t2 * t;
		return s0 * (-0.5*t3 + t2 - 0.5*t)
			+ s1 * (1.5*t3 - 2.5*t2 + 1)
			+ s2 * (-1.5*t3 + 2.*t2 + 0.5*t)
			+ s3 * (0.5*t3 - 0.5*t2);
	};
	y = y * M - 0.5, x = x * M - 0.5;
	int j1 = clamp(int(y), 0, M - 1), i1 = clamp(int(x), 0, M - 1);
	double jf = clamp(y - j1, 0.0, 1.0), if_ = clamp(x - i1, 0.0, 1.0);
	int j0 = clamp(j1 - 1, 0, M - 1), i0 = clamp(i1 - 1, 0, M - 1);
	int j2 = clamp(j1 + 1, 0, M - 1), i2 = clamp(i1 + 1, 0, M - 1);
	int j3 = clamp(j1 + 2, 0, M - 1), i3 = clamp(i1 + 2, 0, M - 1);
	return interp(
		interp(G0[j0][i0], G0[j0][i1], G0[j0][i2], G0[j0][i3], if_),
		interp(G0[j1][i0], G0[j1][i1], G0[j1][i2], G0[j1][i3], if_),
		interp(G0[j2][i0], G0[j2][i1], G0[j2][i2], G0[j2][i3], if_),
		interp(G0[j3][i0], G0[j3][i1], G0[j3][i2], G0[j3][i3], if_),
		jf);
}


// visualization
#define SHOW_ISOLINES 1
#define SHADING_STEP 0.1

vec3 color(double val) {
	val = SHADING_STEP * (floor(val / SHADING_STEP) + 0.5);
	val = clamp(0.5 + 0.5*val, 0.0, 1.0);
	return ColorFunctions<vec3, double>::CMYKColors(val);
}

uint32_t toCOLORREF(vec3 c) {
	uint32_t r = 0;
	uint8_t *k = (uint8_t*)&r;
	k[0] = uint8_t(255 * clamp(c.x, 0, 1));
	k[1] = uint8_t(255 * clamp(c.y, 0, 1));
	k[2] = uint8_t(255 * clamp(c.z, 0, 1));
	k[3] = uint8_t(0xff);
	return r;
}
uint32_t img0[M][M];
uint32_t img1[N][N];

int main() {
	init();

	for (int j = 0; j < M; j++) for (int i = 0; i < M; i++) {
		vec3 col = color(G0[j][i]);
		img0[j][i] = toCOLORREF(col);
	}
	stbi_write_png("D:\\G0.png", M, M, 4, img0, 4 * M);

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			G1[j][i] = CatmullRomInterpolation((i + 0.5) / N, (j + 0.5) / N);
		}
	}

	for (int j = 0; j < N; j++) for (int i = 0; i < N; i++) {
		double val = G1[j][i];
		vec2 grad = (j > 0 && j < N - 1 && i > 0 && i < N - 1) ?
			0.5 * vec2(G1[j][i + 1] - G1[j][i - 1], G1[j + 1][i] - G1[j - 1][i]) : vec2(0.0);
		double exact = G1_ref[j][i];
		vec2 grad_exact = (j > 0 && j < N - 1 && i > 0 && i < N - 1) ?
			0.5 * vec2(G1_ref[j][i + 1] - G1_ref[j][i - 1], G1_ref[j + 1][i] - G1_ref[j - 1][i]) : vec2(0.0);

		vec3 col = color(val);
		if (SHOW_ISOLINES) {
			double dist = abs(exact / length(grad_exact));
			col = mix(vec3(0.0), col, clamp(0.5*dist, 0.0, 1.0));
			double k = PI / SHADING_STEP;
			double sinval = sin(k*exact);
			vec2 singrad = k * grad_exact * cos(k*exact);
			dist = abs(sinval / length(singrad));
			col = mix(vec3(0.5), col, clamp(dist, 0.0, 1.0));
		}

		img1[j][i] = toCOLORREF(col);
	}
	stbi_write_png("D:\\G1.png", N, N, 4, img1, 4 * N);

	return 0;
}
