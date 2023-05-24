#include "numerical/geometry.h"

#include "triangulate/octatree.h"
#include "UI/stl_encoder.h"

// dimension of base grid
#define M 9
// dimension of interpolated grid
#define N 256
// parameters are from 0 to 1
double fun(double x, double y, double z) {
	//return length(vec3(x, y, z) - vec3(0.5)) - 0.5;
	//return pow(x - 0.1, 2.0) + pow(y - 0.5, 2.0) + pow(z - 0.3, 2.0) - 0.3;
	//return length(vec3(x - 0.1, y - 0.5, z - 0.3)) - sqrt(0.3);
	//return x * x / 0.3 + y * y / 0.8 - 1.0 + 0.5*sin(2.0*PI*x) + 0.2*cos(3.0*PI*y) + (z - 0.5);
	//return 2.0 * mod(43758.5453*sin(12.9898*x + 78.233*y + 144.7272*z + 1.0), 1.0) - 1.0;
	//return length(vec3(x, y, z) - vec3(0.5)) < 0.001 ? -1.0 : 1.0;
	return pow(x, 4.0) + pow(y, 4.0) + pow(z, 4.0) - pow(0.5, 4.0);
	//return pow(pow(x, 4.0) + pow(y, 4.0) + pow(z, 4.0), 0.25) - 0.5;
	//return tanh(1000.0*(pow(pow(x, 4.0) + pow(y, 4.0) + pow(z, 4.0), 0.25) - 0.5));
	//return max(z - max(x, y), 1 - (x + y + z));
	//return max(max(x, y), z) - 0.5;
}

// sample values
double G0[M][M][M];

// interpolated values
double G1[N][N][N];

// take samples
void init() {
	for (int k = 0; k < M; k++)
		for (int j = 0; j < M; j++)
			for (int i = 0; i < M; i++)
				G0[k][j][i] = fun((i + 0.5) / M, (j + 0.5) / M, (k + 0.5) / M);
}


// interpolation functions

double ClosestInterpolation(double x, double y, double z) {
	int k1 = clamp(int(z*M), 0, M - 1), j1 = clamp(int(y*M), 0, M - 1), i1 = clamp(int(x*M), 0, M - 1);
	return G0[k1][j1][i1];
}

double TrilinearInterpolation(double x, double y, double z) {
	z = z * M - 0.5, y = y * M - 0.5, x = x * M - 0.5;
	int k0 = clamp(int(z), 0, M - 1), j0 = clamp(int(y), 0, M - 1), i0 = clamp(int(x), 0, M - 1);
	int k1 = min(k0 + 1, M - 1), j1 = min(j0 + 1, M - 1), i1 = min(i0 + 1, M - 1);
	double kf = clamp(z - k0, 0.0, 1.0), jf = clamp(y - j0, 0.0, 1.0), if_ = clamp(x - i0, 0.0, 1.0);
	return mix(
		mix(
			mix(G0[k0][j0][i0], G0[k0][j0][i1], if_),
			mix(G0[k0][j1][i0], G0[k0][j1][i1], if_),
			jf),
		mix(
			mix(G0[k1][j0][i0], G0[k1][j0][i1], if_),
			mix(G0[k1][j1][i0], G0[k1][j1][i1], if_),
			jf),
		kf);
}

double CatmullRomInterpolation(double x, double y, double z) {
	// G1 continuity
	auto interp = [](double s0, double s1, double s2, double s3, double t)->double {
		double t2 = t * t, t3 = t2 * t;
		return s0 * (-0.5*t3 + t2 - 0.5*t)
			+ s1 * (1.5*t3 - 2.5*t2 + 1)
			+ s2 * (-1.5*t3 + 2.*t2 + 0.5*t)
			+ s3 * (0.5*t3 - 0.5*t2);
	};
	z = z * M - 0.5, y = y * M - 0.5, x = x * M - 0.5;
	int k1 = clamp((int)floor(z), -1, M - 1), j1 = clamp((int)floor(y), -1, M - 1), i1 = clamp((int)floor(x), -1, M - 1);
	double kf = clamp(z - k1, 0.0, 1.0), jf = clamp(y - j1, 0.0, 1.0), if_ = clamp(x - i1, 0.0, 1.0);
	int k0 = max(k1 - 1, 0), j0 = max(j1 - 1, 0), i0 = max(i1 - 1, 0);
	int k2 = min(k1 + 1, M - 1), j2 = min(j1 + 1, M - 1), i2 = min(i1 + 1, M - 1);
	int k3 = min(k1 + 2, M - 1), j3 = min(j1 + 2, M - 1), i3 = min(i1 + 2, M - 1);
	k1 = max(k1, 0), j1 = max(j1, 0), i1 = max(i1, 0);
	return interp(
		interp(
			interp(G0[k0][j0][i0], G0[k0][j0][i1], G0[k0][j0][i2], G0[k0][j0][i3], if_),
			interp(G0[k0][j1][i0], G0[k0][j1][i1], G0[k0][j1][i2], G0[k0][j1][i3], if_),
			interp(G0[k0][j2][i0], G0[k0][j2][i1], G0[k0][j2][i2], G0[k0][j2][i3], if_),
			interp(G0[k0][j3][i0], G0[k0][j3][i1], G0[k0][j3][i2], G0[k0][j3][i3], if_),
			jf),
		interp(
			interp(G0[k1][j0][i0], G0[k1][j0][i1], G0[k1][j0][i2], G0[k1][j0][i3], if_),
			interp(G0[k1][j1][i0], G0[k1][j1][i1], G0[k1][j1][i2], G0[k1][j1][i3], if_),
			interp(G0[k1][j2][i0], G0[k1][j2][i1], G0[k1][j2][i2], G0[k1][j2][i3], if_),
			interp(G0[k1][j3][i0], G0[k1][j3][i1], G0[k1][j3][i2], G0[k1][j3][i3], if_),
			jf),
		interp(
			interp(G0[k2][j0][i0], G0[k2][j0][i1], G0[k2][j0][i2], G0[k2][j0][i3], if_),
			interp(G0[k2][j1][i0], G0[k2][j1][i1], G0[k2][j1][i2], G0[k2][j1][i3], if_),
			interp(G0[k2][j2][i0], G0[k2][j2][i1], G0[k2][j2][i2], G0[k2][j2][i3], if_),
			interp(G0[k2][j3][i0], G0[k2][j3][i1], G0[k2][j3][i2], G0[k2][j3][i3], if_),
			jf),
		interp(
			interp(G0[k3][j0][i0], G0[k3][j0][i1], G0[k3][j0][i2], G0[k3][j0][i3], if_),
			interp(G0[k3][j1][i0], G0[k3][j1][i1], G0[k3][j1][i2], G0[k3][j1][i3], if_),
			interp(G0[k3][j2][i0], G0[k3][j2][i1], G0[k3][j2][i2], G0[k3][j2][i3], if_),
			interp(G0[k3][j3][i0], G0[k3][j3][i1], G0[k3][j3][i2], G0[k3][j3][i3], if_),
			jf),
		kf
	);
}

double TriquinticInterpolation(double x, double y, double z) {
	// should be G2 but does not look better than cubic interpolation
	auto interp = [](double s0, double s1, double s2, double s3, double t)->double {
		double t2 = t * t, t3 = t2 * t, t4 = t3 * t, t5 = t4 * t;
		double d0 = 0.5*(s2 - s0), d1 = 0.5*(s3 - s1);
		double l0 = s0 + s2 - 2.0*s1, l1 = s1 + s3 - 2.0*s2;
		return s1 * (-6.0*t5 + 15.0*t4 - 10.0*t3 + 1.0)
			+ s2 * (6.0*t5 - 15.0*t4 + 10.0*t3)
			+ d0 * (-3.0*t5 + 8.0*t4 - 6.0*t3 + t)
			+ d1 * (-3.0*t5 + 7.0*t4 - 4.0*t3)
			+ l0 * (-0.5*t5 + 1.5*t4 - 1.5*t3 + 0.5*t2)
			+ l1 * (0.5*t5 - 1.0*t4 + 0.5*t3);
	};
	z = z * M - 0.5, y = y * M - 0.5, x = x * M - 0.5;
	int k1 = clamp((int)floor(z), -1, M - 1), j1 = clamp((int)floor(y), -1, M - 1), i1 = clamp((int)floor(x), -1, M - 1);
	double kf = clamp(z - k1, 0.0, 1.0), jf = clamp(y - j1, 0.0, 1.0), if_ = clamp(x - i1, 0.0, 1.0);
	int k0 = max(k1 - 1, 0), j0 = max(j1 - 1, 0), i0 = max(i1 - 1, 0);
	int k2 = min(k1 + 1, M - 1), j2 = min(j1 + 1, M - 1), i2 = min(i1 + 1, M - 1);
	int k3 = min(k1 + 2, M - 1), j3 = min(j1 + 2, M - 1), i3 = min(i1 + 2, M - 1);
	k1 = max(k1, 0), j1 = max(j1, 0), i1 = max(i1, 0);
	return interp(
		interp(
			interp(G0[k0][j0][i0], G0[k0][j0][i1], G0[k0][j0][i2], G0[k0][j0][i3], if_),
			interp(G0[k0][j1][i0], G0[k0][j1][i1], G0[k0][j1][i2], G0[k0][j1][i3], if_),
			interp(G0[k0][j2][i0], G0[k0][j2][i1], G0[k0][j2][i2], G0[k0][j2][i3], if_),
			interp(G0[k0][j3][i0], G0[k0][j3][i1], G0[k0][j3][i2], G0[k0][j3][i3], if_),
			jf),
		interp(
			interp(G0[k1][j0][i0], G0[k1][j0][i1], G0[k1][j0][i2], G0[k1][j0][i3], if_),
			interp(G0[k1][j1][i0], G0[k1][j1][i1], G0[k1][j1][i2], G0[k1][j1][i3], if_),
			interp(G0[k1][j2][i0], G0[k1][j2][i1], G0[k1][j2][i2], G0[k1][j2][i3], if_),
			interp(G0[k1][j3][i0], G0[k1][j3][i1], G0[k1][j3][i2], G0[k1][j3][i3], if_),
			jf),
		interp(
			interp(G0[k2][j0][i0], G0[k2][j0][i1], G0[k2][j0][i2], G0[k2][j0][i3], if_),
			interp(G0[k2][j1][i0], G0[k2][j1][i1], G0[k2][j1][i2], G0[k2][j1][i3], if_),
			interp(G0[k2][j2][i0], G0[k2][j2][i1], G0[k2][j2][i2], G0[k2][j2][i3], if_),
			interp(G0[k2][j3][i0], G0[k2][j3][i1], G0[k2][j3][i2], G0[k2][j3][i3], if_),
			jf),
		interp(
			interp(G0[k3][j0][i0], G0[k3][j0][i1], G0[k3][j0][i2], G0[k3][j0][i3], if_),
			interp(G0[k3][j1][i0], G0[k3][j1][i1], G0[k3][j1][i2], G0[k3][j1][i3], if_),
			interp(G0[k3][j2][i0], G0[k3][j2][i1], G0[k3][j2][i2], G0[k3][j2][i3], if_),
			interp(G0[k3][j3][i0], G0[k3][j3][i1], G0[k3][j3][i2], G0[k3][j3][i3], if_),
			jf),
		kf
	);
}


// visualization

int main() {
	init();

	auto fun = [](vec3 p) {
		return CatmullRomInterpolation(p.x, p.y, p.z);
	};

	std::vector<triangle_3d> trigs = ScalarFieldTriangulator_octatree::octatree(fun,
		vec3(0.0), vec3(1.0), ivec3(64), 2);

	writeSTL("D:\\intp.stl", &trigs[0], (int)trigs.size());

	return 0;
}
