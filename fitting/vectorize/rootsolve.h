#include <cmath>


#include "glm/glm/glm.hpp"
using glm::clamp; using glm::mix; using glm::sign;
using glm::vec2; using glm::vec3; using glm::vec4;
using glm::dot; using glm::cross; using glm::outerProduct;
using glm::mat2; using glm::mat3; using glm::mat4;
using glm::mat2x3; using glm::mat3x2;
using glm::mat4x3; using glm::mat3x4;
using glm::mat4x2; using glm::mat2x4;
using glm::inverse; using glm::transpose; using glm::determinant;
using glm::ivec2; using glm::ivec3; using glm::ivec4;


#define PIf 3.14159265358979f

int solvePolynomial(int N, const float* C, float* R,
	float x0 = -1e6f, float x1 = 1e6f, float tol = 1e-7f
) {
	const int MAXN = 5;
	if (N <= 0 || N > MAXN)
		return -1;

	// degenerated cases
    while (N > 1) {
		const float F[MAXN+1] = { 1.0f, 1.0f, 2.0f, 6.0f, 24.0f, 120.0f };
        if (C[N]*F[N] < 1.0f/fmax(fmax(fabs(x0),fabs(x1)), 1e6f))
            N--;
        else break;
    }

	// low-degree cases
	if (N == 0)
		return 0;
	if (N == 1) {
		R[0] = -C[0] / C[1];
		return 1;
	}

	// find the roots of its derivative
	float Cd[MAXN];
	for (int i = 1; i <= N; i++)
		Cd[i - 1] = C[i] * i;
	float Rd[MAXN-1];
	int NRd = solvePolynomial(N-1, Cd, Rd, x0, x1, tol);

    tol *= fmax(1.0f, fmin(fabs(x0), fabs(x1)));
    const float NaN = x0-0.1f*(x1-x0);

	// polynomial evaluation
	auto evalPolynomial = [&](const float* C, int N, float x)->float {
		float r = 0;
		for (int i = N; i >= 0; i--)
			r = r * x + C[i];
		return r;
	};

	// bisection search root finding
	auto binarySearch = [&](float x0, float x1)->float {
		float y0 = evalPolynomial(C, N, x0);
		float y1 = evalPolynomial(C, N, x1);
		if ((y0<0)==(y1<0)) return NaN;
		for (int i = 0; i < 24; i++) {
			float x = 0.5f*(x0+x1);
			float y = evalPolynomial(C, N, x);
			if (x1-x0 < tol)
				return x0 + (x1-x0) * (-y0) / (y1-y0);
			if ((y<0.0f)^(y0<0.0f)) y1 = y, x1 = x;
			else y0 = y, x0 = x;
		}
		return 0.5f*(x1-x0);
	};

	// roots must exist in between when sorted
	// according to differential mean value theorem
	int NR = 0;
	float r;
	if (NRd == 0) {
		if ((r = binarySearch(x0, x1)) != NaN)
            R[NR++] = r;
	}
	else {
		if ((r = binarySearch(x0, Rd[0])) != NaN)
            R[NR++] = r;
		for (int i = 1; i < NRd; i++)
			if ((r = binarySearch(Rd[i-1], Rd[i])) != NaN)
                R[NR++] = r;
		if ((r = binarySearch(Rd[NRd-1], x1)) != NaN)
            R[NR++] = r;
	}

	return NR;
}



template<typename vec>
vec4 cubicCurveDistanceSquared(const vec c[4], vec p) {
	vec c0 = c[0] - p,
		c1 = c[1], c2 = c[2], c3 = c[3];
	vec p0 = c0;
	vec p1 = c0 + c1 + c2 + c3;
	vec4 res = dot(p0,p0) < dot(p1,p1) ?
		vec4(0.0f, dot(p0,p0), p+p0) :
		vec4(1.0f, dot(p1,p1), p+p1);
	float k[6];
	k[5] = 3.0f*dot(c3,c3);
	k[4] = 5.0f*dot(c2,c3);
	k[3] = 4.0f*dot(c1,c3) + 2.0f*dot(c2,c2);
	k[2] = 3.0f*(dot(c0,c3) + dot(c1,c2));
	k[1] = 2.0f*dot(c0,c2) + dot(c1,c1);
	k[0] = dot(c0,c1);
	float R[5];
	int NR = solvePolynomial(5, k, R, 0.0f, 1.0f, 1e-3f);
	for (int i = 0; i < NR; i++) {
		float t = R[i];
		vec b = c0 + t * (c1 + t * (c2 + t * c3));
		float d2 = dot(b, b);
		if (d2 < res.y)
			res = vec4(t, d2, p+b);
	}
	return res;
}


#if 0

int main() {

	// https://www.desmos.com/calculator/dl4f9aewce
	// float C[6] = { 0.5, -2.6, 2.9, 0.7, -2.3, 0.8 };
	// float R[6];
	// int N = solvePolynomial(5, C, R, 0.0f, 1.0f, 1e-3f);
	// printf("%d\n", N);
	// for (int i = 0; i < N; i++)
	// 	printf("%f\n", R[i]);

	// https://www.desmos.com/calculator/fbrjyxvhgg
	vec2 c[4] = { vec2(-1.74,-2), vec2(3.74,2.7), vec2(3.79,-2.73), vec2(-5.3,2.85) };
	vec2 p = vec2(-0.49,-0.36);
	vec4 r = cubicCurveDistanceSquared(c, p);
	printf("%f %f (%f,%f)\n", r.x, sqrt(r.y), r.z, r.w);

    return 0;
}

#endif
