// try to balance a 3d object placed on a plane
// by minimizing its gravitational potential energy

// seems like downhill simplex method wins


#include <stdio.h>
#include "numerical/geometry.h"
#include "ui/stl_encoder.h"

#include "ui/3D Models/parametric/surfaces.h"
const int OBJ_N = ParamSurfaces.size();





// find the maximum dot(P[i],n)
double maxDot(const vec3* P, int N, vec3 n) {
	double md = -INFINITY;
	for (int i = 0; i < N; i++) {
		double d = dot(P[i], n);
		if (d > md) md = d;
	}
	return md;
}


#include "numerical/random.h"


// the stupid way, 1000 random samples
vec3 balance_random(const vec3* P, int N) {
	uint32_t seed1 = 0, seed2 = 1;
	double maxd = INFINITY; vec3 maxn;
	for (int i = 0; i < 1000; i++) {
		vec3 n = rand3(lcg_next(seed1), lcg_next(seed2));
		double d = maxDot(P, N, n);
		if (d < maxd) maxd = d, maxn = n;
	}
	return maxn;
}


// simulated annealing code copied from "balance_2d_c.cpp"
// randomly tries neighbourbood samples regardless of the situation
// 250-300 samples, sometimes converges to non-stationary point (valley-shaped function)
vec3 balance_SA_naive(const vec3* P, int N) {
	// estimate the "radius" of the object
	double R = calcRadius((triangle*)P, N / 3);

	// function to be minimized
	int sampleCount = 0;
	auto Fun = [&](vec3 n) {
		sampleCount++;
		return maxDot(P, N, n);
	};

	// simulated annealing
	uint32_t seed1 = 0, seed2 = 1, seed3 = 2;  // random number seeds
	double rand;
	vec3 n = vec3(0, 0, 1);  // configulation
	double T = 100.0;  // temperature
	double E = Fun(n);  // energy (gravitational potential)
	vec3 min_n = n; double min_E = E, min_T = T;  // record minimum value encountered
	const int max_iter = 360;  // number of iterations
	const int max_try = 10;  // maximum number of samples per iteration
	double T_decrease = 0.95;  // multiply temperature by this each time
	for (int iter = 0; iter < max_iter; iter++) {
		for (int ty = 0; ty < max_try; ty++) {
			rand = (int32_t(seed1 = seed1 * 1664525u + 1013904223u) + .5) / 2147483648.;  // -1<rand<1
			double da = T * erfinv(rand);  // change of configulation
			vec3 n_new = axis_angle(rand3(lcg_next(seed1), lcg_next(seed1)), da) * n;
			double E_new = Fun(n_new);
			double prob = exp(-(E_new - E) / (T*R));  // probability, note that E is approximately between -2 and 2
			rand = (seed2 = seed2 * 1664525u + 1013904223u) / 4294967296.;  // 0<=rand<1
			if (prob > rand) {  // jump
				n = n_new, E = E_new;
				if (E < min_E) {
					min_n = n, min_E = E, min_T = T;
				}
				break;
			}
		}
		// jump to the minimum point encountered
		double prob = tanh((E - min_E) / R);
		rand = (seed3 = seed3 * 1664525u + 1013904223u) / 4294967296.;
		if (prob > rand) {
			//T = min(min_T, max(T, 0.5 * (E - min_E) / R));
			n = min_n, E = min_E;
		}
		// decrease temperature
		T *= T_decrease;
		if (T < 0.0001) break;
	}
	//if (min_E < E) a = min_a;

	printf("%d samples\n", sampleCount);

	return n;
}


// downhill simplex code copied from "numerical/optimization.h"
// except it works on a sphere instead of a 2d function
// converges to a local minimum almost every time
// 60-80 samples in an average case, seems like this one wins
vec3 balance_downhillSimplex(const vec3* P, int N) {

	const double accur_eps = 1e-4 * calcRadius((triangle*)P, N / 3);
	const int noimporv_break = 10;
	const int max_iter = 1000;

	// function to be minimized
	int sampleCount = 0;
	auto Fun = [&](vec3 n) {
		sampleCount++;
		return maxDot(P, N, n);
	};

#if 1
	// kill some local minimums
	// sometimes double sampleCount? (still wins)
	uint32_t seed1 = 0, seed2 = 1;
	double maxd = INFINITY; vec3 maxn;
	for (int i = 0; i < 10; i++) {
		vec3 n = rand3(lcg_next(seed1), lcg_next(seed2));
		double d = maxDot(P, N, n);
		if (d < maxd) maxd = d, maxn = n;
	}
#endif

	struct sample {
		vec3 n;  // normalized vector
		double val;  // value
	} S[3];
	for (int i = 0; i < 3; i++) {
		double t = 2.*PI*i / 3.;
		S[i].n = axis_angle(cross(maxn, vec3(0, 0, 1)), acos(maxn.z))
			* normalize(vec3(.2*cos(t), .2*sin(t), 1.));
		S[i].val = Fun(S[i].n);
	}

	double old_minval = INFINITY;
	int noimporv_count = 0;

	for (int iter = 0; iter < max_iter; iter++) {

		// sort
		sample temp;
		if (S[0].val > S[1].val) temp = S[0], S[0] = S[1], S[1] = temp;
		if (S[1].val > S[2].val) temp = S[1], S[1] = S[2], S[2] = temp;
		if (S[0].val > S[1].val) temp = S[0], S[0] = S[1], S[1] = temp;

		// termination condition
		if (S[0].val < old_minval - accur_eps) {
			noimporv_count = 0;
			old_minval = S[0].val;
		}
		else if (++noimporv_count > noimporv_break) {
			printf("%d samples\n", sampleCount);
			return S[0].n;
		}

		// reflection
		sample refl;
		vec3 center = (S[0].n + S[1].n) * .5;
		refl.n = normalize(center * 2. - S[2].n);  // expected to be normal vector
		refl.val = Fun(refl.n);
		if (refl.val >= S[0].val && refl.val < S[1].val) {
			S[2] = refl;
			continue;
		}

		// expansion
		if (refl.val < S[0].val) {
			sample expd;
			expd.n = normalize(center + (center - S[2].n)*2.);  // replace by slerp may not work better
			expd.val = Fun(expd.n);
			if (expd.val < refl.val)
				S[2] = expd;
			else
				S[2] = refl;
			continue;
		}

		// contraction
		sample ctrct;
		ctrct.n = normalize(center + .5*(S[2].n - center));
		ctrct.val = Fun(ctrct.n);
		if (ctrct.val < S[2].val) {
			S[2] = ctrct;
			continue;
		}

		// compression
		S[1].n = normalize(S[0].n + (S[1].n - S[0].n)*.5);
		S[2].n = normalize(S[0].n + (S[2].n - S[0].n)*.5);

	}

	printf("%d samples\n", sampleCount);

	return S[0].val < S[1].val && S[0].val < S[2].val ? S[0].n
		: S[1].val < S[2].val ? S[1].n : S[2].n;
}



int main(int argc, char* argv[]) {
	std::vector<triangle> trigs;
	int N = ParamSurfaces[17].param2trigs(trigs);
	translateToCOM_shell(&trigs[0], N);

	//vec3 n = balance_random((vec3*)&trigs[0], 3 * N);
	//vec3 n = balance_SA_naive((vec3*)&trigs[0], 3 * N);
	vec3 n = balance_downhillSimplex((vec3*)&trigs[0], 3 * N);
	double d = maxDot((vec3*)&trigs[0], 3 * N, n);
	mat3 M = axis_angle(cross(vec3(0, 0, 1), n), acos(-n.z));
	for (int i = 0; i < N; i++) {
		trigs[i].applyMatrix(M);
		trigs[i].translate(vec3(0, 0, d));
	}

	FILE* fp = fopen(argv[1], "wb");
	writeSTL(fp, &trigs[0], N, nullptr, "bac");
	fclose(fp);
	return 0;
}


