// try to balance a 3d object placed on a plane
// by minimizing its gravitational potential energy

// seems like downhill simplex method wins


#include <stdio.h>
#include "numerical/geometry.h"
#include "ui/stl_encoder.h"

#include <vector>
#include <algorithm>


// test object list
#include "modeling/generators/parametric/surfaces.h"
const int OBJ_N = ParamSurfaces.size();


// global variable for debugging
std::vector<stl_triangle> scene_stl;



// see description inside maxDot()
// p.s. set this to 1 implies debug mode
#define SORT_POINTS 0


// find the maximum dot(P[i],n)
// object function to be minimized
double maxDot(const vec3* P, int N, vec3 n) {
	double md = -INFINITY;
	for (int i = 0; i < N; i++) {
		double d = dot(P[i], n);
		if (d > md) md = d;

		// shortcuts may be taken when:
		//  - n is normalized
		//  - P is sorted in decreasing order of length
		// when using `balance_downhillSimplex`, this does not make faster in an average case
		// faster when thousands of samples are needed and the shape is not too elongated
#if SORT_POINTS
		if (P[i].sqr() < md*md)
			return md;
#endif
	}
	return md;
}




#include "numerical/random.h"


// the stupid way, 1000 random samples
vec3 balance_random(const vec3* P, int N) {
	uint32_t seed1 = 0;
	double maxd = INFINITY; vec3 maxn;
	for (int i = 0; i < 1000; i++) {
		vec3 n = rand3(seed1);
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
	double R = calcMaxRadius((triangle*)P, N / 3);

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
			vec3 n_new = axis_angle(rand3(seed1), da) * n;
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
// 60-80 samples in an average case, seems like this one wins
// biggest problem: often converge to a poor local minimum (idea: combine with SA so it can get out?)
vec3 balance_downhillSimplex(const vec3* P, int N) {

	const double accur_eps = 1e-4 * calcMaxRadius((triangle*)P, N / 3);
	const int noimporv_break = 10;
	const int max_iter = 1000;

	// function to be minimized
	int sampleCount = 0;
	auto Fun = [&](vec3 n) {
		sampleCount++;
		return maxDot(P, N, n);
	};

	// kill some local minimums
	uint32_t seed1 = 0;
	double maxd = INFINITY; vec3 maxn(0, 1e-6, -1);
#if 1
	// sometimes almost doubled sampleCount?
	for (int i = 0; i < 10; i++) {
		vec3 n = -rand3_c(seed1);  // may not be evenly distrubuted
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

		// debug output
		if (SORT_POINTS) {
			const double sc = 1.1;
			scene_stl.push_back(stl_triangle(
				triangle{ S[0].n*S[0].val*sc, S[1].n*S[1].val*sc, S[2].n*S[2].val*sc },
				vec3(0.5 + 0.5*cos(0.5*iter), 0.2 + 0.2*sin(0.8*iter), 0.5 - 0.5*cos(0.5*iter))));
		}

		// sorting
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
		S[1].val = Fun(S[1].n);
		S[2].val = Fun(S[2].n);
	}

	printf("%d samples\n", sampleCount);

	return S[0].val < S[1].val && S[0].val < S[2].val ? S[0].n
		: S[1].val < S[2].val ? S[1].n : S[2].n;
}




#include <chrono>




// generate object from object id
// points: non-repeating vertices of the triangles
void generateObject(int id, std::vector<triangle> &trigs,
	std::vector<vec3> *points = nullptr, bool translateToCOM = false) {
	if (points) {
		ParamSurfaces[id].param2points(*points);
		ParamSurfaces[id].points2trigs(&(*points)[0], trigs);
		if (SORT_POINTS) {
			//printf("%x\n", &(*points)[0]);
			std::sort(&(*points)[0], &(*points)[0] + points->size(),
				[](vec3 a, vec3 b) {return a.sqr() > b.sqr(); });
			// idk why this sort has no effect
			// sort it after calling this function
		}
	}
	else {
		ParamSurfaces[id].param2trigs(trigs);
	}

	if (translateToCOM) {
		vec3 COM = calcCOM_shell(&trigs[0], trigs.size());
		for (int i = 0, N = trigs.size(); i < N; i++) trigs[i].translate(-COM);
		if (points)
			for (int i = 0, N = points->size(); i < N; i++) points->operator[](i) -= COM;
	}
}


// visualization for the function that needs to be minimized
void visualizeObjectFunction(const vec3* P, int PN, std::vector<stl_triangle> &trigs) {

	const int SUBDIV = 5;  // subdivision level
	static triangle T[20 << (2 * SUBDIV)]; int TN = 0;  // store triangles
	double minval = INFINITY, maxval = -INFINITY;  // store minimum and maximum sample values

	auto Fun = [&](vec3 n)->vec3 {
		n = normalize(n);
		double mag = maxDot(P, PN, n);
		minval = min(mag, minval), maxval = max(mag, maxval);
		return mag * n;
	};

	// subdividing a regular icosahedron
	const double p = 0.6180339887498948482;
	const triangle ISO[20] = {
		triangle{vec3(0,p,1),vec3(1,0,p),vec3(p,1,0)}, triangle{vec3(0,-p,1),vec3(1,0,p),vec3(0,p,1)}, triangle{vec3(0,-p,1),vec3(p,-1,0),vec3(1,0,p)}, triangle{vec3(0,-p,-1),vec3(-1,0,-p),vec3(0,p,-1)},
		triangle{vec3(-p,1,0),vec3(0,p,1),vec3(p,1,0)}, triangle{vec3(-p,1,0),vec3(p,1,0),vec3(0,p,-1)}, triangle{vec3(-p,1,0),vec3(0,p,-1),vec3(-1,0,-p)}, triangle{vec3(1,0,-p),vec3(1,0,p),vec3(p,-1,0)},
		triangle{vec3(1,0,-p),vec3(0,-p,-1),vec3(0,p,-1)}, triangle{vec3(1,0,-p),vec3(p,1,0),vec3(1,0,p)}, triangle{vec3(1,0,-p),vec3(0,p,-1),vec3(p,1,0)}, triangle{vec3(1,0,-p),vec3(p,-1,0),vec3(0,-p,-1)},
		triangle{vec3(-1,0,p),vec3(0,-p,1),vec3(0,p,1)}, triangle{vec3(-1,0,p),vec3(0,p,1),vec3(-p,1,0)}, triangle{vec3(-1,0,p),vec3(-p,1,0),vec3(-1,0,-p)}, triangle{vec3(-p,-1,0),vec3(p,-1,0),vec3(0,-p,1)},
		triangle{vec3(-p,-1,0),vec3(0,-p,1),vec3(-1,0,p)}, triangle{vec3(-p,-1,0),vec3(-1,0,-p),vec3(0,-p,-1)}, triangle{vec3(-p,-1,0),vec3(0,-p,-1),vec3(p,-1,0)}, triangle{vec3(-p,-1,0),vec3(-1,0,p),vec3(-1,0,-p)},
	};
	for (int F = 0; F < 20; F++) {
		vec3 P = ISO[F].A;
		vec3 A = ISO[F].C - ISO[F].A;
		vec3 B = ISO[F].B - ISO[F].A;

		const int sd = 1 << SUBDIV;
		const double sdm = exp2(-SUBDIV);
		vec3 ns[sd + 1][sd + 1];

		for (int i = 0; i <= sd; i++) {
			for (int j = 0; i + j <= sd; j++) {
				double u = i * sdm, v = j * sdm;
				ns[i][j] = Fun(P + u * A + v * B);
			}
		}

		for (int i = 0; i < sd; i++) {
			for (int j = 0; i + j < sd; j++) {
				vec3 p00 = ns[i][j];
				vec3 p01 = ns[i][j + 1];
				vec3 p10 = ns[i + 1][j];
				vec3 p11 = ns[i + 1][j + 1];
				T[TN++] = triangle{ p01, p00, p10 };
				if (i + j < sd - 1)
					T[TN++] = triangle{ p10, p11, p01 };
			}
		}
	}

	convertTriangles_color(trigs, T, TN, [&](vec3 p) {
		return ColorFunctions::Rainbow((length(p) - minval) / (maxval - minval));
	});
}



// balance a triangulated object placed on a plane
// points: non-repeating vertices on the object
// trigs: faces of the object
void balanceObject(std::vector<vec3> &points, std::vector<triangle> &trigs,
	vec3 translate = vec3(0.), double resize = 1.0) {

	int PN = points.size(), TN = trigs.size();
	vec3 COM = calcCOM_shell(&trigs[0], TN);
	for (int i = 0; i < TN; i++) trigs[i].translate(-COM);
	for (int i = 0; i < PN; i++) points[i] -= COM;

	//vec3 n = balance_random(&points[0], PN);
	//vec3 n = balance_SA_naive(&points[0], PN);
	vec3 n = balance_downhillSimplex(&points[0], PN);
	mat3 M = axis_angle(cross(vec3(0, 0, 1), n), acos(-n.z));

	double s = 1.0;
	if (resize > 0.) {
		//s = 0.4 / calcMaxRadius(&trigs[0], TN);  // faster
		//s = 0.2 / calcGyrationRadius_shell(&trigs[0], TN);  // visually better but slower
		s = 0.18 / calcRotationRadius_shell(&trigs[0], TN);
		s *= resize;
		for (int i = 0; i < TN; i++) trigs[i].scale(s);
	}
	vec3 disp = vec3(0, 0, maxDot(&points[0], PN, n) * s) + translate;
	for (int u = 0; u < TN; u++) {
		trigs[u].applyMatrix(M);
		trigs[u].translate(disp);
	}

}


int main(int argc, char* argv[]) {


	std::vector<triangle> scene;
	scene.reserve(0x200000);

	auto start_time = std::chrono::high_resolution_clock::now();


#if 1
	// test all shapes
	for (int i = 0; i < OBJ_N; i++) {
		std::vector<vec3> points;
		std::vector<triangle> trigs;
		generateObject(i, trigs, &points);
		balanceObject(points, trigs, vec3(i / 8, i % 8, 0));
		scene.insert(scene.end(), trigs.begin(), trigs.end());
	}
#endif


	// visualization for debug
	if (0) do {
		std::vector<vec3> points;
		std::vector<triangle> trigs;
		generateObject(16, trigs, &points, true);
		if (SORT_POINTS) std::sort(points.begin(), points.end(),
			[](vec3 a, vec3 b) {return a.sqr() > b.sqr(); });

		visualizeObjectFunction(&points[0], points.size(), scene_stl);

		double rad = 2.*calcMaxRadius(&trigs[0], trigs.size());
		vec3 d = vec3(rad, 0, 0);
		scene.insert(scene.end(), trigs.begin(), trigs.end());
		for (int i = 0, n = scene.size(); i < n; i++) scene[i].translate(d);

		balanceObject(points, trigs, vec3(0, rad, 0), 0);
		scene.insert(scene.end(), trigs.begin(), trigs.end());
	} while (0);


	// end timer
	auto end_time = std::chrono::high_resolution_clock::now();
	double time_elapsed = std::chrono::duration<double>(end_time - start_time).count();
	printf("%lf secs\n", time_elapsed);

	// write file
	FILE* fp = fopen(argv[1], "wb");
	convertTriangles_color_normal(scene_stl, &scene[0], scene.size(),
		[](vec3 n) { return 0.5*n + vec3(.5); });
	writeSTL(fp, &scene_stl[0], scene_stl.size(), nullptr, "bac");
	fclose(fp);
	return 0;
}


