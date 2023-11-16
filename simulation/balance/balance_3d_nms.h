
#ifndef __INC_GEOMETRY_H
#include "numerical/geometry.h"
#endif

vec3 balance_3d_NMS(const vec3* P, int N, vec3 initial_n, double accur_eps, double *min_energy = nullptr) {

	const int noimporv_break = 10;
	const int max_iter = 1000;

	// function to be minimized
	int sampleCount = 0;
	auto Fun = [&](vec3 n) {
		sampleCount++;
		double md = -INFINITY;
		for (int i = 0; i < N; i++) {
			double d = dot(P[i], n);
			if (d > md) md = d;
		}
		return md;
	};

	// simplex
	struct sample {
		vec3 n;  // normalized vector
		double val;  // value
	} S[3];
	initial_n = normalize(initial_n);
	for (int i = 0; i < 3; i++) {
		double t = 2.*PI*i / 3.;
		S[i].n = axis_angle(cross(initial_n, vec3(0, 0, 1)), acos(initial_n.z))
			* normalize(vec3(.2*cos(t), .2*sin(t), 1.));
		S[i].val = Fun(S[i].n);
	}

	// downhill simplex
	double old_minval = INFINITY;
	int noimporv_count = 0;
	for (int iter = 0; ; iter++) {

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
		else if (++noimporv_count > noimporv_break || iter >= max_iter) {
			//printf("%d samples\n", sampleCount);
			if (min_energy) *min_energy = S[0].val;
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
			expd.n = normalize(center + (center - S[2].n)*2.);
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

}
