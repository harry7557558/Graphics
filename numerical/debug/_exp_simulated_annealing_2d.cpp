#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include "numerical/geometry.h"
#include "numerical/random.h"
#include "ui/stl_encoder.h"


// debug output
std::vector<stl_triangle> STL;
void drawDot(vec3 p, double r, vec3 col) {
	for (int i = 0; i < 8; i++) {
		vec3 v = vec3(0, 0, i >= 4 ? -1. : 1.);
		vec3 e1 = vec3(cossin(.5*PI*i));
		vec3 e2 = vec3(cossin(.5*PI*(i + v.z)));
		STL.push_back(stl_triangle(p + r * v, p + r * e1, p + r * e2, col));
	}
}


// function (continuous)
// >>> DO NOT let the global minimum close to (0,0) cuz that's the initial guess <<<
double Fun0(double x, double y) {
	// not trying to knock the solver down
	auto sq = [](double x) { return x * x; };

	/* Test */
	//return log(sq(y - x * x) + sq(1 - x) + 1.) + .2*sin(5.*x)*cos(5.*y);
	//return 10.*(1. - exp(-.2*sqrt(.5*abs(x*(x - 2.) + y * y)))) - .5*exp(.5*(cos(6.*x - 1.) + cos(7.*y + 1.)));
	//return .1*(x*x + y * y) + sin(x)*cos(y) + sin(2.*x)*cos(2.*y) / 2. + sin(4.*x)*cos(4.*y) / 4. + sin(8.*x)*cos(8.*y) / 8.;
	//return .1*(x * x + y * y) + .1*(cos(6.*x) + cos(8.*y));
	//return .1*(x * x + y * y) + .1*(asin(cos(6.*x)) + asin(cos(8.*y)));
	//return -cos(x)*cos(y)*exp(-(sq(x - PI) + sq(y - PI)));
	//return sqrt(abs(10.*x - y*y)) + .5*abs(y + 1.);
	//return cos(5.*sqrt(x*x + y * y)) + 0.1*(x * x + y * y + x * abs(y)) + .1*sin(10.*x + 3.*y);
	//return std::max({ sqrt(abs(x)), cos(y*y), 0.1*abs(y*y + y) });
	//return sq(x + y) / (x*x + y * y + 1e-8) + .02*acos(sin(10.*x))*asin(cos(10.*y)) + .1*abs(x - y + 1.);

	/* Valley */
	//return 3.*tanh(.5*sqrt(10.*sq(x - y * y) + sq(1 - y)));
	//return 3.*tanh(.5*sqrt(10.*sq(x - y * y + 1.) + .1*sq(1 - y * y)) + .1*y);
	//return 3.*tanh(.1*sqrt(10.*sq(x - y * y + 1.) + .1*sq(1 - y * y)) + .1*y) + .1*asin(cos(5.*x))*sin(1.5*x + 5.*y);
	//return 3.*tanh(.1*pow(10.*sq(x*x + y * y - 10.) + x + .5*y + 5., .4));
	//return 3.*tanh(.1*pow(10.*sq(x*x + y * y - 10.) + 2.*x + y + 10., .4)) + .1*sin(5.*x - 2.*y);
	//return asinh(sq(y - x * (x*x - 3) - 1.)) + 0.1*abs(y);
	//return asinh(sq(y - x * (x*x - 3) - 1.)) + 0.1*(sin(10.*y) + asin(cos(10.*x)) - 5.*x + y * y);
	//return pow(abs(x * (x*x - y * y*y - 3.) + 3.*y + 1.) + 1., .25) - 1. + tanh(.5*sq(x - 1.) + sq(y - 3.));
	//return pow(abs(x * (x*x - y * y*y - 3.) + 3.*y + 1.) + 1., .25) - 1. + tanh(.5*sq(x - 1.) + sq(y - 3.)) + .5*sin(x)*sin(2.*x)*sin(4.*x)*sin(8.*x)*cos(.7*y - x)*cos(3.*y)*cos(5.*y);
	return (tanh(abs(cos(x)*sin(y) + cos(2.*x - 1.) + sin(3.*y))) + 1.)*.5*log(10.*sq(y - x * x + 2.5) + sq(y - 1.5) - .2*x + 1.) + .05*acos(sin(5.*x - 7.*cos(3.*y)));
}


// make the graph more consistent and the solver's life harder
const double Rad = 5.;  // -Rad<x<Rad, -Rad<y<Rad
const int Dif = 320;  // equal-spaced Dif×Dif sample grid
double Grid[Dif + 1][Dif + 1];  // sample values
void initGrid() {  // initialize a grid of samples
	for (int j = 0; j <= Dif; j++) {
		double y = -Rad + j * (2.*Rad) / Dif;
		for (int i = 0; i <= Dif; i++) {
			double x = -Rad + i * (2.*Rad) / Dif;
			Grid[j][i] = Fun0(x, y);
		}
	}
}
int sampleCount = 0;
double Fun(vec2 p) {  // lookup in the grid and interpolate the function
	sampleCount++;
	vec2 q = (p + vec2(Rad)) / (2.*Rad)*Dif;

	int j0 = (int)floor(q.y);
	int i0 = (int)floor(q.x);
	int i = clamp(i0, 0, Dif - 1);
	int j = clamp(j0, 0, Dif - 1);
	// bilinear interpolation
	double u0 = mix(Grid[j][i], Grid[j][i + 1], q.x - i);
	double u1 = mix(Grid[j + 1][i], Grid[j + 1][i + 1], q.x - i);
	double v = mix(u0, u1, q.y - j);
	if (i0 < 0 || i0 >= Dif || j0 < 0 || j0 >= Dif) {  // smoothed blendings
		return mix(v, Fun0(p.x, p.y),
			clamp(max(abs(q.x - .5*Dif), abs(q.y - .5*Dif)) - .5*Dif, 0., 1.));
	}
	return v;
}



// naive simulated annealing
// converges to a location close to the global minimum, but not always stationary
void simulated_annealing_naive() {
	sampleCount = 0;

	// average size of the function
	double R = 5.;

	// simulated annealing
	uint32_t seed1 = 1, seed2 = 2, seed3 = 3;  // random number seeds
	double rand;
	vec2 x = vec2(0.);  // configulation
	double T = 3.0;  // temperature
	double E = Fun(x);  // energy (gravitational potential)
	drawDot(vec3(x, E), .1*T, vec3(1, 0, 0));  // debug output
	vec2 min_x = x; double min_E = E, min_T = T;  // record minimum value encountered
	const int max_iter = 3600;  // number of iterations
	const int max_try = 10;  // maximum number of samples per iteration
	double T_decrease = 0.98;  // multiply temperature by this each time
	for (int iter = 0; iter < max_iter; iter++) {
		for (int ty = 0; ty < max_try; ty++) {
			rand = ((int32_t)lcg_next(seed1) + .5) / 2147483648.;  // -1<rand<1
			vec2 dx = T * erfinv(rand) * cossin(lcg_next(seed1)*(2.*PI / 4294967296.));  // change of configuration
			vec2 x_new = x + dx;  // new configuration
			double E_new = Fun(x_new);
			double prob = exp(-(E_new - E) / (T*R));  // probability, note that E is approximately between -2 and 2
			rand = (seed2 = seed2 * 1664525u + 1013904223u) / 4294967296.;  // 0<=rand<1
			if (prob > rand) {  // jump
				x = x_new, E = E_new;
				if (E < min_E) {
					min_x = x, min_E = E, min_T = T;  // update value
				}
				drawDot(vec3(x, E), .1*T, vec3(1, 0, 0));  // debug output
				break;
			}
		}
		// jump to the minimum point encountered
		double prob = tanh((E - min_E) / R);
		rand = (seed3 = seed3 * 1664525u + 1013904223u) / 4294967296.;
		if (prob > rand) {
			//T = min(min_T, max(T, 0.5 * (E - min_E) / R));
			x = min_x, E = min_E;
		}
		// decrease temperature
		T *= T_decrease;
		if (T < 1e-6) break;
	}

	drawDot(vec3(x, E), 0.05, vec3(1, .5, 0));
	drawDot(vec3(x, E + 1.), 0.2, vec3(1, 1, 0));
	drawDot(vec3(x, E + 5.), 0.5, vec3(1, 1, 0));
	printf("(%lf,%lf,%lf)\n", x.x, x.y, E);
	printf("%d samples\n", sampleCount);
}


// downhill simplex copied from "numerical/optimization.h"
void downhillSimplex_2d() {
	sampleCount = 0;

	const int noimporv_break = 10;
	const int max_iter = 1000;

	struct sample {
		vec2 p;
		double val;
	} S[3];
	for (int i = 0; i < 3; i++) {
		S[i].p = 1.0 * cossin(2.*PI*i / 3.);
		S[i].val = Fun(S[i].p);
	}

	double old_minval = INFINITY;
	int noimporv_count = 0;

	for (int iter = 0; iter < max_iter; iter++) {

		// debug output
		STL.push_back(stl_triangle(
			*(vec3*)&S[0], *(vec3*)&S[1], *(vec3*)&S[2],
			vec3(0.5 + 0.5*cos(0.5*iter), 0.2 + 0.2*sin(0.8*iter), 0.5 - 0.5*cos(0.5*iter))));

		// sort
		sample temp;
		if (S[0].val > S[1].val) temp = S[0], S[0] = S[1], S[1] = temp;
		if (S[1].val > S[2].val) temp = S[1], S[1] = S[2], S[2] = temp;
		if (S[0].val > S[1].val) temp = S[0], S[0] = S[1], S[1] = temp;

		// termination condition
		if (S[0].val < old_minval - 1e-6) {
			noimporv_count = 0;
			old_minval = S[0].val;
		}
		else if (++noimporv_count > noimporv_break) {
			goto Finish;
		}

		// reflection
		sample refl;
		vec2 center = (S[0].p + S[1].p) * .5;
		refl.p = center * 2. - S[2].p;
		refl.val = Fun(refl.p);
		if (refl.val >= S[0].val && refl.val < S[1].val) {
			S[2] = refl;
			continue;
		}

		// expansion
		if (refl.val < S[0].val) {
			sample expd;
			expd.p = center + (center - S[2].p)*2.;
			expd.val = Fun(expd.p);
			if (expd.val < refl.val)
				S[2] = expd;
			else
				S[2] = refl;
			continue;
		}

		// contraction
		sample ctrct;
		ctrct.p = center + .5*(S[2].p - center);
		ctrct.val = Fun(ctrct.p);
		if (ctrct.val < S[2].val) {
			S[2] = ctrct;
			continue;
		}

		// compression
		S[1].p = S[0].p + (S[1].p - S[0].p)*.5;
		S[2].p = S[0].p + (S[2].p - S[0].p)*.5;

	}

Finish:;
	vec3 P = *(vec3*)&(S[0].val < S[1].val && S[0].val < S[2].val ? S[0]
		: S[1].val < S[2].val ? S[1] : S[2]);

	drawDot(P, 0.05, vec3(0, 1, 0.5));
	drawDot(P + vec3(0, 0, 1), 0.2, vec3(0, 1, 1));
	drawDot(P + vec3(0, 0, 5), 0.5, vec3(0, 1, 1));
	printf("(%lf,%lf,%lf)\n", P.x, P.y, P.z);
	printf("%d samples\n", sampleCount);
}



int main(int argc, char* argv[]) {
	initGrid();

	// plot function
	static stl_triangle trigs[2*Dif*Dif];
	stl_fun2trigs([](double x, double y) { return Fun(vec2(x, y)); }, trigs, -Rad, Rad, -Rad, Rad, Dif, Dif, -20., 20.);
	STL.assign(&trigs[0], &trigs[2 * Dif*Dif]);

	// optimization
	simulated_annealing_naive();
	downhillSimplex_2d();

	FILE *fp = fopen(argv[1], "wb");
	writeSTL(fp, &STL[0], STL.size(), nullptr, STL_CCW);
	fclose(fp);
	return 0;
}

