// Curve fitting experiment - fit 2d parametric curves to cubic Bezier spline
// Using the reparameterization algorithm that is commonly used in vector graphics softwares
// For parametric curves, it doesn't work better than numerical optimization; (faster but very unstable, produces a large number of curve pieces)

#include <stdio.h>
#include <functional>
#include <vector>
#include <string>

#include "numerical/geometry.h"
#include "numerical/integration.h"
#include "numerical/optimization.h"
#include "numerical/rootfinding.h"

#define printerr(...) fprintf(stderr, __VA_ARGS__)
#define printerr(...) {}



// not a vector
inline bool isNAV(vec2 p) {
	double m = p.sqr();
	return isnan(m) || m > 1e32;
}


// samples and line segments
struct param_sample {
	double t;
	vec2 p;
	param_sample(double t = NAN, vec2 p = vec2(NAN))
		:t(t), p(p) {}
};
struct segment {
	vec2 p, q;
};
struct segment_sample {
	param_sample p, q;
	vec2 pq() { return q.p - p.p; }
};

// distance to a line segment
double distSegment(vec2 a, vec2 b, vec2 p) {
	vec2 ap = p - a, ab = b - a;
	double t = dot(ap, ab) / dot(ab, ab);
	return length(ap - ab * clamp(t, 0., 1.));
}


// parametric curve discretization

// P(t0) is NAN and P(t1) is not, find the t closest to t0 such that P(t) is not NAN
template<typename Fun>
bool bisectNAV(const Fun &P, double t0, double t1, vec2 p0, vec2 p1, double &t, vec2 &p, double eps = 1e-12) {
	if (!isNAV(p0)) return false;
	if (isNAV(p1)) return false;
	uint32_t seed = 0;  // requires some hacks for it to work
	for (int i = 0; i < 64; i++) {
		double random = (seed = seed * 1664525u + 1013904223u) / 4294967296.;
		t = t0 + (0.49 + 0.02*random)*(t1 - t0), p = P(t);
		if (isNAV(p)) t0 = t, p0 = p;
		else t1 = t, p1 = p;
		if (abs(t1 - t0) < eps) {
			p = p1;
			return true;
		}
	}
	return false;
}

// bound jump discontinuity, return true if there is a jump and false if the function seems continuous
// throw exception if it encounters a NAN
template<typename Fun/*vec2(double)*/>
bool boundJump(const Fun &P, double &t0, double &t1, vec2 &p0, vec2 &p1, double eps = 1e-12, int max_iter = 64) {
	double dt = t1 - t0;
	double dif0 = length(p1 - p0) / dt;
	double dist = dif0 * dt;
	for (int i = 0; i < max_iter; i++) {
		double tc = 0.5*(t0 + t1);
		vec2 pc = P(tc);
		if (isNAV(pc)) throw(tc);
		dt *= .5;
		double dif1 = length(pc - p0) / dt;
		double dif2 = length(p1 - pc) / dt;
		if (i > 2 && !(max(dif1, dif2) > 1.2*dif0)) return false;
		if (dif1 > dif2) t1 = tc, p1 = pc, dif0 = dif1;
		else t0 = tc, p0 = pc, dif0 = dif2;
		double new_dist = dif0 * dt;
		// consider changing the termination condition (p-based) to save samples
		if (t1 - t0 < eps) {
			if (new_dist / dist < 0.99) {  // make sure it is not continuous
				if (!(t1 == t0 || i == max_iter - 1)) continue;
			}
			return true;
		}
		dist = new_dist;
	}
	// this case: continuous with divergent tangent
	return false;
};


// recursive part for parametric curve discretization
// see discretize_test.cpp for details
std::vector<segment_sample> discretizeParametricCurve_rec(std::function<vec2(double)> F,
	double t0, double t1, vec2 P0, vec2 P1,
	int min_dif, double reqLength, double reqError, int recurse_remain,
	param_sample P_0 = param_sample(), param_sample P_1 = param_sample(),
	bool sin_left = false, bool sin_right = false) {

	std::vector<segment_sample> res;
	if (recurse_remain < 0) {  // recursion limit exceeded
		res.push_back(segment_sample{ param_sample(t0,P0), param_sample(t1,P1) });
		return res;
	}

	// user call
	if (min_dif > 0) {
		// handle the case when t0 or t1 is INF
		std::function<vec2(double)> _F = [&](double t) { return F(t); };
		if (abs(t1 - t0) > 1e4) {  // INF case
			_F = [&](double t) { return F(tan(t)); };
			t0 = atan(t0), t1 = atan(t1);
		}

		reqLength *= 2.0 * 1.3;  // hmmm...
		if (!(reqLength > 0. && reqError > 0.)) return res;  // no messing around

		// take samples
		double dt = (t1 - t0) / min_dif;
		param_sample *samples = new param_sample[min_dif + 1];
		for (int i = 1; i < min_dif; i++) {
			double t = t0 + (i + 0.01*sin(123.456*i)) * dt;
			samples[i] = param_sample(t, _F(t));
			//res.push_back(segment_sample{ param_sample(t, samples[i].p - vec2(.1,0)), param_sample(t, samples[i].p + vec2(.1,0)) });
		}
		samples[0] = param_sample(t0, P0);
		samples[min_dif] = param_sample(t1, P1);

		// recursive calls
		for (int i = 0; i < min_dif; i++) {
			std::vector<segment_sample> app = discretizeParametricCurve_rec(_F,
				samples[i].t, samples[i + 1].t, samples[i].p, samples[i + 1].p,
				0, reqLength, reqError, recurse_remain - 1,
				i == 0 ? param_sample() : samples[i - 1],
				i + 1 == min_dif ? param_sample() : samples[i + 2]
			);
			res.insert(res.end(), app.begin(), app.end());
		}

		/*
		After call, check:
		 * too large turning angles;
		 * rapid change in chord lengths;
		 * discontinuities in segments;
		Perform bisection / golden section search on the dot product of the curve with a chosen vector
			to minimize the number of incorrect samples.
		*/

		delete samples;
		return res;
	}

	// handle NAN
	if (isNAV(P0) && isNAV(P1))
		return res;
	if (isNAV(P0)) {
		if (!bisectNAV(F, t0, t1, P0, P1, t0, P0)) return res;
		P_0.t = NAN;
	}
	else if (isNAV(P1)) {
		if (!bisectNAV(F, t1, t0, P1, P0, t1, P1)) return res;
		P_1.t = NAN;
	}

	// subdivition global variables
	double tc, tc0, tc1;
	vec2 pc, pc0, pc1;
	bool hasJump = false;
	param_sample s0{ t0, P0 }, s1{ t1, P1 };

	// continue subdivision until error is small enough
	vec2 dP = P1 - P0;
	double dPL = length(dP);
	if (dPL == 0.0) return res;
	if (dPL < reqLength || min_dif == -1) {

		// split at the point(s) that produce the maximum value
		vec2 tcp = vec2(.5*(t0 + t1), NAN);

		if (isnan(tcp.y)) {  // divide into 2
			tc = tcp.x; pc = F(tc);
			if (distSegment(P0, P1, pc) < reqError) {
				param_sample sc{ tc, pc };
				res.push_back(segment_sample{ s0, sc });
				res.push_back(segment_sample{ sc, s1 });
				return res;
			}
			// check jump discontinuity
			if (!(sin_left || sin_right)) {
				double l0 = length(pc - P0) / (tc - t0), l1 = length(pc - P1) / (t1 - tc);
				if (l0 > 2.*l1 || l1 > 2.*l0) {
					double u0 = t0, u1 = t1; vec2 v0 = F(u0), v1 = F(u1);
					try {
						bool succeed = boundJump(F, u0, u1, v0, v1);
						tc0 = u0, tc1 = u1, pc0 = v0, pc1 = v1;
						hasJump = succeed;
						goto divideJump;
					}
					catch (double t) {
						tc = t, pc = vec2(NAN);
						goto divide2;
					}
				}
			}
			goto divide2;
		}
		else {  // divide into 3
			tc0 = tcp.x, tc1 = tcp.y;
			pc0 = F(tc0), pc1 = F(tc1);
			param_sample sc0{ tc0, pc0 }, sc1{ tc1, pc1 };
			if (distSegment(P0, P1, pc0) < reqError && distSegment(P0, P1, pc1) < reqError) {
				res.push_back(segment_sample{ s0, sc0 });
				res.push_back(segment_sample{ sc0, sc1 });
				res.push_back(segment_sample{ sc1, s1 });
				return res;
			}
			goto divide3;
		}

	}

	// normal split
	else {
#if 0
		double Th_low = 1.9, Th_high = 2.9;  // experimental values
		if (dPL > Th_low*reqLength && dPL < Th_high*reqLength) {  // divide into 3
			tc0 = t0 + (t1 - t0) / 3.;  // experimental value
			tc1 = t1 - (t1 - t0) / 3.;  // experimental value
			pc0 = F(tc0), pc1 = F(tc1);
			goto divide3;
		}
		else
#endif
		{  // divide into 2
			tc = 0.5*(t0 + t1);  // experimental value
			pc = F(tc);
			// check jump discontinuity
			double l0 = length(pc - P0), l1 = length(pc - P1);
			if (!(sin_left || sin_right) && min(l0, l1) < reqLength) {
				if (l0 > 2.*l1 || l1 > 2.*l0) {
					double u0 = t0, u1 = t1; vec2 v0 = F(u0), v1 = F(u1);
					try {
						bool succeed = boundJump(F, u0, u1, v0, v1);
						tc0 = u0, tc1 = u1, pc0 = v0, pc1 = v1;
						hasJump = succeed;
						goto divideJump;
					}
					catch (double t) {
						tc = t, pc = vec2(NAN);
						goto divide2;
					}
				}
			}
			goto divide2;
		}
	}

	// should never get here
	throw("bug");
	return res;


divide2:
	{
		// split into 2
		std::vector<segment_sample> app0 = discretizeParametricCurve_rec(F,
			t0, tc, P0, pc, 0, reqLength, reqError, recurse_remain - 1,
			P_0, param_sample(t1, P1), sin_left, false);
		std::vector<segment_sample> app1 = discretizeParametricCurve_rec(F,
			tc, t1, pc, P1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(t0, P0), P_1, false, sin_right);
		// try to fix missed samples
		double l0 = length(pc - P0), l1 = length(P1 - pc);
		int n0 = app0.size(), n1 = app1.size();
		if (n0 && n1 && app0.back().q.t == app1.front().p.t) {
			double m0 = l0 / n0, m1 = l1 / n1;
			if ((max(n0 / n1, n1 / n0) > 4 && (
				abs(ndet(app0.back().pq(), -app1.front().pq())) > .2  // apparently doesn't always work
				|| (n1 == 2 && length(app1[0].pq()) > 4.*length(app0.back().pq()))
				|| (n0 == 2 && length(app0.back().pq()) > 4.*length(app0[0].pq())))
				)) {
				//printf("(%lf,%lf) %d %d\n", pc.x, pc.y, n0, n1);
				if (n0 <= 2 || n1 <= 2) {
					std::vector<segment_sample> *app_s = n1 < n0 ? &app1 : &app0;
					std::vector<segment_sample> app;
					int nm = min(n0, n1);
					for (int i = 0; i < nm; i++) {
						segment_sample ps = (*app_s)[i];
						std::vector<segment_sample> tmp = discretizeParametricCurve_rec(F,
							ps.p.t, ps.q.t, ps.p.p, ps.q.p, 0, reqLength, reqError, recurse_remain - 2,
							(*app_s)[max(i - 1, 0)].q, (*app_s)[min(i + 1, nm - 1)].p
						);
						app.insert(app.end(), tmp.begin(), tmp.end());
					}
					*app_s = app;
				}
			}
		}
		// add
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), app1.begin(), app1.end());
		return res;
	}

divide3:
	{
		// split into 3
		std::vector<segment_sample> app0, appc, app1;
		app0 = discretizeParametricCurve_rec(F,
			t0, tc0, P0, pc0, 0, reqLength, reqError, recurse_remain - 1,
			P_0, param_sample(tc1, pc1), sin_left, false);
		appc = discretizeParametricCurve_rec(F,
			tc0, tc1, pc0, pc1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(t0, P0), param_sample(t1, P1));
		app1 = discretizeParametricCurve_rec(F,
			tc1, t1, pc1, P1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(tc0, pc0), P_1, false, sin_right);
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), appc.begin(), appc.end());
		res.insert(res.end(), app1.begin(), app1.end());
		return res;
	}

divideJump:
	{
		// split by at discontinuity
		std::vector<segment_sample> app0, appc, app1;
		param_sample sc0{ tc0, pc0 }, sc1{ tc1, pc1 };
		app0 = discretizeParametricCurve_rec(F,
			t0, tc0, P0, pc0, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(), param_sample(), false, true);
		if (!hasJump && tc1 - tc0 > 1e-6) appc = discretizeParametricCurve_rec(F,
			tc0, tc1, pc0, pc1, 0, reqLength, reqError, recurse_remain - 1);
		app1 = discretizeParametricCurve_rec(F,
			tc1, t1, pc1, P1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(), param_sample(), true, false);
		res.insert(res.end(), app0.begin(), app0.end());
		if (!appc.empty()) res.insert(res.end(), appc.begin(), appc.end());
		else if (!hasJump) res.push_back(segment_sample{ sc0, sc1 });
		res.insert(res.end(), app1.begin(), app1.end());
		return res;
	}

}

// discretize a parametric curve to a list of lists of continued points splitted by jump discontinuities
std::vector<std::vector<vec2>> discretizeParametricCurve(std::function<vec2(double)> F, double t0, double t1,
	double reqLength, double reqErr, int min_dif = 24, int max_recur = 18) {

	std::vector<segment_sample> FS = discretizeParametricCurve_rec(F, t0, t1, F(t0), F(t1),
		min_dif, reqLength, reqErr, max_recur);

	std::vector<std::vector<vec2>> res;
	std::vector<vec2> tmp;
	if (FS.empty()) return res;

	vec2 p0 = FS[0].p.p;
	for (int i = 0, FSN = FS.size(); i < FSN; i++) {
		segment_sample ss = FS[i];
		if (ss.p.p != p0) {
			tmp.push_back(p0);
			res.push_back(tmp);
			tmp.clear();
		}
		tmp.push_back(ss.p.p);
		p0 = ss.q.p;
	}
	tmp.push_back(p0);
	res.push_back(tmp);
	return res;

}






// splines
struct cubicBezier {
	vec2 A, B, C, D;
};
struct cubicCurve {
	vec2 c0, c1, c2, c3;  // c0 + c1 t + c2 t² + c3 t³, 0 < t < 1
	vec2 eval(double t) { return ((c3*t + c2)*t + c1)*t + c0; }
};

// expand a cubic bezier curve to polynomial coefficients
cubicCurve bezierAlg(vec2 A, vec2 B, vec2 C, vec2 D) {
	cubicCurve s;
	s.c0 = A;
	s.c1 = -3 * A + 3 * B;
	s.c2 = 3 * A - 6 * B + 3 * C;
	s.c3 = -A + 3 * B - 3 * C + D;
	return s;
}



// calculate the square of distance to a cubic parametric curve
#include "cubicdist.h"
static uint32_t distCubic2_callCount = 0;  // a counter for testing
double distCubic2(cubicCurve c, vec2 p) {
	distCubic2_callCount++;
	return CubicCurveDistance2(&c.c0, p);
};
double nearestPointOnCurve(cubicCurve c, vec2 p) {  // return the parameter value
	distCubic2_callCount++;
	double t; CubicCurveDistance2(&c.c0, p, &t);
	return t;
}



// analytically find u and v to minimize Σ[w·(Bezier(t)-P)²] with control points P0+u*T0,P1-v*T1
vec2 fitCurve_minimizeParameterSumSqr(vec2 P0, vec2 P1, vec2 T0, vec2 T1,
	int N, const vec2 P[], const double t[], const double w[]) {
	mat2 M(0.); vec2 b(0.);
	for (int i = 0; i < N; i++) {
		double ti = t[i], t2 = ti * ti, t3 = t2 * ti, t4 = t3 * ti, t5 = t4 * ti, t6 = t5 * ti;
		vec2 Pi = P[i];
		double k0 = dot(P0, T0)*(2.*t6 - 7.*t5 + 8.*t4 - 2.*t3 - 2.*t2 + ti)
			+ dot(P1, T0)*(-2.*t6 + 7.*t5 - 8.*t4 + 3.*t3)
			+ dot(Pi, T0)*(-t3 + 2.*t2 - ti);
		double u0 = dot(T0, T0)*(3.*t6 - 12.*t5 + 18.*t4 - 12.*t3 + 3.*t2);
		double v0 = dot(T0, T1)*(3.*t6 - 9.*t5 + 9.*t4 - 3.*t3);
		double k1 = dot(P0, T1)*(2.*t6 - 5.*t5 + 3.*t4 + t3 - t2)
			+ dot(P1, T1)*(-2.*t6 + 5.*t5 - 3.*t4)
			+ dot(Pi, T1)*(-t3 + t2);
		double u1 = dot(T0, T1)*(3.*t6 - 9.*t5 + 9.*t4 - 3.*t3);
		double v1 = dot(T1, T1)*(3.*t6 - 6.*t5 + 3.*t4);
		M += w[i] * mat2(u0, v0, u1, v1);
		b += w[i] * vec2(-k0, -k1);
	}
	return M.inverse()*b;
}

// analytically find control points C0 and C1 to minimize Σ[w·(Bezier(t)-P)²]
void fitCurve_minimizeParameterSumSqr_4(vec2 P0, vec2 P1, vec2 &C0, vec2 &C1,
	int N, const vec2 P[], const double t[], const double w[]) {
	mat2 M(0.), b(0.);
	for (int i = 0; i < N; i++) {
		double ti = t[i];
		double m0 = (1 - ti)*(1 - ti)*(1 - ti);
		double n0 = 3.*ti*(1 - ti)*(1 - ti);
		double n1 = 3.*ti*ti*(1 - ti);
		double m1 = ti * ti*ti;
		vec2 Pm = m0 * P0 + m1 * P1 - P[i];
		double u0 = n0 * n0, u1 = n0 * n1;
		double v0 = n0 * n1, v1 = n1 * n1;
		M += w[i] * mat2(u0, u1, v0, v1);
		b += w[i] * mat2(n0*Pm, n1*Pm);
	}
	b = -b * M.inverse();
	C0 = b.column(0);
	C1 = b.column(1);
}



std::vector<cubicBezier> fitSpline(const vec2 P[], int N, double maxerr,
	double *ld = nullptr, double *wt = nullptr) {

	// @ld, @wt: used in recursive calls to save computing cost

	std::vector<cubicBezier> res;
	if (N < 2) return res;
	if (N == 2) {
		res.push_back(cubicBezier{ P[0], vec2(NAN), vec2(NAN), P[1] });
		return res;
	}

	// parameters
	double *T = new double[N];
	T[0] = 0.;
	for (int i = 1; i < N; i++)
		T[i] = T[i - 1] + length(P[i] - P[i - 1]);
	double tm = 1. / T[N - 1];
	for (int i = 1; i < N; i++) T[i] *= tm;
	// weights
	double *W = new double[N];
	W[0] = length(P[1] - P[0]);
	W[N - 1] = length(P[N - 1] - P[N - 2]);
	for (int i = 1; i < N - 1; i++)
		W[i] = length(P[N + 1] - P[N - 1]);
	// calculated fitting error at each point
	double *Err = new double[N];

	// fitting
	vec2 P0 = P[0], P1 = P[N - 1];
	vec2 C0, C1, C0_old, C1_old;
	// iterative fitting - linear convergence with varying base
	for (int iter = 0; ; iter++) {
		// curve fitting
		fitCurve_minimizeParameterSumSqr_4(P0, P1, C0, C1, N, P, T, W);
		// reparameterization and error calculation
		cubicCurve cc = bezierAlg(P0, C0, C1, P1);
		for (int i = 0; i < N; i++) {
			double t = nearestPointOnCurve(cc, P[i]);
			T[i] = t, Err[i] = length(cc.eval(t) - P[i]);
		}
		// check if the error is small enough
		double maxe = -1.; int maxe_i = N / 2;
		for (int i = 0; i < N; i++) {
			if (Err[i] > maxe) maxe = Err[i], maxe_i = i;
		}
		if (maxe >= 0. && maxe < maxerr) {
			// good, fit with the new parameterization
			fitCurve_minimizeParameterSumSqr_4(P0, P1, C0, C1, N, P, T, W);
			res.push_back(cubicBezier{ P0, C0, C1, P1 });
			return res;
		}

		// error too large, split the curve
		if (iter > 8 || isNAV(C0) || isNAV(C1)) {
			if (!(maxe_i > 0.3*N && maxe_i < 0.7*N)) maxe_i = N / 2;
			std::vector<cubicBezier> a0 = fitSpline(P, maxe_i + 1, maxerr);
			std::vector<cubicBezier> a1 = fitSpline(&P[maxe_i], N - maxe_i, maxerr);
			res.insert(res.end(), a0.begin(), a0.end());
			res.insert(res.end(), a1.begin(), a1.end());
			return res;
		}

		//res.push_back(cubicBezier{ P0, C0, C1, P1 });
		C0_old = C0, C1_old = C1;
	}

	delete T; delete W; delete Err;
	return res;
}




// this function does everything automatically (discretization + fitting)
template<typename ParamCurve>
std::vector<cubicBezier> fitSpline_auto(ParamCurve C, vec2 B0, vec2 B1,
	double allowed_err = 0.001, double* Err = nullptr, double *LengthC = nullptr) {

	// substitution for INF parameters
	if (abs(C.t1 - C.t0) > 1e4) {
		// use tan instead of atanh because tan(atan(x))!=INF
		return fitSpline_auto(
			ParamCurve([&](double t) { return C.p(tan(t)); }, atan(C.t0), atan(C.t1)),
			B0, B1, allowed_err, Err, LengthC);
	}

	std::vector<std::vector<vec2>> CS = discretizeParametricCurve(C.p, C.t0, C.t1, 50 * allowed_err, allowed_err);
	int CSN = CS.size();

	std::vector<cubicBezier> Res;
	double cumErr = 0., cumLength = 0.;

	for (int T = 0; T < CSN; T++) {
		std::vector<vec2> Cs = CS[T];
		int CsN = Cs.size();
		double err = 0., clen = 0.;
		std::vector<cubicBezier> R = fitSpline(&Cs[0], CsN, allowed_err);
		Res.insert(Res.end(), R.begin(), R.end());
		cumErr += err, cumLength += clen;
	}

	if (Err) *Err = cumErr;
	if (LengthC) *LengthC = cumLength;
	return Res;
}





// include test cases
#include "test_cases.h"




// timer
#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

// parameters of the output graph
namespace SVG {
	const int Graph_N = CS1 - CS0;  // number of function graphs
	const int W = 600, H = 400;  // width and height of each function graph
	const double SC = 0.2*W, invSC = 1.0 / SC;  // scale from function coordinate to screen coordinate
	const vec2 Bound = 0.5 * vec2(W, H) * invSC;  // viewbox for each function graph: ±Bound
	const int colspan = 2;  // number of columns of function graphs
	const bool drawChart = Graph_N > 2 * colspan;  // draw charts or not
	const int Chart_N = 7;  // number of statistic charts
	const int Chart_W = 800, Chart_H = 300;  // width and height of each chart
	const vec2 Chart_Margin = vec2(120, 60);  // horizontal and vertical margin of each chart
	const int Width = max(min(Graph_N, colspan) * W, drawChart * (Chart_W + (int)Chart_Margin.x));  // width of the overall graph
	const int Height_F = ((int)((Graph_N - 1) / colspan + 1)*H);  // total height of the function part
	const int Height = Height_F + (int)drawChart * Chart_N * int(Chart_H + 2. * Chart_Margin.y) + 80;  // height of the overall graph
}

// double to string for printf
std::string _f_buffer[0x100];
int _f_buffer_n = 0;
auto _ = [&](double x) {
	char c[256];
	std::string s;
	if (isnan(x)) s = true ? "nan" : "nan";
	else if (x*x >= 1e12) s = true ? (x > 0. ? "1e6" : "-1e6") : "inf";
	else if (x*x >= 1e8) s = std::to_string((int)round(x));
	else if (x*x < 1e-8) s = "0";
	else {
		sprintf(c, x*x >= 1. ? "%.4lg" : "%.4lf", x);
		s = c;
		int d = s.find("0."); if (d != -1) s.erase(s.begin() + d);
		if (s.find('.') != -1)
			while (s[s.size() - 1] == '0' || s[s.size() - 1] == '.') s.erase(s.size() - 1);
		if (s.empty()) s = "0";
	}
	_f_buffer_n %= 0x100;
	_f_buffer[_f_buffer_n] = s;
	return &(_f_buffer[_f_buffer_n++][0]);
};




// fit a curve with the default parameter and write SVG element code to stdout
void drawFittedCurve(const ParametricCurveL &C, int &spn, double &err, double &time_elapsed, int id = -1) {

	// axis
	{
		printf("<g class='axes'>");
		printf("<line x1='%s' x2='%s' y1='0' y2='0'/><line x1='1' x2='1' y1='-.05' y2='.05'/>", _(-SVG::Bound.x), _(SVG::Bound.x));
		printf("<line x1='0' x2='0' y1='%s' y2='%s'/><line x1='-.05' x2='.05' y1='1' y2='1'/>", _(-SVG::Bound.y), _(SVG::Bound.y));
		printf("</g>\n");
	}

	// timer
	auto time0 = NTime::now();
	time_elapsed = 0.;
	auto start_timer = [&]() { time0 = NTime::now(); };
	auto end_timer = [&]() { time_elapsed += fsec(NTime::now() - time0).count(); };

	// debug
	if (1) {
		start_timer();
		std::vector<std::vector<vec2>> S = discretizeParametricCurve([&](double t) {
			vec2 p = C.p(t);
			return abs(p.x) < SVG::Bound.x && abs(p.y) < SVG::Bound.y ?
				p : vec2(NAN);
		}, C.t0, C.t1, 0.05, 0.001, 100);
		end_timer();
		spn = 0;
		for (int T = 0, SN = S.size(); T < SN; T++) {
			std::vector<vec2> s = S[T];
			int sn = s.size();
			printf("<path d='");
			for (int i = 0; i < sn; i++) {
				printf("%c%s,%s", i ? 'L' : 'M', _(s[i].x), _(s[i].y));
			}
			spn += sn;
			printf("' style='stroke:%s;stroke-width:1px;fill:none' vector-effect='non-scaling-stroke'/>\n", T & 1 ? "#00F" : "#F00");
		}
		//fflush(stdout); return;
	}

	Parametric_callCount = 0;
	distCubic2_callCount = 0;
	_f_buffer_n = 0;

	// vectorized path
	{
		double clength;
		start_timer();
		std::vector<cubicBezier> sp = fitSpline_auto(C, -SVG::Bound, SVG::Bound, 0.005, &err, &clength);
		end_timer();
		err = sqrt(err / clength);
		spn = sp.size();

		printf("<path class='vecpath' d='");
		vec2 oldP(NAN);
		for (int i = 0; i < spn; i++) {
			vec2 P0 = sp[i].A, Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
			if (isNAV(P0) || isNAV(P1)) continue;  // abnormal
			if (!((P0 - oldP).sqr() < 1e-8))
				printf("\nM%s,%s", _(P0.x), _(P0.y));
			if (isNAV(Q0) && isNAV(Q1)) printf("L%s,%s", _(P1.x), _(P1.y));
			else printf("C%s,%s %s,%s %s,%s", _(Q0.x), _(Q0.y), _(Q1.x), _(Q1.y), _(P1.x), _(P1.y));
			oldP = P1;
		}
		printf("\n' stroke-width='1' vector-effect='non-scaling-stroke'/>\n");

		// anchor points
		if (spn && spn < 180) {
			printf("<g class='anchors' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)'>");
			auto line = [](vec2 a, vec2 b) {
				if (!(isNAV(a) || isNAV(b)))
					printf("<line x1='%s' y1='%s' x2='%s' y2='%s'/>", _(a.x), _(a.y), _(b.x), _(b.y));
			};
			for (int i = 0; i < spn; i++) {
				vec2 P0 = sp[i].A, Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
				line(P0, Q0); line(P1, Q1);
			}
			printf("</g>\n");
		}
	}

	fflush(stdout);
}


// draw a chart and write SVG element code to stdout, assume data points are non-negative
// when grid is non-positive: automatically
void drawChart(/*non-const warning*/vec2 Data[SVG::Graph_N], const char x_name[], const char y_name[], vec2 grid, const char unit_x[] = "", const char unit_y[] = "", bool show_linear = true) {

	// calculate maximum value for determining grid size
	vec2 maxVal(0.);
	for (int i = 0; i < SVG::Graph_N; i++) maxVal = pMax(maxVal, Data[i]);

	// automatic k/M for large numbers
	char unitx[2] = " ", unity[2] = " ";
	vec2 Mc(1.);
	unitx[0] = maxVal.x > 1e6 ? ('M' + char(Mc.x = 1e-6)) : maxVal.x > 1e3 ? ('k' + char(Mc.x = 1e-3)) : 0;
	unity[0] = maxVal.y > 1e6 ? ('M' + char(Mc.y = 1e-6)) : maxVal.y > 1e3 ? ('k' + char(Mc.y = 1e-3)) : 0;
	for (int i = 0; i < SVG::Graph_N; i++) Data[i] *= Mc;
	maxVal *= Mc;
	double uk = Mc.x / Mc.y;  // dy/dx unit

	// automatic grid size
	if (!(grid.x > 0.)) { double q = log10(maxVal.x), r = q - floor(q); grid.x = pow(10., floor(q)) * (r < .17 ? .1 : r < .5 ? .2 : r < .83 ? .5 : 1.); }
	if (!(grid.y > 0.)) { double q = log10(maxVal.y), r = q - floor(q); grid.y = pow(10., floor(q)) * (r < .17 ? .2 : r < .5 ? .5 : r < .83 ? 1. : 2.); }
	if (maxVal.x == 0.) maxVal.x = grid.x = 1.;
	if (maxVal.y == 0.) maxVal.y = grid.y = 1.;
	vec2 Sc = vec2(SVG::Chart_W, SVG::Chart_H) / (maxVal = pMin(maxVal + grid, maxVal * 1.2));

	// lables
	printf("<text text-anchor='middle' style='font-size:20px;font-family:\"Times New Roman\"' x='%lg' y='%d'>%s</text>\n",
		SVG::Chart_W*.5, SVG::Chart_H + 50, x_name);
	printf("<text transform='translate(%d %lg) rotate(-90)' text-anchor='middle' style='font-size:20px;font-family:\"Times New Roman\"' x='0' y='0'>%s</text>\n",
		-60, SVG::Chart_H*.5, y_name);

	// scale
	printf("<rect x='0' y='0' width='%d' height='%d' style='stroke-width:1px;stroke:#ccc;fill:none;'/>\n", SVG::Chart_W, SVG::Chart_H);
	printf("<g class='chart-scale'>\n");
	printf("<path d='");
	std::string lables = ""; char cbuf[1024];
	for (int i = 0, imax = (int)floor(maxVal.x / grid.x); i <= imax; i++) {
		double x = Sc.x * i * grid.x;
		sprintf(cbuf, "<text text-anchor='middle' x='%.4lg' y='%d'>%.5lg%s%s</text>", x, SVG::Chart_H + 20, i*grid.x, unitx, unit_x);
		lables += std::string(cbuf);
		printf("M%.4lg,0v%d", x, SVG::Chart_H + 5);
	}
	for (int i = 0, imax = (int)floor(maxVal.y / grid.y); i <= imax; i++) {
		double y = Sc.y * i * grid.y;
		sprintf(cbuf, "<text text-anchor='end' x='%d' y='%.4lg'>%.5lg%s%s</text>", -10, SVG::Chart_H - (y - 5), i*grid.y, unity, unit_y);
		lables += std::string(cbuf);
		printf("M%d,%.4lgh%d", -5, SVG::Chart_H - y, SVG::Chart_W + 5);
	}
	printf("' style='stroke-width:0.5;stroke:#888;fill:none;'/>\n");
	printf("%s\n", &lables[0]);
	printf("</g>\n");

	// prevent text selection
	printf("<rect x='%lg' y='%lg' width='%lg' height='%lg' style='stroke:none;fill:#00000000;'/>\n",
		-SVG::Chart_Margin.x, -SVG::Chart_Margin.y, SVG::Chart_W + SVG::Chart_Margin.x*2., SVG::Chart_H + SVG::Chart_Margin.y*2.);

	// lines of best fit
	if (show_linear) {
		double sumxy = 0., sumx2 = 0;
		for (int i = 0; i < SVG::Graph_N; i++)
			sumxy += Data[i].x*Data[i].y, sumx2 += Data[i].x*Data[i].x;
		double k = sumxy / sumx2;  // slope of the line of best fit
		vec2 P = vec2(maxVal.x, k*maxVal.x);
		if (P.y > maxVal.y) P = vec2(maxVal.y / k, maxVal.y);
		printf("<line x1='%d' y1='%d' x2='%lg' y2='%lg' style='stroke:gray;stroke-width:2px;stroke-dasharray:10'/>\n",
			0, SVG::Chart_H, P.x*Sc.x, SVG::Chart_H - P.y*Sc.y);
		char val[64]; sprintf(val, "%.4lg%s", k,
			uk == 1e3 ? "k" : uk == 1e6 ? "M" : uk == 1e-3 ? "/k" : uk == 1e-6 ? "/M" : "");
		printf("<text text-anchor='end' x='%d' y='%d'>m = %s</text>\n",
			SVG::Chart_W - 20, 20, val);
		fprintf(stderr, " - %s", val);
	}

	// data points
	printf("<g class='data-points' style='stroke:none;fill:black;'>");
	for (int i = 0; i < SVG::Graph_N; i++) {
		vec2 p = Data[i] * Sc;
		printf("<circle cx='%lg' cy='%lg' r='3'/>", p.x, SVG::Chart_H - p.y);
	}
	printf("</g>\n");

}


int main(int argc, char** argv) {
	// output format: SVG
	freopen(argv[1], "wb", stdout);
	printf("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>\n", SVG::Width, SVG::Height);
	printf("<defs>\n\
<marker id='anchor-start' viewBox='0 0 10 10' refX='5' refY='5' orient='' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
<rect x='3.8' y='3.8' width='2.4' height='2.4' style='stroke:black;stroke-width:1px;fill:black'></rect></marker>\n\
<marker id='anchor-end' viewBox='0 0 10 10' refX='5' refY='5' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
<ellipse cx='5' cy='5' rx='1.2' ry='1.2' style='stroke:black;stroke-width:1px;fill:black'></ellipse></marker>\n\
<clipPath id='viewbox'><rect x='%lg' y='%lg' width='%lg' height='%lg' /></clipPath>\n\
<clipPath id='chartbox'><rect x='%lg' y='%lg' width='%lg' height='%lg' /></clipPath>\n\
</defs>\n", -SVG::Bound.x, -SVG::Bound.y, 2.*SVG::Bound.x, 2.*SVG::Bound.y,
-SVG::Chart_Margin.x, -SVG::Chart_Margin.y, SVG::Chart_W + 2 * SVG::Chart_Margin.x, SVG::Chart_H + 2 * SVG::Chart_Margin.y);
	printf("<style>text{font-size:13px;font-family:Arial;white-space:pre-wrap;}.axes{stroke:gray;}.anchors{stroke:black;opacity:0.4;}.segpath{stroke-width:3px;stroke:#ccc;}</style>\n");


	// computing cost statistics
	struct statistic {
		int id;  // test case id
		double time;  // time used
		int curve_n;  // number of pieces in the final result
		uint32_t sample;  // number of samples
		uint32_t dist_call;  // number of Bezier curve distance calculations
		double error;  // estimated error
	};
	std::vector<statistic> stats;
	uint32_t distCubic2_callCount_max = 0;


	auto start_time = NTime::now();

	for (int i = CS0; i < CS1; i++) {
		int px = ((i - CS0) % SVG::colspan)*SVG::W, py = ((i - CS0) / SVG::colspan)*SVG::H;
		printf("\n<!-- Test Case #%d -->\n", i);
		printf("<rect x='%d' y='%d' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", px, py, SVG::W, SVG::H);
		printf("<g transform='matrix(%lg,0,0,%lg,%lg,%lg)' clip-path='url(#viewbox)' style='stroke-width:%lgpx;stroke:black;fill:none;'>\n",
			SVG::SC, -SVG::SC, px + .5*SVG::W, py + .5*SVG::H, 1.0 * SVG::invSC);
		fprintf(stderr, "Case %3d - ", i);

		int spn = 0;
		double err = abs(NAN);
		double time_elapsed = 0.;
		distCubic2_callCount = Parametric_callCount = 0;
		drawFittedCurve(Cs[i], spn, err, time_elapsed, i);

		fprintf(stderr, "%.1lfms\n", 1000 * time_elapsed);
		printf("</g>\n");
		if (1) {
			printf("<text x='%d' y='%d'>#%d  %d %s   Err: %lf%s   %.3lgsecs</text>\n", px + 10, py + 20, i
				, spn, spn > 1 ? "pieces" : "piece", err, isnan(err) ? "" : "est.", time_elapsed);
			printf("<text x='%d' y='%d' data-db='%d' data-pc='%d'>DB %.1lfk    PE %.1lfk</text>\n", px + 10, py + 40,
				distCubic2_callCount, Parametric_callCount, .001*distCubic2_callCount, .001*Parametric_callCount);
		}
		stats.push_back(statistic{ i, time_elapsed, spn, Parametric_callCount, distCubic2_callCount, err });
		distCubic2_callCount_max = max(distCubic2_callCount_max, distCubic2_callCount);
		fflush(stdout);
	}

	double time_elapsed = fsec(NTime::now() - start_time).count();
	fprintf(stderr, "\nTotal %lfsecs\n\n", time_elapsed);


	// charts, somewhat messy
	if (SVG::drawChart) {
		vec2 ChartData[SVG::Graph_N];

#define drawIthChart(i, title, var1, var2, ...) \
	do { \
		printf("\n<!-- %s -->\n", title); \
		fprintf(stderr, title); \
		printf("<g transform='translate(%lg %lg)' clip-path='url(#chartbox)'>\n", SVG::Chart_Margin.x, SVG::Height_F + i * SVG::Chart_H + (2 * i + 1) * SVG::Chart_Margin.y); \
		for (int u = 0; u < SVG::Graph_N; u++) ChartData[u] = vec2(stats[u].##var1, stats[u].##var2); \
		drawChart(ChartData, __VA_ARGS__); \
		printf("</g>\n"); \
		fprintf(stderr, "\n"); \
	} while(0)

		drawIthChart(0, "ID-Time graph", id, time*1000.,
			"Test case ID", "Computing Time", vec2(CS1 <= 20 ? 1 : CS1 < 100 ? 4 : 8, 0), "", "ms", false);
		drawIthChart(1, "Piece-Time graph", curve_n, time*1000.,
			"Number of Final Curve Pieces", "Computing Time", vec2(0), "", "ms");
		drawIthChart(2, "Sample-Time graph", sample, time*1000.,
			"Parametric-callCount", "Computing Time", vec2(0), "", "ms");
		drawIthChart(3, "DistCall-Time graph", dist_call, time*1000.,
			"distCubic2-callCount", "Computing Time", vec2(0), "", "ms");
		drawIthChart(4, "Piece-Sample graph", curve_n, sample,
			"Number of Final Curve Pieces", "Parametric-callCount", vec2(0), "", "");
		drawIthChart(5, "Piece-DistCall graph", curve_n, dist_call,
			"Number of Final Curve Pieces", "distCubic2-callCount", vec2(0), "", "");
		drawIthChart(6, "Sample-DistCall graph", sample, dist_call,
			"Parametric-callCount", "distCubic2-callCount", vec2(0), "", "");

#undef drawIthChart
	}


	printf("\n</svg>");

	fclose(stdout);
	return 0;
}
