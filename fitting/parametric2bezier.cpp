// Curve fitting experiment - fit 2d parametric curves to cubic Bezier spline
// SVG image is printed to stdout, and program outputs to stderr

#include <stdio.h>
#include <functional>
#include <vector>
#include <string>

#include "numerical/geometry.h"
#include "numerical/integration.h"
#include "numerical/optimization.h"
#include "numerical/rootfinding.h"
#include "numerical/random.h"

#define printerr(...) fprintf(stderr, __VA_ARGS__)
#define printerr(...) {}


// Parametric curve class
template<typename Fun>
class ParametricCurve {
public:
	double t0, t1;  // parameter interval
	const Fun p;  // equation
	ParametricCurve(Fun p) :t0(NAN), t1(NAN), p(p) { }
	ParametricCurve(Fun p, double t0, double t1) :t0(t0), t1(t1), p(p) { }
	//ParametricCurve() :t0(NAN), t1(NAN) {}  // not recommended
};
typedef ParametricCurve<vec2(*)(double)> ParametricCurveP;
typedef ParametricCurve<std::function<vec2(double)>> ParametricCurveL;



inline bool isNAV(vec2 p) {
	double m = p.sqr();
	return isnan(m) || m > 1e32;
}

// give a parametric curve, calculate its length
template<typename Fun>
double calcLength(const Fun &C, double t0, double t1, int N = 48) {
	double l = NIntegrate_AL_Simpson_p<double, vec2>(
		[](vec2 p) {return 1.; },
		C, t0, t1, N);
	if (isnan(0.*l)) {  // sketchy thing
		const double eps = 1e-6*(t1 - t0);
		return NIntegrate_GL48<double>([&](double t) {
			return length(C(t + eps) - C(t - eps));
		}, t0, t1) / (2.*eps);
	}
	if (l <= 0.0) l = 1e-6;  // u what
	return l;
}

// P(t0) is NAN and P(t1) is not, find the t closest to t0 such that P(t) is not NAN
template<typename Fun>
bool bisectNAN(const Fun &P, double t0, double t1, vec2 p0, vec2 p1, double &t, vec2 &p, double eps = 1e-12) {
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



struct cubicBezier {
	vec2 A, B, C, D;
	cubicBezier toSegment(bool comfirm) { if (comfirm) B = C = vec2(NAN); return *this; }
};
struct cubicCurve {
	vec2 c0, c1, c2, c3;  // c0 + c1 t + c2 t² + c3 t³, 0 < t < 1
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
static uint32_t distCubic2_callCount = 0;  // a counter for testing
double distCubic2(cubicCurve c, vec2 p) {
	distCubic2_callCount++;
	// exact root finding - faster than naive discretization
	vec2 c0 = c.c0 - p, c1 = c.c1, c2 = c.c2, c3 = c.c3;
	vec2 p0 = c0;
	vec2 p1 = c0 + c1 + c2 + c3;
	double md = min(p0.sqr(), p1.sqr());
	double k[6];
	k[5] = 3.*c3.sqr();
	k[4] = 5.*dot(c2, c3);
	k[3] = 4.*dot(c1, c3) + 2.*c2.sqr();
	k[2] = 3.*(dot(c0, c3) + dot(c1, c2));
	k[1] = 2.*dot(c0, c2) + c1.sqr();
	k[0] = dot(c0, c1);
	double R[5];
	int NR = solveQuintic_bisect(k, R, 0, 1, 1e-6);  // the slowest part of this program
	for (int i = 0; i < NR; i++) {
		double t = R[i];
		vec2 b = c0 + t * (c1 + t * (c2 + t * c3));
		md = min(md, b.sqr());
	}
	return md;
};



// curve fitting, return loss
template<typename Fun>
double fitPartCurve(Fun C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1, vec2 &uv,
	double *lengthC = nullptr, double *sample_radius = nullptr, double lod_radius = 0.) {
	int callCount = 0;  // test

	// pre-computing for numerical integral
	double dt = t1 - t0, eps = 1e-6*dt;
	vec2 P[32]; double dL[32];
	for (int i = 0; i < 32; i++) {
		double t = t0 + NIntegrate_GL32_S[i] * dt;
		P[i] = C(t);
		dL[i] = length(C(t + eps) - C(t - eps)) / (2.*eps);
		dL[i] *= NIntegrate_GL32_W[i] * dt;
	}

	// not a good loss function
	// an idea for improvement is integrating the area between the curves
	// another idea is to break if the samples don't look good
	auto cost = [&](double u, double v) {
		callCount++;
		cubicCurve bc = bezierAlg(P0, P0 + T0 * u, P1 - T1 * v, P1);
		double s(0.);
		for (int i = 0; i < 32; i++)
			s += distCubic2(bc, P[i]) * dL[i];
		return s;
	};

#if 1
	double clength = calcLength(C, t0, t1, 48);
#else
	double clength = 0.;
	for (int i = 0; i < 32; i++) clength += dL[i];
#endif

	// calculate the "radius" of the samplings (for computing the level of details)
	if (sample_radius) {
		vec2 c(0.);
		for (int i = 0; i < 32; i++) c += P[i];
		c *= 1. / 32;
		double mr = 0.;
		for (int i = 0; i < 32; i++) mr = max(mr, (P[i] - c).sqr());
		*sample_radius = sqrt(mr);
		// so the following computations can be omitted
		if (*sample_radius < lod_radius) {
			T0 = T1 = P1 - P0;
			return cost(.3, .3);
		}
	}

	vec2 UV0[3] = {
		uv,
		uv * vec2(1.1, 1.),
		uv * vec2(1., 1.1)
	};
	uv = downhillSimplex_2d([&](vec2 uv) {
		double penalty = (uv.x < 0. ? uv.x*uv.x : 0.) + (uv.y < 0. ? uv.y*uv.y : 0.);
		cubicCurve C = bezierAlg(P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1);
		double blength = NIntegrate_GL24<double>([&](double t) { return length(C.c1 + t * (2.*C.c2 + t * 3.*C.c3)); }, 0., 1.);
		double penaltyl = (blength - clength)*(blength - clength);
		return cost(uv.x, uv.y) + 1.0 * penalty + penaltyl;
	}, UV0, 1e-8);

	printerr("%d\n", callCount);

	if (lengthC) *lengthC = clength;
	return cost(uv.x, uv.y);
}
// P0, P1: starting and ending points; T0, T1: tangent vector (derivative)
// Error is calculated as the integral of shortest distance to the bezier curve respect to arc length of C
// lod_radius: radius for the level of detail; use line segments instead of bezier curve for complicated/fractal curves (necessary)
template<typename Fun>
std::vector<cubicBezier> fitSpline(Fun C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1,
	double allowed_err, double lod_radius, double* Err = nullptr, double *LengthC = nullptr) {

	std::vector<cubicBezier> res;
	if (!(t1 > t0)) return res;
	double clength = calcLength(C, t0, t1, 48);

	// try curve fitting
	vec2 uv((t1 - t0) / 3.);
	double spr;
	double err = fitPartCurve(C, P0, P1, T0, T1, t0, t1, uv, nullptr, &spr, lod_radius);
	double aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	// success
	if (aerr < allowed_err || spr < lod_radius) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 }.toSegment(spr < lod_radius));
		if (Err) *Err = err;
		if (LengthC) *LengthC = clength;
		return res;
	}

	// otherwise: try splitting into multiple curves
	const double eps = 1e-6*(t1 - t0);

	// try splitting at the center
	double tc = 0.5*(t0 + t1);
	vec2 Pc = C(tc);
	vec2 Tc = (C(tc + eps) - C(tc - eps))*(.5 / eps);
	vec2 uv0 = uv * .5, uv1 = uv * .5;
	double cl0 = 0, cl1 = 0, spr0, spr1;
	err = fitPartCurve(C, P0, Pc, T0, Tc, t0, tc, uv0, &cl0, &spr0, lod_radius)
		+ fitPartCurve(C, Pc, P1, Tc, T1, tc, t1, uv1, &cl1, &spr1, lod_radius);
	clength = cl0 + cl1;
	aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	if (aerr < allowed_err || max(spr0, spr1) < lod_radius) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uv0.x, Pc - Tc * uv0.y, Pc }.toSegment(spr0 < lod_radius));
		res.push_back(cubicBezier{ Pc, Pc + Tc * uv1.x, P1 - T1 * uv1.y, P1 }.toSegment(spr1 < lod_radius));
		if (Err) *Err = err;
		if (LengthC) *LengthC = clength;
		return res;
	}

	// try splitting into three arcs
	double ta = (2.*t0 + t1) / 3., tb = (t0 + 2.*t1) / 3.;
	vec2 Pa = C(ta), Pb = C(tb);
	vec2 Ta = (C(ta + eps) - C(ta - eps))*(.5 / eps);
	vec2 Tb = (C(tb + eps) - C(tb - eps))*(.5 / eps);
	vec2 uva = uv / 3., uvc = uv / 3., uvb = uv / 3.;
	double cla, clc, clb, spra, sprb, sprc;
	err = fitPartCurve(C, P0, Pa, T0, Ta, t0, ta, uva, &cla, &spra, lod_radius)
		+ fitPartCurve(C, Pa, Pb, Ta, Tb, ta, tb, uvc, &clc, &sprc, lod_radius)
		+ fitPartCurve(C, Pb, P1, Tb, T1, tb, t1, uvb, &clb, &sprb, lod_radius);
	clength = cla + clc + clb;
	aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	if (aerr < allowed_err || max(max(spra, sprb), sprc) < lod_radius) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uva.x, Pa - Ta * uva.y, Pa }.toSegment(spra < lod_radius));
		res.push_back(cubicBezier{ Pa, Pa + Ta * uvc.x, Pb - Tb * uvc.y, Pb }.toSegment(sprc < lod_radius));
		res.push_back(cubicBezier{ Pb, Pb + Tb * uvb.x, P1 - T1 * uvb.y, P1 }.toSegment(sprb < lod_radius));
		if (Err) *Err = err;
		if (LengthC) *LengthC = clength;
		return res;
	}

	// split recursively
	double err0, err1;
	std::vector<cubicBezier> res0 = fitSpline(C, P0, Pc, T0, Tc, t0, tc, allowed_err, lod_radius, &err0, &cl0);
	std::vector<cubicBezier> res1 = fitSpline(C, Pc, P1, Tc, T1, tc, t1, allowed_err, lod_radius, &err1, &cl1);
	clength = cl0 + cl1;
	err = err0 + err1, aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	res = res0; res.insert(res.end(), res1.begin(), res1.end());
	if (Err) *Err = err;
	if (LengthC) *LengthC = clength;
	return res;
}



// split the curve by discontinuities, return parametric pairs [t0,t1]
template<typename Fun>
std::vector<vec2> splitDiscontinuity(const Fun &C, double t0, double t1, int mindif = 210/*2x3x5x7*/,
	vec2 Bound0 = vec2(-INFINITY), vec2 Bound1 = vec2(INFINITY)) {

	auto P = [&](double t)->vec2 {
		vec2 p = C(t);
		if (p.x<Bound0.x || p.x>Bound1.x) return vec2(NAN);
		if (p.y<Bound0.y || p.y>Bound1.y) return vec2(NAN);
		return p;
	};

	std::vector<vec2> ints;  // sample intervals
	std::vector<vec2> res;  // result intervals
	double dt = (t1 - t0) / mindif;

	// samples
	struct sample {
		double t;  // parameter, t0+id*dt
		vec2 p;  // value
	};
	std::vector<sample> Ss;
	bool hasNAN = false;
	for (int i = 0; i <= mindif; i++) {
		double t = t0 + i * dt;
		vec2 p = P(t);
		hasNAN |= isNAV(p);
		Ss.push_back(sample{ t, p });
	}

	// detect NAN
	if (hasNAN) {
		for (int i = 0; i < mindif; i++) {
			sample s0 = Ss[i], s1 = Ss[i + 1];
			if (isNAV(s0.p) && !isNAV(s1.p)) {
				bisectNAN(P, s0.t, s1.t, s0.p, s1.p, s0.t, s0.p);
			}
			else if (!isNAV(s0.p) && isNAV(s1.p)) {
				bisectNAN(P, s1.t, s0.t, s1.p, s0.p, s1.t, s1.p);
			}
			if (!isNAV(s0.p) && !isNAV(s1.p))
				ints.push_back(vec2(s0.t, s1.t));
		}
		// merge non-NAN intervals
		for (int i = 0, l = ints.size(); i < l; i++) {
			if (res.empty() || res[res.size() - 1].y != ints[i].x)
				res.push_back(ints[i]);
			else res[res.size() - 1].y = ints[i].y;
		}
	}
	else res.push_back(vec2(t0, t1));

	ints = res;

#if 0
	if (!hasNAN) {
		// calculate the derivatives from neighborhood samples
		std::vector<sample> d1, d2, d3, d4;

	}
#endif

	return res;

}



// this function does everything automatically
template<typename ParamCurve>
std::vector<cubicBezier> fitSpline_auto(ParamCurve C, vec2 B0, vec2 B1,
	double allowed_err = 0.001, double lod_radius = 0.01, double* Err = nullptr, double *LengthC = nullptr) {

	std::vector<vec2> Cs = splitDiscontinuity(C.p, C.t0, C.t1, 210, B0, B1);
	int CsN = Cs.size();

	std::vector<cubicBezier> Res;
	double cumErr = 0., cumLength = 0.;

	for (int i = 0; i < CsN; i++) {
		double t0 = Cs[i].x + 1e-12, t1 = Cs[i].y - 1e-12;
		const double eps = 0.0001;

		vec2 P0 = C.p(t0);
		vec2 P1 = C.p(t1);
		vec2 T0 = (C.p(t0 + eps) - P0) / eps;
		vec2 T1 = (P1 - C.p(t1 - eps)) / eps;

		double err = 0., clen = 0.;
		std::vector<cubicBezier> R = fitSpline(C.p, P0, P1, T0, T1, t0, t1, allowed_err, lod_radius, &err, &clen);
		Res.insert(Res.end(), R.begin(), R.end());
		cumErr += err, cumLength += clen;
	}

	if (Err) *Err = cumErr;
	if (LengthC) *LengthC = cumLength;
	return Res;
}




// this one is for reference
// should have no problem with NAN
struct segment {
	vec2 a, b;
	double d2(vec2 p) { // square of distance to a point
		vec2 ab = b - a, ap = p - a;
		double m = dot(ab, ab);
		if (m*m < 1e-12) return length(ap);
		double h = dot(ap, ab) / m;
		return (ab * h - ap).sqr();
	}
};
template<typename ParamCurve>
std::vector<segment> Param2Segments(ParamCurve C, vec2 P0, vec2 P1, double t0, double t1
	, double allowed_err, int min_dif = 17, int max_recur = 24) {

	// minimum number of pieces in initial splitting
	if (min_dif > 1) {
		std::vector<segment> res, r;
		double dt = (t1 - t0) / min_dif, _t = t0;
		vec2 _p = C(_t), p;
		for (int i = 0; i < min_dif; i++) {
			double t = t0 + (i + 1)*dt; p = C(t);
			r = Param2Segments(C, _p, p, _t, t, allowed_err, 1, max_recur);
			res.insert(res.end(), r.begin(), r.end());
			_p = p, _t = t;
		}
		return res;
	}

	// handle the cases when there are NANs at the endings
	if (isNAV(P0) && !isNAV(P1)) {
		double t_ = t0 + 1e-6*(t1 - t0), _t0 = t0;
		if (!bisectNAN(C, t0, t_, P0, C(t_), t0, P0))
			bisectNAN(C, _t0, t1, vec2(NAN), P1, t0, P0);
	}
	else if (isNAV(P1) && !isNAV(P0)) {
		double t_ = t1 - 1e-6*(t1 - t0), _t1 = t1;
		if (!bisectNAN(C, t1, t_, P1, C(t_), t1, P1))
			bisectNAN(C, _t1, t0, vec2(NAN), P0, t1, P1);
	}
	if (isNAV(P0) && isNAV(P1)) return std::vector<segment>();

	// check if the error is small enough
	segment s{ P0, P1 };
	double err = 0., d, mt = 0.5*(t0 + t1);
	vec2 p, mp = C(mt);
	bool hasNAN = isNAV(P0) || isNAV(P1);
	for (int i = 0; i < 14; i++) {
		double t = t0 + (t1 - t0) * NIntegrate_GL14_S[i];
		p = C(t), d = s.d2(p);
		if (d > err) err = d, mt = t, mp = p;
		hasNAN |= isnan(d);
	}
	//if (hasNAN) mt = 0.5*(t0 + t1), mp = C(mt);

	// if small enough, add the segment to the list
	if (max_recur < 0 || (!hasNAN && err < allowed_err*allowed_err) || (t1 - t0 < 1e-12)) {
		std::vector<segment> r;
		if (!((s.b - s.a).sqr() > 1e-16))
			return std::vector<segment>();
		r.push_back(s); return r;
	}

	// otherwise, recursively split at the point with maximum error
	// this also applies when there are occurrences of NANs inside the interval
	std::vector<segment> r0 = Param2Segments(C, P0, mp, t0, mt, allowed_err, 1, max_recur - 1);
	std::vector<segment> r1 = Param2Segments(C, mp, P1, mt, t1, allowed_err, 1, max_recur - 1);
	r0.insert(r0.end(), r1.begin(), r1.end());
	return r0;
}






// a call time counter for testing
static uint32_t Parametric_callCount = 0;
#define _return Parametric_callCount++; return

// summation
template<typename Fun> double Sum(Fun f, int n0, int n1, int step = 1) {
	double r = 0;
	for (int n = n0; n <= n1; n += step) r += f(n);
	return r;
};


// Test equations - some from Wikipedia
// Most are relatively cheap to evaluate
//  - contains undefined points (0/0=c)
//  - contains infinite discontinuities (c/0=inf)
//  - contains jump discontinuities
//  - continuous but non-differentiable at certain points
//  - contains infinite-curvature points
//  - complicated (requires more numerical integration samples)
//  - only defined inside the parametric interval
//  - shortcuts for periodic/symmetric functions
//  - analytical derivative is known
const int CSN = 114;  // number of test functions
const int CS0 = 0, CS1 = 114;  // only test a range of functions
#define _disabled  /* mark cases that cause infinite loop/stack overflow */
const std::vector<int> _disabled_list({ 59, 62, 63, 64, 65, 67, 68, 69, 70, 71 });
const ParametricCurveL Cs[CSN] = {
#pragma region General Tests (44)
ParametricCurveL([](double t) { _return vec2(sin(t), cos(t) + .5*sin(t)); }, -PI, PI),
ParametricCurveL([](double t) { _return vec2(sin(t), 0.5*sin(2.*t)); }, -0.1, 2.*PI - 0.1),
ParametricCurveL([](double t) { _return vec2(sin(t), cos(t))*cos(2 * t); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(sin(t),cos(t))*cos(3.*t); }, 0, PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*sin(5.*t); }, 0, PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*sin(6.*t); }, 0, 2.*PI),
ParametricCurveL([](double x) { _return vec2(x, exp(-x * x)); }, -1., 2.),
ParametricCurveL([](double t) { _return vec2(sinh(t), cosh(t) - 1.); }, -1., 1.4),
ParametricCurveL([](double x) { _return vec2(x, sin(5.*x)); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, x == 0. ? 1. : sin(2.*PI*x) / (2.*PI*x)); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, x*x*x - x); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(.5*x, 0.04*(x*x*x*x + 2.*x*x*x - 6.*x*x - x + 1.)); }, -4., 4.),
ParametricCurveL([](double t) { _return vec2(cos(t) + .5*cos(2.*t), sin(t) + .5*sin(2.*t)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(2.*t), sin(2.*t))*sin(t); }, 0, PI),
ParametricCurveL([](double t) { _return vec2(cos(2.*t), sin(2.*t))*sin(t); }, -1, 2 * PI - 1),
ParametricCurveL([](double t) { _return vec2(cos(t) + cos(2.*t), sin(t) + sin(2.*t))*.5; }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, -.5*PI, 1.5*PI),
ParametricCurveL([](double t) { _return vec2(cos(3.*t), sin(2.*t)); }, -PI + 1., PI + 1.),
ParametricCurveL([](double t) { _return vec2(cos(5.*t + PI / 4.), sin(4.*t)); }, 1., 2.*PI + 1.),
ParametricCurveL([](double a) { _return vec2(cos(a),sin(a)) * .8*(pow(cos(6.*a),2.) + .5); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(2.*PI*a),sin(2.*PI*a)) * pow(abs(1.2*a),3.8) + vec2(-.7,0.); }, -1., 1.),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*a; }, 0, 6.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a); }, -PI, 4.*PI),
ParametricCurveL([](double t) { _return vec2(.1*t + .3*cos(t), sin(t)); }, -13., 14.),
ParametricCurveL([](double t) { _return 0.4*vec2(cos(1.5*t), sin(1.5*t)) + vec2(cos(t), -sin(t)); }, 0, 4.*PI),
ParametricCurveL([](double a) { _return(sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return(sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 3.*PI),
ParametricCurveL([](double a) { _return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, -1, 2.*PI - 1.),
ParametricCurveL([](double x) { _return vec2(x, exp(sin(x)) - 1.5); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, exp(sin(PI*x)) - 1.5); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, sin(sin(5.*x))); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, sin(10.*x*x)); }, -2, 2),
ParametricCurveL([](double t) { _return vec2(cos(t) + .1*cos(10.*t), sin(t) + .1*sin(10.*t)); }, 0, 2.*PI),
ParametricCurveL([](double x) { _return vec2(x, x*x - cos(10.*x) - 1.)*.5; }, -1.8, 2.),
ParametricCurveL([](double t) { _return vec2(cos(4.*t) + sin(t), cos(3.*t) + .7*sin(5.*t))*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(-.7*cos(5.*t) + sin(t), cos(3.*t) + .7*sin(5.*t))*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(1.5*cos(t) + cos(1.5*t), 1.5*sin(t) - sin(1.5*t))*.5; }, 0., 4.*PI),
ParametricCurveL([](double t) { _return vec2(2.1*cos(t) + cos(2.1*t), 2.1*sin(t) - sin(2.1*t))*.5; }, 0., 10.*PI),
ParametricCurveL([](double t) { _return vec2(-1.2*cos(t) + cos(1.2*t), -1.2*sin(t) + sin(1.2*t))*.5; }, 0., 10.*PI),
ParametricCurveL([](double t) { _return vec2(-1.9*cos(t) + cos(1.9*t), -1.9*sin(t) + sin(1.9*t))*.5; }, 0., 20.*PI),
ParametricCurveL([](double t) { _return vec2(-sin(t) - .3*cos(t), .1*sin(t) - .5*cos(t))*sin(5.*t) + vec2(0., 1. - .5*pow(sin(5.*t) - 1., 2.)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(sin(t) + .2*cos(30.*t)*sin(t), -.4*cos(t) - .1*cos(30.*t)*cos(t) + .2*sin(30.*t)); }, 0, 2.*PI),
#pragma endregion  $ 0-44
#pragma region Ill-conditioned Functions (34)
ParametricCurveL([](double x) { _return vec2(x, log(x + 1)); }, -0.99, 2.),
ParametricCurveL([](double x) { _return vec2(0.5*x - 1., 0.1*tgamma(x) - 1.); }, 0.05, 5),
ParametricCurveL([](double x) { _return vec2(x, sqrt(1. - x * x)); }, -1., 1.),
ParametricCurveL([](double x) { _return vec2(x, asin(x)*(2. / PI)); }, -1., 1.),
ParametricCurveL([](double x) { _return vec2(x, abs(x - 0.123) - 1.); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, 0.1*tan(x)); }, -.499*PI, .499*PI),
ParametricCurveL([](double x) { _return vec2(x, sin(10.*sqrt(x + 2.))); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, .5*acos(cos(5.*x)) - .25*PI); }, -2, 2),
ParametricCurveL([](double x) { _return vec2(x, x - floor(x)); }, -2, 2),
ParametricCurveL([](double t) { _return vec2(.1*floor(10.*t + 1.), sin(2.*PI*t)*(10.*t - floor(10.*t))); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, sin(x - 1) / log(x + PI)) * 0.5; }, -PI, PI),
ParametricCurveL([](double x) { _return vec2(x, sqrt(x + 1.) - 1.); }, -1., 1.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(x)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(abs(x)) - 1.); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(abs(x) - 1.)); }, -2., 2.),
_disabled ParametricCurveL([](double x) { _return vec2(x, sqrt(abs(x) - 1.)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, cbrt(abs(abs(x) - 1.) - .5)); }, -2., 2.),
ParametricCurveL([](double x) { _return vec2(x, pow(abs(cbrt(abs(abs(x) - 1.) - .5)), .2) * (abs(abs(x) - 1.) > .5 ? 1. : -1.)); }, -2., 2.),
_disabled ParametricCurveL([](double x) { _return vec2(x, log(exp(abs(x))*sin(x)) - 10.)*.1; }, -20., 20.),
_disabled ParametricCurveL([](double x) { _return vec2(x, log(abs(exp(abs(x))*sin(x))) - 10.)*.1; }, -20., 20.),
_disabled ParametricCurveL([](double x) { _return vec2(x, log(tgamma(x)))*.5; }, -4., 4.),
_disabled ParametricCurveL([](double x) { _return vec2(x, log(abs(tgamma(x))))*.5; }, -4., 4.),
ParametricCurveL([](double x) { _return vec2(x, (x > 0. ? asin(sqrt(x)) / sqrt(x) : asinh(sqrt(-x)) / sqrt(-x)) - 1.); }, -2., 1.),
_disabled ParametricCurveL([](double x) { _return vec2(x, (x > 0. ? asin(sqrt(x)) / sqrt(x) : asinh(sqrt(-x)) / sqrt(-x)) - 1.); }, -2., 2.),
_disabled ParametricCurveL([](double t) { _return vec2(tan(t), 1. / tan(t)); }, -.25*PI, .75*PI),
_disabled ParametricCurveL([](double t) { _return vec2(tan(t), 1. / tan(t)); }, 0., PI),
_disabled ParametricCurveL([](double t) { _return vec2(cos(t) / sin(t), 1.)*pow(tan(t / 2.), 1.05) * 2. - vec2(0,1); }, 0., .5*PI),
_disabled ParametricCurveL([](double t) { _return vec2(cos(t) / sin(t), sin(t))*pow(tan(t / 2.), 1.02) * 2. - vec2(0,1); }, 0., .5*PI),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return sin(PI*n*x) / n; }, 1, 21, 2)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return sin(2.*PI*n*x) / (PI*n); }, 1, 20)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return sin(2.*PI*n*x) / (n*n); }, 1, 5)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return (sin(PI*n*x) - n * cos(PI*n*x)) / (n*(n*n + 1)); }, 1, 1001, 2)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { return exp(-n)*sin(exp(n)*(x + n)); }, 1, 5)); }, -2.5, 2.5),
ParametricCurveL([](double x) { _return vec2(x, Sum([&](int n) { double u = exp2(n)*x, v = u - floor(u); return exp2(-n)*mix(hashf(floor(u),-n), hashf(ceil(u),-n), v*v*(3. - 2.*v)); }, 1, 6)); }, -2.5, 2.5),
#pragma endregion  $ 44-78
#pragma region Complicated Curves (36)
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a)*(1. - .2*exp(sin(10.*a))); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(sin(10.*a) + 1.2); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(pow(sin(10.*a), 10.) + 1.); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(-exp(10.*(sin(20.*a) - 1)) + 1.8); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.06*exp(0.25*a)*(pow(0.6*asin(sin(10.*a)) - .05, 8.) + 0.8); }, -PI, 4.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a)*(0.1*(asin(sin(10.*a)) + asin(sin(30.*a))) + 1.); }, -PI, 4.*PI),
ParametricCurveL([](double t) { _return vec2(.04041 + .6156*cos(t) - .3412*sin(t) + .1344*cos(2.*t) - .1224*sin(2.*t) + .08335*cos(3.*t) + .2634*sin(3.*t) - .07623*cos(4.*t) - .09188*sin(4.*t) + .01339*cos(5.*t) - .01866*sin(5.*t) + .1631*cos(6.*t) + .006984*sin(6.*t) + .02867*cos(7.*t) - .01512*sin(7.*t) + .00989*cos(8.*t) + .02405*sin(8.*t) + .002186*cos(9.*t), +.04205 + .2141*cos(t) + .4436*sin(t) + .1148*cos(2.*t) - .146*sin(2.*t) - .09506*cos(3.*t) - .06217*sin(3.*t) - .0758*cos(4.*t) - .02987*sin(4.*t) + .2293*cos(5.*t) + .1629*sin(5.*t) + .005689*cos(6.*t) + .07154*sin(6.*t) - .02175*cos(7.*t) + .1169*sin(7.*t) - .01123*cos(8.*t) + .02682*sin(8.*t) - .01068*cos(9.*t)); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return 0.3*(exp(sin(a)) - 2.*cos(4.*a) + sin((2.*a - PI) / 24.))*vec2(cos(a), sin(a)); }, -8.*PI, 8.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) - pow(cos(40.*t), 3.), sin(40.*t) - pow(sin(t), 4.) + .5)*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(60.*t) - 1.6*pow(cos(t), 3.), sin(60.*t) - pow(sin(t), 3.))*.6; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) - cos(t)*sin(60.*t), 2.*sin(t) - sin(60.*t))*.5; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(80.*t) - 1.4*cos(t)*sin(2.*t), 2.*sin(t) - sin(80.*t))*.5; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(sin(3.*t) + sin(t),cos(3.*t))*.8 + vec2(sin(160.*t),cos(160.*t))*.4; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(3.*t),2.*cos(t))*.6 + vec2(cos(100.*t),sin(100.*t))*sin(t)*.6; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(1.8*cos(t), 0.6*cos(5.*t)) + vec2(cos(100.*t),sin(100.*t))*pow(abs(sin(t)),0.6)*0.6; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t),0.618*cos(2.*t)) + 0.618*vec2(cos(60.*t),sin(60.*t))*cos(2.*t); }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(1.12*t),sin(t)) + .5*vec2(cos(60.*t),-pow(sin(60.*t),2.)); }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t),sin(t)) + vec2(cos(60.*t),sin(60.*t))*.5; }, 0., 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t) + .2*sin(20.*t), sin(t) + .2*cos(20.*t)) * .02*t; }, 0., 20.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*.08*floor(a); }, 0., 6.*PI),
ParametricCurveL([](double x) { _return vec2(x, sin(x) + sin(10.*x) + sin(100.*x))*.4; }, -5., 5.),
ParametricCurveL([](double x) { _return vec2(x, .5*(sin(40.*x) + sin(45.*x))); }, -2.5, 2.5),
ParametricCurveL([](double a) { _return vec2(cos(a),sin(a))*(2.0*Sum([&](int n) { return exp2(-n)*sin(exp2(n)*a); }, 1, 10)); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a) + .3)*(2.0*Sum([&](int n) { return exp2(-n)*cos(exp2(n)*a); }, 1, 16)); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a),-sin(a))*1.2*(sin(a) + Sum([&](int n) { return exp2(-n)*cos(exp2(n)*a); }, 1, 10)) + vec2(0.,.6); }, 0., 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(0.1*exp(.25*a)) + vec2(.08*exp(.2*a)*Sum([&](int n) { return exp2(-n)*pow(cos(exp2(n)*a),2.); }, 1, 10)); }, 0., 12.),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .4*Sum([&](int n) { return sin(5.*n*a) / n; }, 1, 11, 2)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .3*Sum([&](int n) { return sin(5.*n*a) / n; }, 1, 10, 1)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .5*Sum([&](int n) { return sin(5.*n*n*a) / (n*n); }, 1, 3, 1)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + .4*Sum([&](int n) { return (n & 2 ? -1. : 1.)*sin(5.*n*a) / (n*n); }, 1, 21, 2)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.8 + 3.*Sum([&](int n) { return (n & 2 ? -1. : 1.)*cos(6.*n*a) / (n*n); }, 3, 21, 2)); }, 0, 2.*PI),
ParametricCurveL([](double a) { _return vec2(cos(a), sin(a))*(.9 + .15*Sum([&](int n) { return (cos(8.*n*(a + .1)) + cos(8.*n*(a - .1))) / n; }, 1, 11, 2)); }, 0, 2.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.005*exp(.25*t)*(1. + Sum([&](int n) { return pow(cos(5.*n*t), 2.) / n; }, 1, 5, 1)); }, 0, 6.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.006*exp(.25*t)*(1. + Sum([&](int n) { return pow(sin(5.*n*t), 2.) / n; }, 1, 5, 1)); }, 0, 6.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.007*exp(.25*t)*(1. + Sum([&](int n) { return pow(sin(5.*n*n*t), 2.) / (n*n); }, 1, 5, 1)); }, 0, 6.*PI),
ParametricCurveL([](double t) { _return vec2(cos(t), sin(t))*.009*exp(.25*t)*(1. + Sum([&](int n) { return pow(sin(5.*exp2(n)*t), 2.) / exp2(n); }, 1, 5, 1)); }, 0, 6.*PI),
#pragma endregion  $ 78-114
};



// timer
#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

// scaling coefficients as global constants
const int W = 600, H = 400;
const double SC = 120.0;

// double to string for printf
std::string _f_buffer[0x10000];
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
	_f_buffer[_f_buffer_n] = s;
	return &(_f_buffer[_f_buffer_n++][0]);
};


void drawVectorizeCurve(const ParametricCurveL &C, int &spn, double &err, double &time_elapsed, int id = -1) {

	_f_buffer_n = 0;

	// axis
	{
		printf("<g class='axes'>");
		printf("<line x1='%s' x2='%s' y1='0' y2='0'/><line x1='1' x2='1' y1='-.05' y2='.05'/>", _(-0.5*W / SC), _(0.5*W / SC));
		printf("<line x1='0' x2='0' y1='%s' y2='%s'/><line x1='-.05' x2='.05' y1='1' y2='1'/>", _(-0.5*H / SC), _(0.5*H / SC));
		printf("</g>\n");
	}

	// discretized path
	if (1) {
		printf("<path class='segpath' d='");
		std::vector<segment> S = Param2Segments(C.p, C.p(C.t0), C.p(C.t1), C.t0, C.t1, 0.01, 16);
		vec2 old_pos(NAN);
		spn = S.size();
		for (int i = 0; i < spn; i++) {
			segment s = S[i];
			if (!((old_pos - s.a).sqr() < 1e-6)) printf("M%s,%s", _(s.a.x), _(s.a.y));
			printf("L%s,%s", _(s.b.x), _(s.b.y));
			old_pos = s.b;
		}
		printf("' vector-effect='non-scaling-stroke'/>\n");
		fflush(stdout); //return;
	}

	// debug
	if (0) {
		std::vector<vec2> Ps = splitDiscontinuity(C.p, C.t0, C.t1, 210, vec2(-2.5, -1.7), vec2(2.5, 1.7));
		int L = Ps.size();
		for (int i = 0; i < L; i++) {
			double t0 = Ps[i].x, t1 = Ps[i].y;
			std::vector<segment> S = Param2Segments(C.p, C.p(t0), C.p(t1), t0, t1, 0.001, 16);
			printf("<path d='");
			vec2 old_pos(NAN); int spn = S.size();
			for (int i = 0; i < spn; i++) {
				segment s = S[i];
				if (!((old_pos - s.a).sqr() < 1e-6)) printf("M%s,%s", _(s.a.x), _(s.a.y));
				printf("L%s,%s", _(s.b.x), _(s.b.y));
				old_pos = s.b;
			}
			printf("' style='stroke:%s;stroke-width:1px;fill:none' vector-effect='non-scaling-stroke'/>\n", i & 1 ? "#00F" : "#F00");
		}
		fflush(stdout); return;
	}

	//if (std::find(_disabled_list.begin(), _disabled_list.end(), id) != _disabled_list.end() \
		&& !(spn *= 0)) return;

	Parametric_callCount = 0;

	// vectorized path
	{
		printf("<path class='vecpath' d='");

		auto time0 = NTime::now();
		const vec2 RB = .5*vec2(W, H) / SC;
		double clength;
		std::vector<cubicBezier> sp = fitSpline_auto(C, -RB, RB, 0.001, 0.01, &err, &clength);
		err = sqrt(err / clength);
		time_elapsed = fsec(NTime::now() - time0).count();
		spn = sp.size();

		vec2 oldP(NAN);
		for (int i = 0; i < spn; i++) {
			vec2 P0 = sp[i].A, Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
			if (!((P0 - oldP).sqr() < 1e-12))
				printf("M%s,%s\n", _(P0.x), _(P0.y));
			if (isNAV(Q0) && isNAV(Q1)) printf("L%s,%s", _(P1.x), _(P1.y));
			else printf("C%s,%s %s,%s %s,%s", _(Q0.x), _(Q0.y), _(Q1.x), _(Q1.y), _(P1.x), _(P1.y));
			oldP = P1;
		}
		printf("' stroke-width='1' vector-effect='non-scaling-stroke'/>\n");

		// anchor points
		if (spn && spn < 100) {
			printf("<g class='anchors' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)'>\n");
			auto line = [](vec2 a, vec2 b, const char end[] = "\n") {
				if (!(isNAV(a) || isNAV(b)))
					printf("<line x1='%s' y1='%s' x2='%s' y2='%s'/>%s", _(a.x), _(a.y), _(b.x), _(b.y), end);
			};
			for (int i = 0; i < spn; i++) {
				vec2 P0 = sp[i].A, Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
				line(P0, Q0, ""); line(P1, Q1);
			}
			printf("</g>\n");
		}
	}

	fflush(stdout);
}


int main(int argc, char** argv) {
	// output format: SVG
	freopen(argv[1], "wb", stdout);
	printf("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>\n", 2 * W, (int)(((CS1 - CS0 + 1) / 2 + 0.2)*H));
	printf("<defs>\n\
<marker id='anchor-start' viewBox='0 0 10 10' refX='5' refY='5' orient='' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
<rect x='3.8' y='3.8' width='2.4' height='2.4' style='stroke:black;stroke-width:1px;fill:black'></rect></marker>\n\
<marker id='anchor-end' viewBox='0 0 10 10' refX='5' refY='5' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
<ellipse cx='5' cy='5' rx='1.2' ry='1.2' style='stroke:black;stroke-width:1px;fill:black'></ellipse></marker>\n\
<clipPath id='viewbox'><rect x='%lg' y='%lg' width='%lg' height='%lg' /></clipPath>\n\
</defs>\n", -.5*W / SC, -.5*H / SC, W / SC, H / SC);
	printf("<style>text{font-size:13px;font-family:Arial;white-space:pre-wrap;}.anchors{stroke:black;opacity:0.4;}.segpath{stroke-width:3px;stroke:#ccc;}</style>\n");

	auto start_time = NTime::now();

	for (int i = CS0; i < CS1; i++) {
		int px = ((i - CS0) % 2)*W, py = ((i - CS0) / 2)*H;
		printf("<!-- Test Path #%d -->\n", i);
		printf("<rect x='%d' y='%d' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", px, py, W, H);
		printf("<g transform='matrix(%lg,0,0,%lg,%lg,%lg)' clip-path='url(#viewbox)' style='stroke-width:%lgpx;stroke:black;fill:none;'>\n", SC, -SC, px + .5*W, py + .5*H, 1.0 / SC);
		fprintf(stderr, "Case %3d - ", i);
		int spn; double err = abs(NAN), time_elapsed = 0.;
		distCubic2_callCount = 0, Parametric_callCount = 0;
		drawVectorizeCurve(Cs[i], spn, err, time_elapsed, i);
		fprintf(stderr, "%.1lfms\n", 1000 * time_elapsed);
		printf("</g>\n");
		if (1) {
			printf("<text x='%d' y='%d'>#%d  %d %s   Err: %lf%s   %.3lgsecs</text>\n", px + 10, py + 20, i
				, spn, spn > 1 ? "pieces" : "piece", err, isnan(err) ? "" : "est.", time_elapsed);
			printf("<text x='%d' y='%d' data-db='%d' data-pc='%d'>DB %.1lfk    PE %.1lfk</text>\n", px + 10, py + 40,
				distCubic2_callCount, Parametric_callCount, .001*distCubic2_callCount, .001*Parametric_callCount);
		}
	}

	double time_elapsed = fsec(NTime::now() - start_time).count();
	fprintf(stderr, "\nTotal %lfsecs\n", time_elapsed);

	printf("</svg>");

	return 0;
}
