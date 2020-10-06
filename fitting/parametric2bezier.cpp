#include <stdio.h>
#include <functional>
#include <vector>

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



struct cubicBezier {
	vec2 A, B, C, D;
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
template<typename ParamCurve>
double fitPartCurve(ParamCurve C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1, vec2 &uv,
	double *lengthC = nullptr) {
	int callCount = 0;  // test

	// pre-computing for numerical integral
	double dt = t1 - t0, eps = 1e-6*dt;
	vec2 P[32]; double dL[32];
	for (int i = 0; i < 32; i++) {
		double t = t0 + NIntegrate_GL32_S[i] * dt;
		P[i] = C.p(t);
		dL[i] = length(C.p(t + eps) - C.p(t - eps)) / (2.*eps);
		dL[i] *= NIntegrate_GL32_W[i] * dt;
	}

	// not a good loss function
	// an idea for improvement is integrating the area between the curves
	auto cost = [&](double u, double v) {
		callCount++;
		cubicCurve bc = bezierAlg(P0, P0 + T0 * u, P1 - T1 * v, P1);
		double s(0.);
		for (int i = 0; i < 32; i++)
			s += distCubic2(bc, P[i]) * dL[i];
		return s;
	};

	double clength = calcLength(C.p, t0, t1, 48);
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
template<typename ParamCurve>
std::vector<cubicBezier> fitSpline(ParamCurve C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1,
	double allowed_err, double* Err = nullptr, double *LengthC = nullptr) {

	std::vector<cubicBezier> res;
	if (!(t1 > t0)) return res;
	double clength = calcLength(C.p, t0, t1, 48);

	// try curve fitting
	vec2 uv((C.t1 - C.t0) / 3.);
	double err = fitPartCurve(C, P0, P1, T0, T1, t0, t1, uv);
	double aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	// success
	if (aerr < allowed_err) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 });
		if (Err) *Err = err;
		if (LengthC) *LengthC = clength;
		return res;
	}

	// otherwise: try splitting into multiple curves
	const double eps = 1e-6*(t1 - t0);

	// try splitting at the center
	double tc = 0.5*(t0 + t1);
	vec2 Pc = C.p(tc);
	vec2 Tc = (C.p(tc + eps) - C.p(tc - eps))*(.5 / eps);
	vec2 uv0 = uv * .5, uv1 = uv * .5;
	double cl0 = 0, cl1 = 0;
	err = fitPartCurve(C, P0, Pc, T0, Tc, t0, tc, uv0, &cl0)
		+ fitPartCurve(C, Pc, P1, Tc, T1, tc, t1, uv1, &cl1);
	clength = cl0 + cl1;
	aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	if (aerr < allowed_err) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uv0.x, Pc - Tc * uv0.y, Pc });
		res.push_back(cubicBezier{ Pc, Pc + Tc * uv1.x, P1 - T1 * uv1.y, P1 });
		if (Err) *Err = err;
		if (LengthC) *LengthC = clength;
		return res;
	}

	// try splitting into three arcs
	double ta = (2.*t0 + t1) / 3., tb = (t0 + 2.*t1) / 3.;
	vec2 Pa = C.p(ta), Pb = C.p(tb);
	vec2 Ta = (C.p(ta + eps) - C.p(ta - eps))*(.5 / eps);
	vec2 Tb = (C.p(tb + eps) - C.p(tb - eps))*(.5 / eps);
	vec2 uva = uv / 3., uvc = uv / 3., uvb = uv / 3.;
	double cla, clc, clb;
	err = fitPartCurve(C, P0, Pa, T0, Ta, t0, ta, uva, &cla)
		+ fitPartCurve(C, Pa, Pb, Ta, Tb, ta, tb, uvc, &clc)
		+ fitPartCurve(C, Pb, P1, Tb, T1, tb, t1, uvb, &clb);
	clength = cla + clc + clb;
	aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	if (aerr < allowed_err) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uva.x, Pa - Ta * uva.y, Pa });
		res.push_back(cubicBezier{ Pa, Pa + Ta * uvc.x, Pb - Tb * uvc.y, Pb });
		res.push_back(cubicBezier{ Pb, Pb + Tb * uvb.x, P1 - T1 * uvb.y, P1 });
		if (Err) *Err = err;
		if (LengthC) *LengthC = clength;
		return res;
	}

	// split recursively
	double err0, err1;
	std::vector<cubicBezier> res0 = fitSpline(C, P0, Pc, T0, Tc, t0, tc, allowed_err, &err0, &cl0);
	std::vector<cubicBezier> res1 = fitSpline(C, Pc, P1, Tc, T1, tc, t1, allowed_err, &err1, &cl1);
	clength = cl0 + cl1;
	err = err0 + err1, aerr = sqrt(err / clength);
	printerr("%lf\n", aerr);
	res = res0; res.insert(res.end(), res1.begin(), res1.end());
	if (Err) *Err = err;
	if (LengthC) *LengthC = clength;
	return res;
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
const int CSN = 96;  // number of test functions
const int CS0 = 0, CS1 = CSN;  // only test a range of functions
#define _disabled ParametricCurveL([](double _){return vec2(0.);},0.,0.),/##/ // causes infinite loop/stack overflow
//#define _disabled /##*override*##/
const ParametricCurveL Cs[CSN] = {
#pragma region General Tests (42)
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
#pragma endregion  $ 0-42
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
#pragma endregion  $ 42-76
#pragma region Complicated (20)
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
ParametricCurveL([](double x) { _return vec2(x, sin(x) + sin(10.*x) + sin(100.*x))*.4; }, -5., 5.),
#pragma endregion  $ 76-96
};



// timer
#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

// scaling coefficients as global constants
const int W = 600, H = 400;
const double SC = 120.0;

void drawVectorizeCurve(const ParametricCurveL &C, int &spn, double &err, double &time_elapsed) {

	// axis
	{
		printf("<g class='axes'>");
		printf("<line x1='%lg' x2='%lg' y1='0' y2='0'/><line x1='1' x2='1' y1='-.05' y2='.05'/>", -0.5*W / SC, 0.5*W / SC);
		printf("<line x1='0' x2='0' y1='%lg' y2='%lg'/><line x1='-.05' x2='.05' y1='1' y2='1'/>", -0.5*H / SC, 0.5*H / SC);
		printf("</g>\n");
	}

	// discretized path
	if (0) {
		printf("<path class='segpath' d='");
		const int D = 1024; double dt = (C.t1 - C.t0) / D;
		vec2 p = C.p(C.t0);
		bool started = !isnan(0.*p.sqr());
		if (started) printf("M%lg,%lg", p.x, p.y);
		for (int i = 1; i <= D; i++) {
			p = C.p(C.t0 + i * dt);
			if (!isnan(0.*p.sqr())) {
				printf("%c%lg,%lg", started ? 'L' : 'M', p.x, p.y);
				started = true;
			}
			else started = false;
		}
		printf("' style='stroke-width:3px;stroke:#ccc;' vector-effect='non-scaling-stroke'/>\n");
		fflush(stdout); return;
	}

	// vectorized path
	{
		printf("<path class='vecpath' d='");
		vec2 P0 = C.p(C.t0);
		vec2 P1 = C.p(C.t1);
		vec2 uv((C.t1 - C.t0) / 3.);
		const double eps = 0.001;
		vec2 T0 = (C.p(C.t0 + eps) - C.p(C.t0)) / eps;
		vec2 T1 = (C.p(C.t1) - C.p(C.t1 - eps)) / eps;

		auto time0 = NTime::now();
		double clength;
		std::vector<cubicBezier> sp = fitSpline(C, P0, P1, T0, T1, C.t0, C.t1, 0.001, &err, &clength);
		err = sqrt(err / clength);
		time_elapsed = fsec(NTime::now() - time0).count();
		spn = sp.size();

		printf("M%lg,%lg\n", P0.x, P0.y);
		for (int i = 0; i < spn; i++) {
			vec2 Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
			printf("C%lg,%lg %lg,%lg %lg,%lg\n", Q0.x, Q0.y, Q1.x, Q1.y, P1.x, P1.y);
		}
		printf("' stroke-width='1' vector-effect='non-scaling-stroke'/>\n");

		// anchor points
		if (spn && spn < 100) {
			printf("<g class='anchors' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)'>\n");
			auto line = [](vec2 a, vec2 b) {
				printf("<line x1='%lg' y1='%lg' x2='%lg' y2='%lg'/>\n", a.x, a.y, b.x, b.y);
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
	printf("<style>text{font-size:13px;font-family:Arial;white-space:pre-wrap;}.anchors{stroke:black;opacity:0.4;}</style>\n");

	for (int i = CS0; i < CS1; i++) {
		int px = ((i - CS0) % 2)*W, py = ((i - CS0) / 2)*H;
		printf("<!-- Path #%d -->\n", i);
		printf("<rect x='%d' y='%d' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", px, py, W, H);
		printf("<g transform='matrix(%lg,0,0,%lg,%lg,%lg)' clip-path='url(#viewbox)' style='stroke-width:%lgpx;stroke:black;fill:none;'>\n", SC, -SC, px + .5*W, py + .5*H, 1.0 / SC);
		fprintf(stderr, "#%d\n", i);
		int spn; double err = abs(NAN), time_elapsed = 0.;
		distCubic2_callCount = 0, Parametric_callCount = 0;
		drawVectorizeCurve(Cs[i], spn, err, time_elapsed);
		printf("</g>\n");
		printf("<text x='%d' y='%d'>#%d  %d %s   Err: %lf%s   %.3lgsecs</text>\n", px + 10, py + 20, i
			, spn, spn > 1 ? "pieces" : "piece", err, isnan(err) ? "" : "est.", time_elapsed);
		printf("<text x='%d' y='%d'>DB %.1lfk    PE %.1lfk</text>\n", px + 10, py + 40, .001*distCubic2_callCount, .001*Parametric_callCount);
	}

	printf("</svg>");

	return 0;
}
