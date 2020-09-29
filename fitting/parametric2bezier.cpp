#include <stdio.h>
#include <functional>
#include <vector>

#include "numerical/geometry.h"
#include "numerical/integration.h"
#include "numerical/optimization.h"
#include "numerical/rootfinding.h"

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
};
typedef ParametricCurve<vec2(*)(double)> ParametricCurveP;
typedef ParametricCurve<std::function<vec2(double)>> ParametricCurveL;


// give the derivative of the parametric curve, calculate its length
template<typename ParamCurve>
double calcLength_d(const ParamCurve &dC, int N) {
	return NIntegrate_Simpson<double>([&](double t) {
		return length(dC.p(t));
	}, dC.t0, dC.t1, N);
}

// give a parametric curve, calculate its length
template<typename Fun>
double calcLength(const Fun &C, double t0, double t1, int N) {
	return NIntegrate_AL_Simpson_p<double, vec2>(
		[](vec2 p) {return 1.; },
		C, t0, t1, N);
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
double distCubic2(cubicCurve c, vec2 p) {
	vec2 c0 = c.c0 - p, c1 = c.c1, c2 = c.c2, c3 = c.c3;
#if 0
	// naive discretization
	const int N = 30;
	double dt = 1.0 / N;
	double md = c0.sqr();
	vec2 a = c0, b, ab;
	for (int i = 1; i <= N; i++) {
		double t = i * dt;
		vec2 b = c0 + t * (c1 + t * (c2 + t * c3));
		vec2 ab = b - a;
		double h = -dot(ab, a) / ab.sqr();
		if (h > 0) {
			h = min(h, 1.);
			double d = (a + h * ab).sqr();
			if (d < md) md = d;
		}
		a = b;
	}
	return md;
#else
	// exact root finding - faster!
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
	int NR = solveQuintic_bisect(k, R, 0, 1, 1e-6);  // >5x faster than solvePolynomial_bisect
	for (int i = 0; i < NR; i++) {
		double t = R[i];
		vec2 b = c0 + t * (c1 + t * (c2 + t * c3));
		md = min(md, b.sqr());
	}
	return md;
#endif
};

// loss function for curve fitting
template<typename Fun>
double loss(Fun S, vec2 A, vec2 B, vec2 C, vec2 D, double t0, double t1) {
	cubicCurve bc = bezierAlg(A, B, C, D);
	return NIntegrate_AL_Simpson_p<double, vec2>([&](vec2 p) {
		return distCubic2(bc, p);
	}, S, t0, t1, 48);
	// not a good loss function
	// an idea of improvement is integrating the area between the curves
}


// curve fitting, return loss
template<typename ParamCurve>
double fitPartCurve(ParamCurve C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1, vec2 &uv) {
	int callCount = 0;  // test
	auto cost = [&](double u, double v) {
		callCount++;
		vec2 P = P0 + T0 * u, Q = P1 - T1 * v;
		return loss(C.p, P0, P, Q, P1, t0, t1);
	};

#if 0
	uv = Newton_Iteration_2d_([&](vec2 uv) {
		return cost(uv.x, uv.y);
	}, uv);
#else
	double clength = calcLength(C.p, t0, t1, 48);
	vec2 UV0[3] = {
		uv,
		uv * vec2(1.1, 1.),
		uv * vec2(1., 1.1)
	};
	uv = downhillSimplex_2d([&](vec2 uv) {
		double penalty = (uv.x < 0. ? uv.x*uv.x : 0.) + (uv.y < 0. ? uv.y*uv.y : 0.);
		cubicCurve C = bezierAlg(P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1);
		double blength = calcLength([&](double t) { return C.c0 + t * (C.c1 + t * (C.c2 + t * C.c3)); }, 0., 1., 48);
		double penaltyl = (blength - clength)*(blength - clength);
		return cost(uv.x, uv.y) + 1.0 * penalty + penaltyl;
	}, UV0, 1e-8);
#endif

	printerr("%d\n", callCount);

	return cost(uv.x, uv.y);
}
// P0, P1: starting and ending points; T0, T1: tangent vector (derivative)
// Error is calculated as the average absolute difference per unit arc length (consider using maximum error instead)
template<typename ParamCurve>
std::vector<cubicBezier> fitSpline(ParamCurve C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1,
	double allowed_err, double* Err = nullptr) {

	std::vector<cubicBezier> res;
	double clength = calcLength(C.p, t0, t1, 48);

	// try curve fitting
	vec2 uv((C.t1 - C.t0) / 3.);
	double err = fitPartCurve(C, P0, P1, T0, T1, t0, t1, uv);
	err = sqrt(err / clength);
	printerr("%lf\n", err);
	// success
	if (err < allowed_err) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 });
		if (Err) *Err = err;
		return res;
	}

	// otherwise: try splitting into multiple curves
	const double eps = 0.001*(t1 - t0);

	// try splitting at the center
	double tc = 0.5*(t0 + t1);
	vec2 Pc = C.p(tc);
	vec2 Tc = (C.p(tc + eps) - C.p(tc - eps))*(.5 / eps);
	vec2 uv0 = uv * .5, uv1 = uv * .5;
	err = fitPartCurve(C, P0, Pc, T0, Tc, t0, tc, uv0)
		+ fitPartCurve(C, Pc, P1, Tc, T1, tc, t1, uv1);
	err = sqrt(err / clength);
	printerr("%lf\n", err);
	if (err < allowed_err) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uv0.x, Pc - Tc * uv0.y, Pc });
		res.push_back(cubicBezier{ Pc, Pc + Tc * uv1.x, P1 - T1 * uv1.y, P1 });
		if (Err) *Err = err;
		return res;
	}

	// try splitting into three arcs
	double ta = (2.*t0 + t1) / 3., tb = (t0 + 2.*t1) / 3.;
	vec2 Pa = C.p(ta), Pb = C.p(tb);
	vec2 Ta = (C.p(ta + eps) - C.p(ta - eps))*(.5 / eps);
	vec2 Tb = (C.p(tb + eps) - C.p(tb - eps))*(.5 / eps);
	vec2 uva = uv / 3., uvc = uv / 3., uvb = uv / 3.;
	err = fitPartCurve(C, P0, Pa, T0, Ta, t0, ta, uva)
		+ fitPartCurve(C, Pa, Pb, Ta, Tb, ta, tb, uvc)
		+ fitPartCurve(C, Pb, P1, Tb, T1, tb, t1, uvb);
	err = sqrt(err / clength);
	printerr("%lf\n", err);
	if (err < allowed_err) {
		res.push_back(cubicBezier{ P0, P0 + T0 * uva.x, Pa - Ta * uva.y, Pa });
		res.push_back(cubicBezier{ Pa, Pa + Ta * uvc.x, Pb - Tb * uvc.y, Pb });
		res.push_back(cubicBezier{ Pb, Pb + Tb * uvb.x, P1 - T1 * uvb.y, P1 });
		if (Err) *Err = err;
		return res;
	}

	// split recursively
	double err0, err1;
	double cl0 = calcLength(C.p, t0, tc, 48);
	double cl1 = calcLength(C.p, tc, t1, 48);
	std::vector<cubicBezier> res0 = fitSpline(C, P0, Pc, T0, Tc, t0, tc, allowed_err, &err0);
	std::vector<cubicBezier> res1 = fitSpline(C, Pc, P1, Tc, T1, tc, t1, allowed_err, &err1);
	err = sqrt((err0*err0*cl0 + err1 * err1 * cl1) / clength);
	printerr("%lf\n", err);
	res = res0; res.insert(res.end(), res1.begin(), res1.end());
	if (Err) *Err = err;
	return res;
}




// Test equations - some from Wikipedia
//  - contains undefined points (0/0=c)
//  - contains infinite discontinuities (c/0=inf)
//  - contains jump discontinuities
//  - continuous but non-differentiable at certain points
//  - contains infinite-curvature points
//  - complicated (requires more numerical integration samples)
//  - NAN outside the parametric interval (requires an integrator than does not sample outside the interval)
//  - noisy or costy to evaluate
//  - shortcuts for periodic/symmetric functions
//  - analytical derivative is known
const int CSN = 64;
const ParametricCurveL Cs[CSN] = {
ParametricCurveL([](double t) { return vec2(sin(t), cos(t) + .5*sin(t)); }, -PI, PI),
ParametricCurveL([](double t) { return vec2(sin(t), 0.5*sin(2.*t)); }, -0.1, 2.*PI - 0.1),
ParametricCurveL([](double t) { return vec2(sin(t), cos(t))*cos(2 * t); }, 0, 2.*PI),
ParametricCurveL([](double t) { return vec2(sin(t),cos(t))*cos(3.*t); }, 0, PI),
ParametricCurveL([](double t) { return vec2(cos(t), sin(t))*sin(5.*t); }, 0, PI),
ParametricCurveL([](double t) { return vec2(cos(t), sin(t))*sin(6.*t); }, 0, 2.*PI),
ParametricCurveL([](double x) { return vec2(x, exp(-x * x)); }, -1., 2.),
ParametricCurveL([](double t) { return vec2(sinh(t), cosh(t) - 1.); }, -1., 1.4),
ParametricCurveL([](double x) { return vec2(x, sin(5.*x)); }, -2, 2),
ParametricCurveL([](double x) { return vec2(x, x == 0. ? 1. : sin(2.*PI*x) / (2.*PI*x)); }, -2, 2),
ParametricCurveL([](double t) { return vec2(cos(t) + .5*cos(2.*t), sin(t) + .5*sin(2.*t)); }, 0, 2.*PI),  // too many curves
ParametricCurveL([](double t) { return vec2(cos(2.*t), sin(2.*t))*sin(t); }, 0, PI),
ParametricCurveL([](double t) { return vec2(cos(2.*t), sin(2.*t))*sin(t); }, -1, 2 * PI - 1),
ParametricCurveL([](double t) { return vec2(cos(t) + cos(2.*t), sin(t) + sin(2.*t))*.5; }, 0, 2.*PI),
ParametricCurveL([](double t) { return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, 0, 2.*PI),
ParametricCurveL([](double t) { return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, -.5*PI, 1.5*PI),
ParametricCurveL([](double t) { return vec2(cos(3.*t), sin(2.*t)); }, -PI + 1., PI + 1.),
ParametricCurveL([](double t) { return vec2(cos(5.*t + PI / 4.), sin(4.*t)); }, 1., 2.*PI + 1.),
ParametricCurveL([](double x) { return vec2(x, log(x + 1)); }, -0.99, 2.),  // too many curves, one may say it fails
ParametricCurveL([](double x) { return vec2(0.5*x - 1., 0.1*tgamma(x) - 1.); }, 0.05, 5),  // too many curves
ParametricCurveL([](double x) { return vec2(x, x*x*x - x); }, -2, 2),
ParametricCurveL([](double x) { return vec2(.5*x, 0.04*(x*x*x*x + 2.*x*x*x - 6.*x*x - x + 1.)); }, -4., 4.),
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.08*a; }, 0, 6.*PI),
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a); }, -PI, 4.*PI),
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a)*(1. - .2*exp(sin(10.*a))); }, -PI, 4.*PI),  // too slow
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(sin(10.*a) + 1.2); }, -PI, 4.*PI),  // too slow
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(pow(sin(10.*a), 10.) + 1.); }, -PI, 4.*PI),  // too slow
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.04*exp(0.25*a)*(-exp(10.*(sin(20.*a) - 1)) + 1.8); }, -PI, 4.*PI),  // too slow and too many curves
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.06*exp(0.25*a)*(pow(0.6*asin(sin(10.*a)) - .05, 8.) + 0.8); }, -PI, 4.*PI),  // too slow and too many curves
ParametricCurveL([](double a) { return vec2(cos(a), sin(a)) * 0.08*exp(0.25*a)*(0.1*(asin(sin(10.*a)) + asin(sin(30.*a))) + 1.); }, -PI, 4.*PI),  // too slow and too many curves; contains non-differentiable points
ParametricCurveL([](double t) { return vec2(.1*t + .3*cos(t), sin(t)); }, -13., 14.),
ParametricCurveL([](double t) { return 0.4*vec2(cos(1.5*t), sin(1.5*t)) + vec2(cos(t), -sin(t)); }, 0, 4.*PI),
ParametricCurveL([](double a) { return (sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 2.*PI),
ParametricCurveL([](double a) { return (sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 3.*PI),
ParametricCurveL([](double a) { return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, 0, 2.*PI),
ParametricCurveL([](double a) { return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, -1, 2.*PI - 1.),
ParametricCurveL([](double x) { return vec2(x, exp(sin(x)) - 1.5); }, -2, 2),
ParametricCurveL([](double x) { return vec2(x, exp(sin(PI*x)) - 1.5); }, -2, 2),
ParametricCurveL([](double x) { return vec2(x, sin(sin(5.*x))); }, -2, 2),
ParametricCurveL([](double x) { return vec2(x, sin(10.*x*x)); }, -2, 2),
ParametricCurveL([](double x) { return vec2(x, sin(10.*sqrt(x + 2.))); }, -2, 2),  // too many curves, fail
ParametricCurveL([](double x) { return vec2(x, .5*acos(cos(5.*x)) - .25*PI); }, -2, 2),  // too many curves, fail
ParametricCurveL([](double x) { return vec2(x, abs(x - .123) - 1.); }, -2, 2),  // too many curves
ParametricCurveL([](double x) { return vec2(x, x - floor(x)); }, -2, 2),  // failed; contains jump discontinuities
ParametricCurveL([](double t) { return vec2(.1*floor(10.*t + 1.), sin(2.*PI*t)*(10.*t - floor(10.*t))); }, -2., 2.),  // failed due to jump discontinuities
//ParametricCurveL([](double t) { return vec2(tan(t), 1. / tan(t)); }, 0, PI),  // INFINITE RECURSION
ParametricCurveL([](double x) { return vec2(x, 0.1*tan(x)); }, -.499*PI, .499*PI),  // too many curves, fail
ParametricCurveL([](double x) { return vec2(x, sqrt(1. - x * x)); }, -1., 1.),  // obviously fails
ParametricCurveL([](double x) { return vec2(x, asin(x)*(2. / PI)); }, -1., 1.),  // obviously fails
ParametricCurveL([](double t) { return vec2(cos(t) + .1*cos(10.*t), sin(t) + .1*sin(10.*t)); }, 0, 2.*PI),
ParametricCurveL([](double x) { return vec2(x, x*x - cos(10.*x) - 1.)*.5; }, -1.8, 2.),
ParametricCurveL([](double t) { return vec2(cos(4.*t) + sin(t), cos(3.*t) + .7*sin(5.*t))*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { return vec2(-.7*cos(5.*t) + sin(t), cos(3.*t) + .7*sin(5.*t))*.8; }, 0., 2.*PI),
ParametricCurveL([](double t) { return vec2(1.5*cos(t) + cos(1.5*t), 1.5*sin(t) - sin(1.5*t))*.5; }, 0., 4.*PI),
ParametricCurveL([](double t) { return vec2(2.1*cos(t) + cos(2.1*t), 2.1*sin(t) - sin(2.1*t))*.5; }, 0., 10.*PI),
ParametricCurveL([](double t) { return vec2(-1.2*cos(t) + cos(1.2*t), -1.2*sin(t) + sin(1.2*t))*.5; }, 0., 10.*PI),
ParametricCurveL([](double t) { return vec2(-1.9*cos(t) + cos(1.9*t), -1.9*sin(t) + sin(1.9*t))*.5; }, 0., 20.*PI),  // too slow
ParametricCurveL([](double t) { return vec2(-sin(t) - .3*cos(t), .1*sin(t) - .5*cos(t))*sin(5.*t) + vec2(0., 1. - .5*pow(sin(5.*t) - 1., 2.)); }, 0, 2.*PI),
ParametricCurveL([](double t) { return vec2(sin(t) + .2*cos(30.*t)*sin(t), -.4*cos(t) - .1*cos(30.*t)*cos(t) + .2*sin(30.*t)); }, 0, 2.*PI),  // too slow
ParametricCurveL([](double t) { return vec2(cos(t) - pow(cos(40.*t), 3.), sin(40.*t) - pow(sin(t), 4.) + .5)*.8; }, 0., 2.*PI),  // tooo slow
ParametricCurveL([](double t) { return vec2(cos(60.*t) - 1.6*pow(cos(t), 3.), sin(60.*t) - pow(sin(t), 3.))*.6; }, 0., 2.*PI),  // tooo slow
ParametricCurveL([](double t) { return vec2(cos(t) - cos(t)*sin(60.*t), 2.*sin(t) - sin(60.*t))*.5; }, 0., 2.*PI),  // tooo slow
ParametricCurveL([](double t) { return vec2(cos(80.*t) - 1.4*cos(t)*sin(2.*t), 2.*sin(t) - sin(80.*t))*.5; }, 0., 2.*PI),  // tooo slow
ParametricCurveL([](double a) { return 0.3*(exp(sin(a)) - 2.*cos(4.*a) + sin((2.*a - PI) / 24.))*vec2(cos(a), sin(a)); }, -8.*PI, 8.*PI),  // too slow
ParametricCurveL([](double t) { return vec2(.04041 + .6156*cos(t) - .3412*sin(t) + .1344*cos(2.*t) - .1224*sin(2.*t) + .08335*cos(3.*t) + .2634*sin(3.*t) - .07623*cos(4.*t) - .09188*sin(4.*t) + .01339*cos(5.*t) - .01866*sin(5.*t) + .1631*cos(6.*t) + .006984*sin(6.*t) + .02867*cos(7.*t) - .01512*sin(7.*t) + .00989*cos(8.*t) + .02405*sin(8.*t) + .002186*cos(9.*t), +.04205 + .2141*cos(t) + .4436*sin(t) + .1148*cos(2.*t) - .146*sin(2.*t) - .09506*cos(3.*t) - .06217*sin(3.*t) - .0758*cos(4.*t) - .02987*sin(4.*t) + .2293*cos(5.*t) + .1629*sin(5.*t) + .005689*cos(6.*t) + .07154*sin(6.*t) - .02175*cos(7.*t) + .1169*sin(7.*t) - .01123*cos(8.*t) + .02682*sin(8.*t) - .01068*cos(9.*t)); }, 0., 2.*PI),
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
		const int D = 200; double dt = (C.t1 - C.t0) / D;
		vec2 p = C.p(C.t0);
		printf("M%lg,%lg", p.x, p.y);
		for (int i = 1; i <= D; i++) {
			p = C.p(C.t0 + i * dt);
			printf("L%lg,%lg", p.x, p.y);
		}
		printf("' style='stroke-width:3px;stroke:#ccc;' vector-effect='non-scaling-stroke'/>\n");
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
		std::vector<cubicBezier> sp = fitSpline(C, P0, P1, T0, T1, C.t0, C.t1, 0.001, &err);
		time_elapsed = fsec(NTime::now() - time0).count();
		spn = sp.size();
		fprintf(stderr, "%lf secs\n", time_elapsed);

		fprintf(stderr, "%d curves\n", sp.size());

		printf("M%lg,%lg\n", P0.x, P0.y);
		for (int i = 0, l = sp.size(); i < l; i++) {
			vec2 Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
			printf("C%lg,%lg %lg,%lg %lg,%lg\n", Q0.x, Q0.y, Q1.x, Q1.y, P1.x, P1.y);
		}
		printf("' stroke-width='1' vector-effect='non-scaling-stroke'/>\n");

		// anchor points
		{
			printf("<g class='anchors' style='stroke:black;opacity:0.4' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)'>\n");
			auto line = [](vec2 a, vec2 b) {
				printf("<line x1='%lg' y1='%lg' x2='%lg' y2='%lg' />\n", a.x, a.y, b.x, b.y);
			};
			for (int i = 0, l = sp.size(); i < l; i++) {
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
	printf("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>\n", 2 * W, (int)(((CSN + 1) / 2 + 0.2)*H));
	printf("<defs>\n\
<marker id='anchor-start' viewBox='0 0 10 10' refX='5' refY='5' orient='' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
<rect x='3.8' y='3.8' width='2.4' height='2.4' style='stroke:black;stroke-width:1px;fill:black'></rect></marker>\n\
<marker id='anchor-end' viewBox='0 0 10 10' refX='5' refY='5' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
<ellipse cx='5' cy='5' rx='1.2' ry='1.2' style='stroke:black;stroke-width:1px;fill:black'></ellipse></marker>\n\
<clipPath id='viewbox'><rect x='%lg' y='%lg' width='%lg' height='%lg' /></clipPath>\n\
</defs>\n", -.5*W / SC, -.5*H / SC, W / SC, H / SC);
	printf("<style>text{font-size:13px;font-family:Arial;white-space:pre-wrap;}</style>\n");

	for (int i = 0; i < CSN; i++) {
		int px = (i % 2)*W, py = (i / 2)*H;
		printf("<!-- Path #%d -->\n", i);
		printf("<rect x='%d' y='%d' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", px, py, W, H);
		printf("<g transform='matrix(%lg,0,0,%lg,%lg,%lg)' clip-path='url(#viewbox)' style='stroke-width:%lgpx;stroke:black;fill:none;'>\n", SC, -SC, px + .5*W, py + .5*H, 1.0 / SC);
		int spn; double err, time_elapsed;
		drawVectorizeCurve(Cs[i], spn, err, time_elapsed);
		printf("</g>\n");
		printf("<text x='%d' y='%d'>#%d  %d %s   Err: %lf%s   %.3lgsecs</text>\n", px + 10, py + 20, i
			, spn, spn > 1 ? "pieces" : "piece", err, isnan(err) ? "" : "est.", time_elapsed);
	}

	printf("</svg>");
	return 0;
}
