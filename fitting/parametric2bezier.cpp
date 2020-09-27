#include <stdio.h>
#include <functional>
#include <vector>

#include "numerical/geometry.h"
#include "numerical/integration.h"
#include "numerical/optimization.h"
#include "numerical/rootfinding.h"

#include "ui/stl_encoder.h"
stl_triangle T[0x100000];


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
	// (not optimized) exact root finding
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
	int NR = solvePolynomial_bisect(5, k, R, 0, 1, 1e-6);
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

	// debug output
	// I can see how horrible they are...
#if 0
	static int nthCalled = 0;
	if (++nthCalled == 5) {
		FILE* fp = fopen("D:\\test.stl", "wb");
		int N = stl_fun2trigs(cost, T, -10, 10, -10, 10, 200, 200);
		double maxz = 0.; for (int i = 0; i < N; i++) maxz = max(maxz, (double)std::max({ T[i].a.z, T[i].b.z, T[i].c.z }));
		double scz = 10. / maxz; for (int i = 0; i < N; i++) T[i].a.z *= scz, T[i].b.z *= scz, T[i].c.z *= scz;
		writeSTL(fp, T, N);
		fclose(fp);
		exit(0);
	}
	else return 100;
#endif

#if 0
	uv = Newton_Iteration_2d_([&](vec2 uv) {
		return cost(uv.x, uv.y);
	}, uv);
#else
	vec2 UV0[3] = {
		uv,
		uv * vec2(1.1, 1.),
		uv * vec2(1., 1.1)
	};
	uv = downhillSimplex_2d([&](vec2 uv) {
		double penalty = max(-min(uv.x, uv.y), 0.);
		return cost(uv.x, uv.y) + 1.0 * penalty*penalty;
	}, UV0, 1e-8);
#endif

	fprintf(stderr, "%d\n", callCount);

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
	fprintf(stderr, "%lf\n", err);
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
	fprintf(stderr, "%lf\n", err);
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
	fprintf(stderr, "%lf\n", err);
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
	fprintf(stderr, "%lf\n", err);
	res = res0; res.insert(res.end(), res1.begin(), res1.end());
	if (Err) *Err = err;
	return res;
}




// test equations
//  - contains undefined points (0/0=c)
//  - contains infinite discontinuities (c/0=inf)
//  - contains jump discontinuities
//  - continuous but non-differentiable at certain points
//  - contains infinite-curvature points
//  - complicated (requires more numerical integration samples)
//  - NAN outside the parametric interval (requires an integrator than does not sample outside the interval)
//  - noisy or costy to evaluate
//  - periodic
//  - analytical derivative is known
#if 1
ParametricCurveL C([](double t) { return vec2(sin(t), cos(t) + .5*sin(t)); }, -PI, PI);
//ParametricCurveL C([](double t) { return vec2(sin(t), 0.5*sin(2.*t)); }, -0.1, 2.*PI - 0.1);
//ParametricCurveL C([](double t) { return vec2(sin(t), cos(t))*cos(2 * t); }, 0, 2.*PI);
//ParametricCurveL C([](double x) { return vec2(x, exp(-x * x)); }, -1., 2.);
//ParametricCurveL C([](double t) { return vec2(sinh(t), cosh(t) - 1.); }, -1., 1.4);
//ParametricCurveL C([](double t) { return vec2(cos(t), sin(t))*sin(5.*t); }, 0, PI);
//ParametricCurveL C([](double t) { return vec2(cos(t), sin(t))*sin(6.*t); }, 0, 2.*PI);  // FAIL
//ParametricCurveL C([](double x) { return vec2(x, sin(5.*x)); }, -2, 2);
//ParametricCurveL C([](double x) { return vec2(x, x == 0. ? 1. : sin(2.*PI*x) / (2.*PI*x)); }, -2, 2);
//ParametricCurveL C([](double t) { return vec2(cos(t) + .5*cos(2.*t), sin(t) + .5*sin(2.*t)); }, 0, 2.*PI);
//ParametricCurveL C([](double t) { return vec2(cos(2.*t), sin(2.*t))*sin(t); }, 0, PI);
//ParametricCurveL C([](double t) { return vec2(cos(2.*t), sin(2.*t))*sin(t); }, -1, 2 * PI - 1);
//ParametricCurveL C([](double t) { return vec2(cos(t) + cos(2.*t), sin(t) + sin(2.*t))*.5; }, 0, 2.*PI);
//ParametricCurveL C([](double t) { return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, 0, 2.*PI);
//ParametricCurveL C([](double t) { return vec2(cos(t) + .5*cos(2.*t), sin(t) - .5*sin(2.*t)); }, -.5*PI, 1.5*PI);  // one may call it fail
//ParametricCurveL C([](double t) { return vec2(cos(5.*t + PI / 4.), sin(4.*t)); }, 1., 2.*PI + 1.);
//ParametricCurveL C([](double x) { return vec2(x, log(x + 1)); }, -0.99, 2.);  // FAIL
//ParametricCurveL C([](double x) { return vec2(0.5*x - 1., 0.1*tgamma(x) - 1.); }, 0.05, 5);
//ParametricCurveL C([](double x) { return vec2(.5*x, 0.04*(x*x*x*x + 2.*x*x*x - 6.*x*x - x + 1.)); }, -4., 4.);
//ParametricCurveL C([](double t) { return vec2(cos(t), sin(t)) * 0.08*t; }, 0, 6.*PI);
//ParametricCurveL C([](double t) { return vec2(cos(t), sin(t)) * 0.08*exp(0.25*t); }, -PI, 4.*PI);
//ParametricCurveL C([](double t) { return vec2(.1*t + .3*cos(t), sin(t)); }, -13., 14.);
//ParametricCurveL C([](double t) { return 0.4*vec2(cos(1.5*t), sin(1.5*t)) + vec2(cos(t), -sin(t)); }, 0, 4.*PI);
//ParametricCurveL C([](double a) { return (exp(sin(a)) - 2.*cos(4.*a) + sin((2.*a - PI) / 24.))*vec2(cos(a), sin(a)); }, -8.*PI, 8.*PI);  // FAIL (Wikipedia homepage)
//ParametricCurveL C([](double a) { return (sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 2.*PI);
//ParametricCurveL C([](double a) { return (sin(a) - cos(2.*a) + sin(3.*a))*vec2(cos(a), sin(a)); }, 0, 3.*PI);  // FAIL
//ParametricCurveL C([](double a) { return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, 0, 2.*PI);
//ParametricCurveL C([](double a) { return 0.5*(cos(a) + sin(a)*sin(a) + 1.)*vec2(cos(a), sin(a)); }, -1, 2.*PI - 1.);  // FAIL
//ParametricCurveL C([](double x) { return vec2(x, sin(sin(5.*x))); }, -2, 2);
//ParametricCurveL C([](double x) { return vec2(x, exp(sin(x)) - 1.5); }, -2, 2);
//ParametricCurveL C([](double x) { return vec2(x, exp(sin(2.*x)) - 1.5); }, -2, 2);  // FAIL
//ParametricCurveL C([](double x) { return vec2(x, .5*acos(cos(5.*x)) - .25*PI); }, -2, 2);  // FAIL
//ParametricCurveL C([](double t) { return vec2(tan(t), 1. / tan(t)); }, 0, PI);  // INFINITE LOOP
//ParametricCurveL C([](double x) { return vec2(x, tan(x)); }, -.499*PI, .499*PI);  // FAIL
//ParametricCurveL C([](double x) { return vec2(x, sqrt(1. - x * x)); }, -1., 1.);  // FAIL
//ParametricCurveL C([](double t) { return vec2(cos(t) + .01*cos(10.*t), sin(t) + .01*sin(10.*t)); }, 0, 2.*PI);  // FAIL
#endif



int main(int argc, char** argv) {
	// output format: SVG
	freopen(argv[1], "w", stdout);
	const int W = 600, H = 400;
	const double SC = 120.0;
	printf("<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>", W, H);
	printf("<g transform='matrix(%lg,0,0,%lg,%lg,%lg)' style='stroke-width:1px;stroke:black;fill:none;'>", SC, -SC, 0.5*W, 0.5*H);

	// axis
	{
		printf("<g style='stroke-width:%lgpx'>", 1.0 / SC);
		printf("<line x1='-10' x2='10' y1='0' y2='0'/><line x1='1' x2='1' y1='-.05' y2='.05'/>");
		printf("<line x1='0' x2='0' y1='-10' y2='10'/><line x1='-.05' x2='.05' y1='1' y2='1'/>");
		printf("</g>");
	}

	// discretized path
	{
		printf("<path d='");
		const int D = 200; double dt = (C.t1 - C.t0) / D;
		vec2 p = C.p(C.t0);
		printf("M%lg,%lg", p.x, p.y);
		for (int i = 1; i <= D; i++) {
			p = C.p(C.t0 + i * dt);
			printf("L%lg,%lg", p.x, p.y);
		}
		printf("' style='stroke-width:3px;stroke:#ccc;' vector-effect='non-scaling-stroke'/>");
	}

	// vectorized path
	{
		printf("<path d='");
		vec2 P0 = C.p(C.t0);
		vec2 P1 = C.p(C.t1);
		vec2 uv((C.t1 - C.t0) / 3.);
		const double eps = 0.001;
		vec2 T0 = (C.p(C.t0 + eps) - C.p(C.t0)) / eps;
		vec2 T1 = (C.p(C.t1) - C.p(C.t1 - eps)) / eps;
		std::vector<cubicBezier> sp = fitSpline(C, P0, P1, T0, T1, C.t0, C.t1, 0.001);
		fprintf(stderr, "%d curves\n", sp.size());

		printf("M%lg,%lg", P0.x, P0.y);
		for (int i = 0, l = sp.size(); i < l; i++) {
			vec2 Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
			printf("C%lg,%lg %lg,%lg %lg,%lg", Q0.x, Q0.y, Q1.x, Q1.y, P1.x, P1.y);
		}
		printf("' vector-effect='non-scaling-stroke'/>");

		// anchor points
		{
			printf("<g style='stroke-width:%lgpx;stroke:black;opacity:0.6'>", 1.0 / SC);
			printf("<defs>\
<marker id='anchor-start' viewBox='0 0 10 10' refX='5' refY='5' orient='' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
	<rect x='3.8' y='3.8' width='2.4' height='2.4' style='stroke:black;stroke-width:1px;fill:black'></rect>\
</marker><marker id='anchor-end' viewBox='0 0 10 10' refX='5' refY='5' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'>\
	<ellipse cx='5' cy='5' rx='1.2' ry='1.2' style='stroke:black;stroke-width:1px;fill:black'></ellipse>\
</marker></defs>");
			auto line = [](vec2 a, vec2 b) {
				printf("<line x1='%lg' y1='%lg' x2='%lg' y2='%lg' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)'/>", a.x, a.y, b.x, b.y);
			};
			for (int i = 0, l = sp.size(); i < l; i++) {
				vec2 P0 = sp[i].A, Q0 = sp[i].B, Q1 = sp[i].C, P1 = sp[i].D;
				line(P0, Q0); line(P1, Q1);
			}
			printf("</g>");
		}
	}

	printf("</g></svg>");
	return 0;
}
