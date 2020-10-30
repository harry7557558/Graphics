// Curve fitting experiment - fit 2d parametric curves to cubic Bezier spline
// SVG graphs to stdout, and program outputs to stderr

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



inline bool isNAV(vec2 p) {
	double m = p.sqr();
	return isnan(m) || m > 1e32;
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
#include "cubicdist.h"
static uint32_t distCubic2_callCount = 0;  // a counter for testing
double distCubic2(cubicCurve c, vec2 p) {
	distCubic2_callCount++;
	return CubicCurveDistance2(&c.c0, p);
};



// fit a curve interval using one cubic bezier curve
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

	// length of the curve
	double clength = 0.;
	for (int i = 0; i < 32; i++) clength += dL[i];

	// calculate the "radius" of the samplings (for computing the level of details)
	if (sample_radius) {
		vec2 c(0.);
		for (int i = 0; i < 32; i++) c += P[i];
		c *= 1. / 32;
		double mr = 0.;
		for (int i = 0; i < 32; i++) mr = max(mr, (P[i] - c).sqr());
		*sample_radius = sqrt(mr);
		// so the following computations can be saved
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
	double cst = cost(uv.x, uv.y);
	if (cst < 1e-8*clength && !(abs(ndet(T0, P1 - P0)) > 1e-6) && !(abs(ndet(T1, P1 - P0)) > 1e-6)) {
		uv = vec2(NAN);  // automatically becomes line segment
	}
	return cst;
}

// P0, P1: starting and ending points; T0, T1: tangent vector (derivative)
// Error is calculated as the integral of shortest distance to the bezier curve respect to arc length of C
// lod_radius: radius for the level of detail; use line segments instead of bezier curve for complicated/fractal curves (necessary)
template<typename Fun>
std::vector<cubicBezier> fitSpline(Fun C, vec2 P0, vec2 P1, vec2 T0, vec2 T1, double t0, double t1,
	double allowed_err, double lod_radius, double* Err = nullptr, double *LengthC = nullptr) {

	std::vector<cubicBezier> res;
	if (!(t1 > t0)) return res;
	double clength = 0.;

	// try curve fitting
	vec2 uv((t1 - t0) / 3.);
	double spr;
	double err = fitPartCurve(C, P0, P1, T0, T1, t0, t1, uv, &clength, &spr, lod_radius);
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



// split the curve by singularities, return parametric pairs [t0,t1]
template<typename Fun>
std::vector<vec2> splitSingularity(const Fun &C, double t0, double t1, int mindif = 420/*2x2x3x5x7*/,
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

	ints.clear();
	int TN = res.size();
	for (int i = 0; i < TN; i++) ints.push_back(res[i] + vec2(1e-12, -1e-12));
	res.clear();


	// find jump discontinuity, assume v0,v1 are already evaluated
	// t=NAN: not found
	auto findJump = [&](double &t0, double &t1, vec2 &v0, vec2 &v1, double eps = 1e-12) {
		double dt = t1 - t0;
		double dif0 = length(v1 - v0) / dt;
		double dist = dif0 * dt;
		for (int i = 0; i < 60; i++) {
			double tc = 0.5*(t0 + t1);
			vec2 vc = P(tc);
			if (isNAV(vc)) throw(tc);
			dt *= .5;
			double dif1 = length(vc - v0) / dt;
			double dif2 = length(v1 - vc) / dt;
			if (i > 2 && !(max(dif1, dif2) > 1.2*dif0)) break;
			if (dif1 > dif2) t1 = tc, v1 = vc, dif0 = dif1;
			else t0 = tc, v0 = vc, dif0 = dif2;
			double new_dist = dif0 * dt;
			if (t1 - t0 < eps) {
				// needs to makes sure there is a jump
				if (new_dist / dist < 0.99) {
					//fprintf(stderr, "%lf\n", (new_dist / dist));
					if (!(t1 == t0 || i == 59)) continue;
				}
				//fprintf(stderr, "%lg\n", t1 - t0);
				return;
			}
			dist = new_dist;
		}
		t0 = t1 = NAN;
	};

	// find non-differentiable point
	auto findC0 = [&](double &t0, double &t1, vec2 &v0, vec2 &v1, double eps = 1e-12) {
		double dt2 = (t1 - t0)*(t1 - t0);
		double tc = 0.5*(t0 + t1);
		vec2 vc = P(tc); if (isNAV(vc)) throw(tc);
		double dif = length(v0 + v1 - 2.*vc) / dt2;
		for (int i = 0; i < 64; i++) {
			double tc0 = 0.5*(t0 + tc), tc1 = 0.5*(tc + t1);
			vec2 vc0 = P(tc0), vc1 = P(tc1);
			if (isNAV(vc0)) throw(tc0);
			if (isNAV(vc1)) throw(tc1);
			dt2 *= .25;
			double dif0 = length(v0 + vc - 2.*vc0) / dt2;
			double dif1 = length(vc + v1 - 2.*vc1) / dt2;
			double difc = length(vc0 + vc1 - 2.*vc) / dt2;
			if (i > 2 && !(max(max(dif0, dif1), difc) > max(1.2*dif, 1e-6))) break;
			if (difc >= dif0 && difc >= dif1) {
				t0 = tc0, t1 = tc1, v0 = vc0, v1 = vc1;
				dif = difc;
			}
			else if (dif0 > dif1) {
				t1 = tc, tc = tc0, v1 = vc, vc = vc0;
				dif = dif0;
			}
			else {
				t0 = tc, tc = tc1, v0 = vc, vc = vc1;
				dif = dif1;
			}
			if (t1 - t0 < eps) return;
		}
		t0 = t1 = NAN;
	};

	for (int T = 0; T < TN; T++) {
		double u0 = ints[T].x, u1 = ints[T].y;
		int i0 = (int)ceil((u0 - t0 + 1e-6) / dt), i1 = (int)floor((u1 - t0 - 1e-6) / dt);
		if (i0 > 0 && !isNAV(Ss[i0 - 1].p)) i0--;
		if (i1 < mindif && !isNAV(Ss[i1 + 1].p)) i1++;
		int di = i1 - i0;

		// calculate the differences from neighborhood samples
		std::vector<vec2> d1, d2, d3, d4;
		{
			for (int i = i0; i <= i1; i++) {
				//d1.push_back(i == 0 || i == mindif ? vec2(NAN) : .5 *(Ss[i + 1].p - Ss[i - 1].p) / dt);
				d1.push_back((i == 0 ? (Ss[1].p - Ss[0].p) : i == mindif ? (Ss[i].p - Ss[i - 1].p) \
					: .5 *(Ss[i + 1].p - Ss[i - 1].p)) / dt);
			}
			for (int i = 0; i <= di; i++) {
				d2.push_back((i == 0 ? (d1[1] - d1[0]) : i == di ? (d1[i] - d1[i - 1])
					: .5 *(d1[i + 1] - d1[i - 1])) / dt);
			}
			for (int i = 0; i <= di; i++) {
				d3.push_back((i == 0 ? (d2[1] - d2[0]) : i == di ? (d2[i] - d2[i - 1])
					: .5 *(d2[i + 1] - d2[i - 1])) / dt);
			}
			for (int i = 0; i <= di; i++) {
				d4.push_back((i == 0 ? (d3[1] - d3[0]) : i == di ? (d3[i] - d3[i - 1])
					: .5 *(d3[i + 1] - d3[i - 1])) / dt);
			}
		}

		// sort the 4th differences
		struct sampled {
			int id; double mag;
		};
		std::vector<sampled> d4s;
		for (int i = 0; i <= di; i++) {
			d4s.push_back(sampled{ i + i0, isNAV(d4[i]) ? -1. : length(d4[i]) });
		}
		std::sort(d4s.begin(), d4s.end(), [](sampled a, sampled b) { return a.mag > b.mag; });
		for (int i = 0; i <= di; i++) {
			//if (d4s[i].mag < 1e4) break;
			//printf("<circle cx='%lf' cy='%lf' r='%lf'/>", Ss[d4s[i].id].p.x, Ss[d4s[i].id].p.y, 0.05 * d4s[i].mag / d4s[0].mag);
		}

		// find singularities
		std::vector<vec2> sings;  // vec2(tl,tr)
		int failcount = 0; const int maxfailcount = max(mindif / 10, 20);
		for (int j = 0; failcount < maxfailcount && j <= di; j++) {
			int i = d4s[j].id; double t = Ss[i].t;
			double t0 = t - 1.5*dt, t1 = t + 1.5*dt;
			t0 = max(t0, u0), t1 = min(t1, u1);
			vec2 v0 = P(t0), v1 = P(t1);
			double _t0 = t0, _t1 = t1; vec2 _v0 = v0, _v1 = v1;  // backup
			double t_NAV = NAN;  // t that produces NAV
			try { findJump(t0, t1, v0, v1); }  // detect jump discontinuity
			catch (double tc) { t_NAV = tc; }
			if ((isnan(t0) || isnan(t1)) && isnan(t_NAV)) {  // detect non-differentiable point
				try { findC0(t0 = _t0, t1 = _t1, v0 = _v0, v1 = _v1); }
				catch (double tc) { t_NAV = tc; }
			}
			if (!isnan(t_NAV)) {  // most commonly infinite discontinuity
				double tl = max(t - 2.*dt, u0), tr = min(t + 2.*dt, u1);
				vec2 pl = P(tl), pr = P(tr);
				if (bisectNAN(P, t_NAV, tl, vec2(NAN), pl, tl, pl)
					&& bisectNAN(P, t_NAV, tr, vec2(NAN), pr, tr, pr))
					sings.push_back(vec2(tl, tr));
				continue;
			}
			if (isnan(t0) || isnan(t1)) {
				failcount++;
			}
			else {
				sings.push_back(vec2(t0, t1));
				failcount = 0;
			}
		}
		std::sort(sings.begin(), sings.end(), [](vec2 a, vec2 b) {return a.x < b.x; });

		// split the interval by singularities
		sings.push_back(vec2(u1));
		double tl = u0, tr = u0;
		int add_count = 0;  // newly added interval count
		for (int i = 0, l = sings.size(); i < l; i++) {
			double t0 = sings[i].x, t1 = sings[i].y;
			if (!(t0 - tl < 1e-6)) {
				//printf("<line x1='%lf' y1='%lf' x2='%lf' y2='%lf' style='stroke-width:0.05px;'/>", P(tl).x, P(tl).y, P(tr).x, P(tr).y);
				if (add_count != 0) res.back().y = tl;
				res.push_back(vec2(tr, t0)); add_count++;
				tl = t0, tr = t1;
			}
			else {
				//if (t1 > tr) tr = t1;
				if (t0 > tl) tl = t0;
				if (t1 < tr) tr = t1;
			}
		}
	}

	// happens when the curve looks smooth but its derivative diverges
	for (int i = 1, l = res.size(); i < l; i++) {
		if (res[i - 1].y > res[i].x)
			res[i - 1].y = res[i].x = 0.5*(res[i - 1].y + res[i].x);
	}

	fprintf(stderr, "%d; ", res.size());
	return res;

}



// this function does everything automatically
template<typename ParamCurve>
std::vector<cubicBezier> fitSpline_auto(ParamCurve C, vec2 B0, vec2 B1,
	double allowed_err = 0.001, double lod_radius = 0.01, double* Err = nullptr, double *LengthC = nullptr) {

	// substitution for INF parameters
	if (abs(C.t1 - C.t0) > 1e4) {
		// use tan instead of atanh because tan(atan(x))!=INF
		return fitSpline_auto(
			ParamCurve([&](double t) { return C.p(tan(t)); }, atan(C.t0), atan(C.t1)),
			B0, B1, allowed_err, lod_radius, Err, LengthC);
	}

	std::vector<vec2> Cs = splitSingularity(C.p, C.t0, C.t1, 420, B0, B1);
	int CsN = Cs.size();

	std::vector<cubicBezier> Res;
	double cumErr = 0., cumLength = 0.;

	for (int i = 0; i < CsN; i++) {
		double t0 = Cs[i].x, t1 = Cs[i].y, _t0 = t0, _t1 = t1;
		const double eps = 0.0001*(t1 - t0);

		auto P = C.p;
		vec2 P0 = P(t0);
		vec2 P1 = P(t1);
		vec2 T0, T1;
		auto calcTangent = [&]() {
			T0 = (P(t0 + eps) - P0) / eps;
			T1 = (P1 - P(t1 - eps)) / eps;
		};
		calcTangent();

		// check endpoint singularities
		vec2 U0 = (P(t0 + .5*eps) - P0) / (.5*eps);
		vec2 U1 = (P1 - P(t1 - .5*eps)) / (.5*eps);
		double M0 = U0.sqr() / T0.sqr(), M1 = U1.sqr() / T1.sqr();
		bool M0_ok = abs(M0 - 1.) < .1, M1_ok = abs(M1 - 1.) < .1;
		if (M0_ok && M1_ok);  // good-conditioned function
		else if (isnan(M0) || isnan(M1)) {  // idk, barely cause problems
			fprintf(stderr, "NAN %d; ", __LINE__);
		}
		else if (M0 < 1.1 && M1 < 1.1);  // zero-tangent, may be improved
		else {  // remap
			if (max(M0, M1) > 2.8) {  // very bad-conditioned parameters
				fprintf(stderr, "High dif %lf; ", max(M0, M1));
			}
			// remap using power function
			if (M1_ok && M0 > 1.) {
				P = [&](double t) {
					double mp = pow(t, M0);
					return C.p(_t0 + (_t1 - _t0)*mp);
				};
				t0 = 0., t1 = 1.; calcTangent();
			}
			else if (M0_ok && M1 > 1.) {
				P = [&](double t) {
					double mp = 1. - pow(1. - t, M1);
					return C.p(_t0 + (_t1 - _t0)*mp);
				};
				t0 = 0., t1 = 1.; calcTangent();
			}
			// warn that this remapping function is C0 continuous
			else if (abs(M1 - M0) < .1) {
				double k = sqrt(M0*M1);
				P = [&](double t) {
					double mp = t < .5 ? .5*pow(2.*t, k) : 1. - .5*pow(2.*(1 - t), k);
					return C.p(_t0 + (_t1 - _t0)*mp);
				};
				t0 = 0., t1 = 1.; calcTangent();
			}
			// relatively rare but sometimes happen
			else {
				fprintf(stderr, "Remapping needed; ");
			}
		}

		double err = 0., clen = 0.;
		std::vector<cubicBezier> R = fitSpline(P, P0, P1, T0, T1, t0, t1, allowed_err, lod_radius, &err, &clen);
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
		if (m*m < 1e-12) return ap.sqr();
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
	const double SC = 120.0, invSC = 1.0 / SC;  // scale from function coordinate to screen coordinate
	const vec2 Bound = 0.5 * vec2(W, H) * invSC;  // viewbox for each function graph: ±Bound
	const int Chart_N = 7;  // number of statistic charts
	const int Chart_W = 800, Chart_H = 300;  // width and height of each chart
	const vec2 Chart_Margin = vec2(120, 60);  // horizontal and vertical margin of each chart
	const int Width = min(Graph_N, 2) * W;  // width of the overall graph
	const int Height_F = (int)(((Graph_N + 1) / 2)*H);  // total height of the function part
	const int Height = Height_F + (Graph_N > 4) * Chart_N * int(Chart_H + 2 * Chart_Margin.y) + 80;  // height of the overall graph
}

// double to string for printf
std::string _f_buffer[0x40000];
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
	if (0) {
		_f_buffer_n = 0;
		start_timer();
		std::vector<vec2> Ps = splitSingularity(C.p, C.t0, C.t1, 420, vec2(-2.5, -1.7), vec2(2.5, 1.7));
		end_timer();
		spn = 0;
		for (int i = 0, L = Ps.size(); i < L; i++) {
			double t0 = Ps[i].x, t1 = Ps[i].y, dt = t1 - t0;
			if (dt < 1e-4) fprintf(stderr, "dt=%lg\n", dt);
			start_timer();
			std::vector<segment> S = Param2Segments(C.p, C.p(t0), C.p(t1), t0, t1, 0.001, 17);
			end_timer();
			spn += S.size();
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

	Parametric_callCount = 0;
	distCubic2_callCount = 0;
	_f_buffer_n = 0;

	// vectorized path
	{
		double clength;
		start_timer();
		std::vector<cubicBezier> sp = fitSpline_auto(C, -SVG::Bound, SVG::Bound, 0.001, 0.01, &err, &clength);
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
		int px = ((i - CS0) % 2)*SVG::W, py = ((i - CS0) / 2)*SVG::H;
		printf("\n<!-- Test Case #%d -->\n", i);
		printf("<rect x='%d' y='%d' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", px, py, SVG::W, SVG::H);
		printf("<g transform='matrix(%lg,0,0,%lg,%lg,%lg)' clip-path='url(#viewbox)' style='stroke-width:%lgpx;stroke:black;fill:none;'>\n",
			SVG::SC, -SVG::SC, px + .5*SVG::W, py + .5*SVG::H, 1.0 * SVG::invSC);
		fprintf(stderr, "Case %3d - ", i);

		int spn = 0;
		double err = abs(NAN);
		double time_elapsed = 0.;
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
	if (SVG::Graph_N > 4) {
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
