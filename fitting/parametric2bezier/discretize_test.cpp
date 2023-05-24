// test methods to convert continuous parametric curves to samples

#include <stdio.h>
#include <vector>
#include <functional>
#include <algorithm>

#include "numerical/geometry.h"
#include "numerical/integration.h"
#include "test_cases.h"


struct segment {
	vec2 p, q;
	vec2 pq() { return q - p; }
};

// used by discretizeParametricCurve()
struct param_sample {
	double t;
	vec2 p;
	param_sample(double t = NAN, vec2 p = vec2(NAN))
		:t(t), p(p) {}
};
struct segment_sample {
	param_sample p, q;
	vec2 pq() { return q.p - p.p; }
};

// visualization functions
void initSVG(const char* filename);
void writeBlock(int id, const std::vector<segment_sample> &segs);
void endSVG();

// write file
void write_stdout(const std::vector<segment_sample> &segs);



inline bool isNAV(vec2 p) {
	double m = p.sqr();
	return isnan(m) || m > 1e32;
}

// P(t0) is NAV and P(t1) is not, find the t closest to t0 such that P(t) is not NAV
template<typename Fun/*vec2(double)*/>
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
// throw exception if it encounters NAN
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


// distance to a line segment
double distSegment(vec2 a, vec2 b, vec2 p) {
	vec2 ap = p - a, ab = b - a;
	double t = dot(ap, ab) / dot(ab, ab);
	return length(ap - ab * clamp(t, 0., 1.));
}





// maximization using cubic interpolation, used by discretizeParametricCurve()
// at most 2 extremums; NAN padding
vec2 cubicInterpolation_max(double x[4], double y[4]) {
	double x0 = x[1], x1 = x[2];
	if (!(x0 < x1)) throw("bug");
	int N = 3;
	if (isnan(x[3]) || isnan(y[3])) N--;
	if (isnan(x[0]) || isnan(y[0])) x++, y++, N--;
	if (N == 1) return vec2(NAN);  // linear interpolation, no extremum
	if (N == 2) {  // quadratic interpolation
		vec3 n(y[0], y[1], y[2]);
		vec3 u(x[1] * x[1] - x[2] * x[2], x[2] * x[2] - x[0] * x[0], x[0] * x[0] - x[1] * x[1]);
		vec3 v(x[1] - x[2], x[2] - x[0], x[0] - x[1]);
		double k = 0.5 * dot(n, u) / dot(u, v);
		return vec2(k<x0 && k>x1 ? k : NAN, NAN);
	}
	if (N == 3) {  // cubic interpolation, Lagrange
		double w[5] = { 1, 0, 0, 0, 0 };
		double C[4] = { 0, 0, 0, 0 };
		for (int d = 0; d <= N; d++) {
			for (int k = d + 1; k > 0; k--) {
				w[k] = w[k - 1] - x[d] * w[k];
			}
			w[0] = -w[0] * x[d];
		}
		for (int i = 0; i <= N; i++) {
			double m = 1;
			for (int k = 0; k <= N; k++) if (k != i) {
				m = m * (x[k] - x[i]);
			}
			m = y[i] / m;
			double c = 1;
			for (int k = N + 1; k > 0; k--) {
				C[k - 1] = C[k - 1] + m * c;
				c = w[k - 1] + c * x[i];
			}
		}
		for (int i = 1; i <= N; i++) C[i - 1] = -C[i] * i;
		double a = C[2], b = C[1], c = C[0];
		if (a < 0.) a *= -1., b *= -1., c *= -1.;
		double delta = b * b - 4.*a*c;
		if (delta < 0.) return vec2(NAN);
		delta = sqrt(delta);
		double xm = NAN;
		double r = (-delta - b) / (2.*a);
		if (r > x0 && r < x1) xm = r;
		r = (delta - b) / (2.*a);
		if (r > x0 && r < x1) return isnan(xm) ? vec2(r, NAN) : vec2(xm, r);
		return vec2(xm, NAN);
	}
	return vec2(NAN);
}
vec2 cubicInterpolation_max(double p0, double p1, double d0, double d1) {
	double a = 6.*p0 + 3.*d0 - 6.*p1 + 3.*d1;
	double b = -6.*p0 - 4.*d0 + 6.*p1 - 2.*d1;
	double c = d0;
	double mt = abs(p0) > abs(p1) ? 0. : 1.;
	double mv = max(abs(p0), abs(p1));
	if (a*a < 1e-16) {
		if (b*b < 1e-16) {
			return vec2(NAN);
		}
		double t = -c / b;
		if (t<1e-6 || t>.999999) return vec2(NAN);
		return vec2(t, NAN);
	}
	double d = b * b - 4.*a*c;
	if (d < 0.) return vec2(NAN);
	d = sqrt(d);
	double t0 = (d - b) / (2.*a), t1 = (-d - b) / (2.*a);
	if (t0<.000001 || t0>.999999) t0 = NAN;
	if (t1<.000001 || t1>.999999) t1 = NAN;
	if (isnan(t0)) std::swap(t0, t1);
	if (t0 > t1) std::swap(t0, t1);
	return vec2(t0, t1);
}



// attempts to make segments with even chord lengths
// attempts to fit in the error with as few samples as possible (but not wasting any sample)
// @min_dif: minimum number of initial subdivisions, must be positive for user call (doesn't needs to be too large)
// @reqLength: required chord length, length of segments will be around this number when the curvature of the curve isn't too large
// @reqError: required error, the distance from one point on the curve to the closest line segment will *usually* be less than this number
// @recurse_remain: pass the maximum number of allowed recursions in user call
// @P_0, @P_1: neighbourhood samples for better interpolation
// @sin_left, @sin_right: indicate if there is a previously confirmed jump discontinuity at the left/right of the interval
std::vector<segment_sample> discretizeParametricCurve(std::function<vec2(double)> F,
	param_sample s0, param_sample s1,
	int min_dif, double reqLength, double reqError, int recurse_remain,
	param_sample P_0 = param_sample(), param_sample P_1 = param_sample(),
	bool sin_left = false, bool sin_right = false) {

	std::vector<segment_sample> res;
	if (recurse_remain < 0) {  // recursion limit exceeded
		res.push_back(segment_sample{ s0, s1 });
		return res;
	}

	// user call
	if (min_dif > 0) {
		// handle the case when t0 or t1 is INF
		std::function<vec2(double)> _F = [&](double t) { return F(t); };
		if (abs(s1.t - s0.t) > 1e4) {  // INF case
			_F = [&](double t) { return F(tan(t)); };
			s0.t = atan(s0.t), s1.t = atan(s1.t);
		}

		reqLength *= 2.0 * 1.3;  // hmmm...
		if (!(reqLength > 0. && reqError > 0.)) return res;  // no messing around

		// take samples
		double dt = (s1.t - s0.t) / min_dif;
		param_sample *samples = new param_sample[min_dif + 1];
		for (int i = 1; i < min_dif; i++) {
			double t = s0.t + (i + 0.01*sin(123.456*i)) * dt;
			samples[i] = param_sample(t, _F(t));
			//res.push_back(segment_sample{ param_sample(t, samples[i].p - vec2(.1,0)), param_sample(t, samples[i].p + vec2(.1,0)) });
		}
		samples[0] = s0;
		samples[min_dif] = s1;

		// recursive calls
		for (int i = 0; i < min_dif; i++) {
			std::vector<segment_sample> app = discretizeParametricCurve(_F,
				samples[i], samples[i + 1],
				0, reqLength, reqError, recurse_remain - 1,
				i == 0 ? param_sample() : samples[i - 1],
				i + 1 == min_dif ? param_sample() : samples[i + 2]
			);
			res.insert(res.end(), app.begin(), app.end());
		}
		delete samples;

		/*
		After the call, check:
		 * too large turning angles;
		 * rapid change in chord lengths;
		 * discontinuities in segments;
		Perform bisection / golden section search on the dot product of the curve with a chosen vector
			to minimize the number of incorrect samples.
		*/

		return res;
	}

	// handle NAN
	if (isNAV(s0.p) && isNAV(s1.p))
		return res;
	if (isNAV(s0.p)) {
		if (!bisectNAV(F, s0.t, s1.t, s0.p, s1.p, s0.t, s0.p)) return res;
		P_0.t = NAN;
	}
	else if (isNAV(s1.p)) {
		if (!bisectNAV(F, s1.t, s0.t, s1.p, s0.p, s1.t, s1.p)) return res;
		P_1.t = NAN;
	}

	// subdivision global variables
	double tc, tc0, tc1;
	vec2 pc, pc0, pc1;
	bool hasJump = false;

	// continue subdivision until error is small enough
	vec2 dP = s1.p - s0.p;
	double dPL = length(dP);
	if (dPL == 0.0) return res;
	if (dPL < reqLength || min_dif == -1) {

#if 0
		// split at the point(s) that produce the maximum value
		vec2 n = (s1.p - s0.p).rot();
		double x[4] = { P_0.t, s0.t, s1.t, P_1.t };
		double y[4] = { dot(n, P_0.p), dot(n, s0.p), dot(n, s1.p), dot(n, P_1.p) };
		vec2 tcp = cubicInterpolation_max(x, y);
		if (isnan(tcp.x)) {
			tcp = vec2(0.5*(s0.t + s1.t), NAN);
			//tcp = (1. / 3.)*(vec2(2., 1.)*t0 + vec2(1., 2.)*t1);
		}
#else
		// better honestly split in half :(
		vec2 tcp = vec2(.5*(s0.t + s1.t), NAN);
#endif

		if (isnan(tcp.y)) {  // divide into 2
			tc = tcp.x; pc = F(tc);
			if (distSegment(s0.p, s1.p, pc) < reqError) {
				param_sample sc{ tc, pc };
				res.push_back(segment_sample{ s0, sc });
				res.push_back(segment_sample{ sc, s1 });
				return res;
			}
			// check jump discontinuity
			if (!(sin_left || sin_right)) {
				double l0 = length(pc - s0.p) / (tc - s0.t), l1 = length(pc - s1.p) / (s1.t - tc);
				if (l0 > 2.*l1 || l1 > 2.*l0) {
					double u0 = s0.t, u1 = s1.t; vec2 v0 = F(u0), v1 = F(u1);
					try {
						bool succeed = boundJump(F, u0, u1, v0, v1);
						tc0 = u0, tc1 = u1, pc0 = v0, pc1 = v1;
						hasJump = succeed;
						goto divideJump;
					} catch (double t) {
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
			if (distSegment(s0.p, s1.p, pc0) < reqError && distSegment(s0.p, s1.p, pc1) < reqError) {
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
			tc0 = s0.t + (s1.t - s0.t) / 3.;  // experimental value
			tc1 = s1.t - (s1.t - s0.t) / 3.;  // experimental value
			pc0 = F(tc0), pc1 = F(tc1);
			goto divide3;
		}
		else
#endif
		{  // divide into 2
			tc = 0.5*(s0.t + s1.t);  // experimental value
			pc = F(tc);
			// check jump discontinuity
			double l0 = length(pc - s0.p), l1 = length(pc - s1.p);
			if (!(sin_left || sin_right) && min(l0, l1) < reqLength) {
				if (l0 > 2.*l1 || l1 > 2.*l0) {
					double u0 = s0.t, u1 = s1.t; vec2 v0 = F(u0), v1 = F(u1);
					try {
						bool succeed = boundJump(F, u0, u1, v0, v1);
						tc0 = u0, tc1 = u1, pc0 = v0, pc1 = v1;
						hasJump = succeed;
						goto divideJump;
					} catch (double t) {
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
		param_sample sc(tc, pc);
		std::vector<segment_sample> app0 = discretizeParametricCurve(F,
			s0, sc, 0, reqLength, reqError, recurse_remain - 1,
			P_0, s1, sin_left, false);
		std::vector<segment_sample> app1 = discretizeParametricCurve(F,
			sc, s1, 0, reqLength, reqError, recurse_remain - 1,
			s0, P_1, false, sin_right);


		// attempt to fix missed samples [not quite successful]
		if (1) do {
			double l0 = length(pc - s0.p), l1 = length(s1.p - pc);
			int n0 = (int)app0.size(), n1 = (int)app1.size();
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
							std::vector<segment_sample> tmp = discretizeParametricCurve(F,
								ps.p, ps.q, 0, reqLength, reqError, recurse_remain - 2,
								(*app_s)[max(i - 1, 0)].q, (*app_s)[min(i + 1, nm - 1)].p
							);
							app.insert(app.end(), tmp.begin(), tmp.end());
						}
						*app_s = app;
					}
				}
			}
		} while (0);


		// add
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), app1.begin(), app1.end());
		return res;
	}

divide3:
	{
		// split into 3, not actually used
		param_sample sc0(tc0, pc0), sc1(tc1, pc1);
		std::vector<segment_sample> app0, appc, app1;
		app0 = discretizeParametricCurve(F,
			s0, sc0, 0, reqLength, reqError, recurse_remain - 1,
			P_0, sc1, sin_left, false);
		appc = discretizeParametricCurve(F,
			sc0, sc1, 0, reqLength, reqError, recurse_remain - 1,
			s0, s1);
		app1 = discretizeParametricCurve(F,
			sc1, s1, 0, reqLength, reqError, recurse_remain - 1,
			sc0, P_1, false, sin_right);
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
		app0 = discretizeParametricCurve(F,
			s0, sc0, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(), param_sample(), false, true);
		if (!hasJump && tc1 - tc0 > 1e-6) appc = discretizeParametricCurve(F,
			sc0, sc1, 0, reqLength, reqError, recurse_remain - 1);
		app1 = discretizeParametricCurve(F,
			sc1, s1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(), param_sample(), true, false);
		res.insert(res.end(), app0.begin(), app0.end());
		if (!appc.empty()) res.insert(res.end(), appc.begin(), appc.end());
		else if (!hasJump) res.push_back(segment_sample{ sc0, sc1 });
		res.insert(res.end(), app1.begin(), app1.end());
		return res;
	}

}


#if 0
std::vector<std::vector<segment_sample>> discretizeParametricCurve_check(
	std::function<vec2(double)> F, std::vector<segment_sample> ss) {

	std::vector<std::vector<segment_sample>> sss;
	if (ss.empty()) return sss;
	for (int i = 0; i < (int)ss.size(); i++) {
		if (sss.empty()) {
			sss.push_back(std::vector<segment_sample>());
		}
		else if (sss.back().back().q.t != ss[i].p.t) {
			// check back
			std::vector<segment_sample> &s = sss.back();
			{}

			sss.push_back(std::vector<segment_sample>());
		}
		sss.back().push_back(ss[i]);
	}

	return sss;
}
#endif





int main(int argc, char* argv[]) {
	initSVG(argv[1]);

	for (int i = 0; i < CSN; i++) {
		Parametric_callCount = 0;

		ParametricCurveL Curve = Cs[i];
		std::function<vec2(double)> Fun = [&](double t) {
			vec2 p = Curve.p(t);
			return abs(p.x) < 2.5 && abs(p.y) < 2.5 ? p : vec2(NAN);
		};
		std::vector<segment_sample> segs = discretizeParametricCurve(
			Fun,
			param_sample(Curve.t0, Curve.p(Curve.t0)), param_sample(Curve.t1, Curve.p(Curve.t1)),
			96, 0.05, 0.001, 18);

		writeBlock(i, segs);
		write_stdout(segs);
		fprintf(stderr, "%d\n", i);
	}

	endSVG();
	printf("-1\n");

	return 0;
}



// SVG writer
FILE* fp = 0;
const double scale = 128.;  // scaling from graph to svg
const int width = 640;  // width and height of each block
const int colspan = 2;  // number of blocks in a row
int blockCount = 0;  // number of blocks writed
void initSVG(const char* filename) {
	fp = fopen(filename, "wb");
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='%d' height='%d'>\n", width*colspan, 180 * width);
	fprintf(fp, "<defs><clipPath id='viewbox'><rect x='%d' y='%d' width='%d' height='%d'/></clipPath></defs>\n", 0, 0, width, width);
	fprintf(fp, "<style>text{font-family:Consolas;font-size:14px;}</style>\n");
}
void writeBlock(int id, const std::vector<segment_sample> &segs) {
	int vn = (int)segs.size();

	fprintf(fp, "<g transform='translate(%d,%d)' clip-path='url(#viewbox)'>\n",
		width*(blockCount%colspan), width*(blockCount / colspan));
	fprintf(fp, "<rect x='0' y='0' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", width, width);
	fprintf(fp, "<text x='10' y='20'>#%d - %d segments, %d samples</text>\n", id, vn, Parametric_callCount);  // should work

	// calculate arc length average and standard deviation
	double avrL = 0.;
	for (int i = 0; i < vn; i++)
		avrL += length(segs[i].q.p - segs[i].p.p) / vn;
	double stdL = 0.;
	for (int i = 0; i < vn; i++)
		stdL += pow(length(segs[i].q.p - segs[i].p.p) - avrL, 2.);
	stdL = sqrt(stdL / (vn - 1));
	fprintf(fp, "<text x='10' y='40'>avrL=%lf, stdL=%lf</text>\n", avrL, stdL);

	fprintf(fp, "<g transform='matrix(%lg,0,0,%lg,%lg,%lg)'>\n", scale, -scale, .5*width, .5*width);


	// segments (colored)
#if 0
	fprintf(fp, "<g style='stroke-width:%lg'>", 1. / scale);
	for (int i = 0; i < vn; i++) {
		fprintf(fp, "<line x1='%lg' y1='%lg' x2='%lg' y2='%lg' stroke='%s'/>",
			points[i].p.p.x, points[i].p.p.y, points[i].q.p.x, points[i].q.p.y, i == 0 ? "green" : i & 1 ? "red" : "blue");
	}
	fprintf(fp, "</g>\n");
#else
	fprintf(fp, "<g style='stroke-width:1'>");
	if (vn) fprintf(fp, "<line x1='%lg' y1='%lg' x2='%lg' y2='%lg' stroke='green' vector-effect='non-scaling-stroke'/>",
		segs[0].p.p.x, segs[0].p.p.y, segs[0].q.p.x, segs[0].q.p.y);
	fprintf(fp, "<path d='");
	for (int i = 1; i < vn; i += 2) fprintf(fp, "M%lg,%lgL%lg,%lg",
		segs[i].p.p.x, segs[i].p.p.y, segs[i].q.p.x, segs[i].q.p.y);
	fprintf(fp, "' stroke='red' fill='none' vector-effect='non-scaling-stroke'/>");
	fprintf(fp, "<path d='");
	for (int i = 2; i < vn; i += 2) fprintf(fp, "M%lg,%lgL%lg,%lg",
		segs[i].p.p.x, segs[i].p.p.y, segs[i].q.p.x, segs[i].q.p.y);
	fprintf(fp, "' stroke='blue' fill='none' vector-effect='non-scaling-stroke'/>");
	fprintf(fp, "</g>\n");
#endif

	// starting points
	vec2 p0(NAN);
	for (int i = 0; i < vn; i++) {
		if (segs[i].p.p != p0)
			fprintf(fp, "<circle cx='%lf' cy='%lf' r='%lf' style='stroke:none;fill:black;opacity:0.3;'/>\n",
				segs[i].p.p.x, segs[i].p.p.y, 3.0 / scale);
		p0 = segs[i].q.p;
	}

	fprintf(fp, "</g></g>");
	fflush(fp);
	blockCount++;
}
void endSVG() {
	fprintf(fp, "</svg>");
	fclose(fp);
}


void write_stdout(const std::vector<segment_sample> &segs) {
	std::vector<std::vector<vec2>> points;

	vec2 old_q = vec2(NAN);
	for (int i = 0; i < (int)segs.size(); i++) {
		if (segs[i].p.p != old_q) {
			points.push_back(std::vector<vec2>());
			points.back().push_back(segs[i].p.p);
		}
		points.back().push_back(segs[i].q.p);
		old_q = segs[i].q.p;
	}

	printf("%d\n", (int)points.size());
	for (int i = 0; i < (int)points.size(); i++) {
		printf("%d\n", (int)points[i].size());
		for (vec2 p : points[i]) {
			printf("%.8lg %.8lg\n", p.x, p.y);
		}
	}
	printf("\n");

}






