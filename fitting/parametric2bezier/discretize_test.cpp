// test methods to convert continuous parametric curves to samples

#include <stdio.h>
#include <vector>
#include <functional>
#include <algorithm>

#include "numerical/geometry.h"
#include "numerical/integration.h"
#include "test_cases.h"


// visualization functions
void initSVG(const char* filename);
void writeBlock(const std::vector<vec2> &points);
void endSVG();



inline bool isNAV(vec2 p) {
	double m = p.sqr();
	return isnan(m) || m > 1e32;
}

// distance to a line segment
double distSegment(vec2 a, vec2 b, vec2 p) {
	vec2 ap = p - a, ab = b - a;
	double t = dot(ap, ab) / dot(ab, ab);
	return length(ap - ab * clamp(t, 0., 1.));
}

// used by discretizeParametricCurve()
struct param_sample {
	double t;
	vec2 p;
	param_sample(double t = NAN, vec2 p = vec2(NAN))
		:t(t), p(p) {}
};

// estimate the tangent (derivative) at Sc from neighborhood samples
// try to get rid of the O(h²) term in the Taylor series expansion
vec2 estimateDerivative(param_sample S0, param_sample Sc, param_sample S1, bool map01 = true) {
	double t1 = S1.t - Sc.t, t0 = S0.t - Sc.t;
	vec2 p1 = S1.p - Sc.p, p0 = S0.p - Sc.p;
	// S0 and S1 must not be both NAN
	if (isnan(t0)) return p1 / t1;
	if (isnan(t1)) return p0 / t0;
	// map01: map [t0,t1] to [0,1], multiply by t1-t0
	//return (p1 - p0) / (map01 ? 1. : t1 - t0);
	return (p0 * t1*t1 - p1 * t0*t0) / (t0*t1*(map01 ? 1. : t1 - t0));
}

// estimate the arc length between P0 and P1 from neighbourhood samples
double estimateArcLength(param_sample P_0, param_sample P0, param_sample P1, param_sample P_1) {
	if (isnan(P_0.t) /*&&*/ || isnan(P_1.t))
		return length(P1.p - P0.p);
	vec2 d0 = estimateDerivative(P_0, P0, P1);
	vec2 d1 = estimateDerivative(P0, P1, P_1);
	// cubic Hermite spline: p(t) = (2t³-3t²+1)P0+(t³-2t²+t)d0+(-2t³+3t²)P1+(t³-t²)d1
	return NIntegrate_GL4<double>([&](double t) {
		return length((6.*t*t - 6.*t)*P0.p + (3.*t*t - 4.*t + 1.)*d0 + (-6.*t*t + 6.*t)*P1.p + (3.*t*t - 2.*t)*d1);
	}, 0., 1.);
}

// select arc length
// debug this function
double selectArcLength(param_sample P_0, param_sample P0, param_sample P1, param_sample P_1, double k) {
	if (isnan(P_0.t) || isnan(P_1.t))
		return k;
	vec2 d0 = estimateDerivative(P_0, P0, P1);
	vec2 d1 = estimateDerivative(P0, P1, P_1);
	auto ds = [&](double t) {
		return length((6.*t*t - 6.*t)*P0.p + (3.*t*t - 4.*t + 1.)*d0 + (-6.*t*t + 6.*t)*P1.p + (3.*t*t - 2.*t)*d1);
	};
	double l0 = NIntegrate_GL48<double>(ds, 0., 1.);
	double t0 = 0., t1 = 1., t;
	for (int i = 0; i < 16; i++) {
		t = 0.5*(t0 + t1);
		double v = NIntegrate_GL48<double>(ds, 0, t) / l0;
		if (v > k) t1 = t;
		else t0 = t;
	}
	printf("%lf\n", t);
	return t;
}


// attempts to make segments with even chord lengths
// attempts to fit in the error with as few samples as possible (but not wasting any sample)
// @min_dif must be non-zero for user call (doesn't needs to be too large)
// @P_0, @P_1: neighbourhood samples for better interpolation
template<typename Fun>
std::vector<vec2> discretizeParametricCurve(Fun F,
	double t0, double t1, vec2 P0, vec2 P1,
	int min_dif, double reqLength, double reqError, int recurse_remain,
	param_sample P_0 = param_sample(),
	param_sample P_1 = param_sample()) {

	std::vector<vec2> res;
	if (recurse_remain < 0) return res;

	if (min_dif > 0) {  // user call
		//reqLength *= 2.0 * 1.3;  // hmmm...
		if (!(reqLength > 0. && reqError > 0.)) return res;  // no messing around

		// take samples
		double dt = (t1 - t0) / min_dif;
		param_sample *samples = new param_sample[min_dif + 1];
		for (int i = 1; i < min_dif; i++) {
			double t = t0 + i * dt;
			samples[i] = param_sample(t, F(t));
		}
		samples[0] = param_sample(t0, P0);
		samples[min_dif] = param_sample(t1, P1);
		// recursive calls
		for (int i = 0; i < min_dif; i++) {
			std::vector<vec2> app = discretizeParametricCurve(F,
				samples[i].t, samples[i + 1].t, samples[i].p, samples[i + 1].p,
				0, reqLength, reqError, recurse_remain - 1,
				i == 0 ? param_sample() : samples[i - 1],
				i + 1 == min_dif ? param_sample() : samples[i + 2]
			);
			res.insert(res.end(), app.begin(), app.end());
		}
		res.push_back(P1);
		delete samples;
		return res;
	}

	vec2 dP = P1 - P0;
	double dPL = length(dP);
	if (dPL < reqLength) {
		double tc = 0.5*(t0 + t1);  // experimental value
		//tc = t0 + selectArcLength(P_0, param_sample(t0, P0), param_sample(t1, P1), P_1, 0.5)*(t1 - t0);

		vec2 pc = F(tc);
		if (distSegment(P0, P1, pc) < reqError || true) {
			res.push_back(P0);
			res.push_back(pc);
		}
		else {
			std::vector<vec2> app0 = discretizeParametricCurve(F,
				t0, tc, P0, pc, 0, reqLength, reqError, recurse_remain - 1);
			std::vector<vec2> app1 = discretizeParametricCurve(F,
				tc, t1, pc, P1, 0, reqLength, reqError, recurse_remain - 1);
			res.insert(res.end(), app0.begin(), app0.end());
			res.insert(res.end(), app1.begin(), app1.end());
		}
		return res;
	}

#if 1
	double Th_low = 1.9, Th_high = 2.9;  // experimental value
	if (dPL > Th_low*reqLength && dPL < Th_high*reqLength) {  // split into 3
		double tc0 = t0 + (t1 - t0) / 3.;  // experimental value
		double tc1 = t1 - (t1 - t0) / 3.;  // experimental value

		vec2 pc0 = F(tc0), pc1 = F(tc1);
		std::vector<vec2> app0 = discretizeParametricCurve(F,
			t0, tc0, P0, pc0, 0, reqLength, reqError, recurse_remain - 1);
		std::vector<vec2> appc = discretizeParametricCurve(F,
			tc0, tc1, pc0, pc1, 0, reqLength, reqError, recurse_remain - 1);
		std::vector<vec2> app1 = discretizeParametricCurve(F,
			tc1, t1, pc1, P1, 0, reqLength, reqError, recurse_remain - 1);
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), appc.begin(), appc.end());
		res.insert(res.end(), app1.begin(), app1.end());
	}
	else
#endif
	{  // split into 2
		double tc = 0.5*(t0 + t1);  // experimental value
		vec2 pc = F(tc);
		std::vector<vec2> app0 = discretizeParametricCurve(F,
			t0, tc, P0, pc, 0, reqLength, reqError, recurse_remain - 1,
			P_0, param_sample(t1, P1));
		std::vector<vec2> app1 = discretizeParametricCurve(F,
			tc, t1, pc, P1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(t0, P0), P_1);
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), app1.begin(), app1.end());
	}

	return res;
}


int main(int argc, char* argv[]) {
	initSVG(argv[1]);

	for (int i = 0; i < 180; i++) {
		ParametricCurveL Curve = Cs[i];
		std::vector<vec2> points = discretizeParametricCurve(\
			[&](double t) { return Curve.p(t); }, \
			Curve.t0, Curve.t1, Curve.p(Curve.t0), Curve.p(Curve.t1), \
			10, 0.1, 0.01, 32);

		writeBlock(points);
		printf("%d\n", i);
	}

	endSVG();
	return 0;
}



// SVG writer
FILE* fp = 0;
const double scale = 128.;  // scaling from graph to svg
const int width = 640;  // width and height of each block
const int colspan = 1;  // number of blocks in a row
int blockCount = 0;  // number of blocks writed
void initSVG(const char* filename) {
	fp = fopen(filename, "wb");
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='%d' height='%d'>\n", width*colspan, 100 * width);
	fprintf(fp, "<defs><clipPath id='viewbox'><rect x='%d' y='%d' width='%d' height='%d'/></clipPath></defs>\n", 0, 0, width, width);
	fprintf(fp, "<style>text{font-family:Consolas;font-size:14px;}</style>\n");
}
void writeBlock(const std::vector<vec2> &points) {
	int vn = points.size();

	fprintf(fp, "<g transform='translate(%d,%d)' clip-path='url(#viewbox)'>\n",
		width*(blockCount%colspan), width*(blockCount / colspan));
	fprintf(fp, "<rect x='0' y='0' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", width, width);
	fprintf(fp, "<text x='10' y='20'>%d segments, %d samples</text>\n", vn, Parametric_callCount);  // should work

	// calculate arc length average and standard deviation
	double avrL = 0.;
	for (int i = 1; i < vn; i++)
		avrL += length(points[i] - points[i - 1]) / (vn - 1);
	double stdL = 0.;
	for (int i = 1; i < vn; i++)
		stdL += pow(length(points[i] - points[i - 1]) - avrL, 2.);
	stdL = sqrt(stdL / (vn - 1));
	fprintf(fp, "<text x='10' y='40'>avrL=%lf, stdL=%lf</text>\n", avrL, stdL);

	fprintf(fp, "<g transform='matrix(%lg,0,0,%lg,%lg,%lg)'>\n", scale, -scale, .5*width, .5*width);


	// segments (colored)
	fprintf(fp, "<g style='stroke-width:%lg'>", 1. / scale);
	for (int i = 1; i < vn; i++) {
		fprintf(fp, "<line x1='%lg' y1='%lg' x2='%lg' y2='%lg' stroke='%s'/>",
			points[i - 1].x, points[i - 1].y, points[i].x, points[i].y, i == 1 ? "green" : i & 1 ? "blue" : "red");
	}
	fprintf(fp, "</g>\n");

	// starting point
	fprintf(fp, "<circle cx='%lf' cy='%lf' r='%lf' style='stroke:none;fill:black;opacity:0.6;'/>\n",
		points[0].x, points[0].y, 3.0 / scale);

	fprintf(fp, "</g></g>");
	fflush(fp);
	blockCount++;
}
void endSVG() {
	fprintf(fp, "</svg>");
	fclose(fp);
}
