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
void writeBlock(const std::vector<segment_sample> &points);
void endSVG();



inline bool isNAV(vec2 p) {
	double m = p.sqr();
	return isnan(m) || m > 1e32;
}

// P(t0) is NAV and P(t1) is not, find the t closest to t0 such that P(t) is not NAV
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
template<typename Fun>
std::vector<segment_sample> discretizeParametricCurve(Fun F,
	double t0, double t1, vec2 P0, vec2 P1,
	int min_dif, double reqLength, double reqError, int recurse_remain,
	param_sample P_0 = param_sample(), param_sample P_1 = param_sample()) {

	std::vector<segment_sample> res;
	if (recurse_remain < 0) return res;

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
			//res.push_back(segment_sample{ param_sample(t, samples[i].p - vec2(0,.1)), param_sample(t, samples[i].p + vec2(0,.1)) });
		}
		samples[0] = param_sample(t0, P0);
		samples[min_dif] = param_sample(t1, P1);

		// recursive calls
		for (int i = 0; i < min_dif; i++) {
			std::vector<segment_sample> app = discretizeParametricCurve(_F,
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
	param_sample s0{ t0, P0 }, s1{ t1, P1 };

	// continue subdivision until error is small enough
	vec2 dP = P1 - P0;
	double dPL = length(dP);
	if (dPL == 0.0) return res;
	if (dPL < reqLength || min_dif == -1) {

		// split at the point(s) that produce the maximum value
		vec2 n = (P1 - P0).rot();
		double x[4] = { P_0.t, t0, t1, P_1.t };
		double y[4] = { dot(n, P_0.p), dot(n, P0), dot(n, P1), dot(n, P_1.p) };
		vec2 tcp = cubicInterpolation_max(x, y);
		if (isnan(tcp.x)) {
			tcp = vec2(0.5*(t0 + t1), NAN);
			//tcp = (1. / 3.)*(vec2(2., 1.)*t0 + vec2(1., 2.)*t1);
		}
		//tcp.y = NAN;

		if (isnan(tcp.y)) {  // divide into 2
			tc = tcp.x; pc = F(tc);
			param_sample sc{ tc, pc };
			if (distSegment(P0, P1, pc) < reqError) {
				res.push_back(segment_sample{ s0, sc });
				res.push_back(segment_sample{ sc, s1 });
				return res;
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
		double Th_low = 1.9, Th_high = 2.9;  // experimental values
		if (dPL > Th_low*reqLength && dPL < Th_high*reqLength) {  // divide into 3
			tc0 = t0 + (t1 - t0) / 3.;  // experimental value
			tc1 = t1 - (t1 - t0) / 3.;  // experimental value
			pc0 = F(tc0), pc1 = F(tc1);
			goto divide3;
		}
		else {  // divide into 2
			tc = 0.5*(t0 + t1);  // experimental value
			pc = F(tc);
			goto divide2;
		}
	}

	// should never get here
	throw("bug");
	return res;


divide2:
	{
		// split into 2
		std::vector<segment_sample> app0 = discretizeParametricCurve(F,
			t0, tc, P0, pc, 0, reqLength, reqError, recurse_remain - 1,
			P_0, param_sample(t1, P1));
		std::vector<segment_sample> app1 = discretizeParametricCurve(F,
			tc, t1, pc, P1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(t0, P0), P_1);
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
						std::vector<segment_sample> tmp = discretizeParametricCurve(F,
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
		std::vector<segment_sample> app0 = discretizeParametricCurve(F,
			t0, tc0, P0, pc0, 0, reqLength, reqError, recurse_remain - 1,
			P_0, param_sample(tc1, pc1));
		std::vector<segment_sample> appc = discretizeParametricCurve(F,
			tc0, tc1, pc0, pc1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(t0, P0), param_sample(t1, P1));
		std::vector<segment_sample> app1 = discretizeParametricCurve(F,
			tc1, t1, pc1, P1, 0, reqLength, reqError, recurse_remain - 1,
			param_sample(tc0, pc0), P_1);
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), appc.begin(), appc.end());
		res.insert(res.end(), app1.begin(), app1.end());
		return res;
	}

}





int main(int argc, char* argv[]) {
	initSVG(argv[1]);

	for (int i = 0; i < 182; i++) {
		Parametric_callCount = 0;
		ParametricCurveL Curve = Cs[i];
		std::vector<segment_sample> points = discretizeParametricCurve(
			[&](double t) {
			vec2 p = Curve.p(t);
			return abs(p.x) < 2.5 && abs(p.y) < 2.5 ? p : vec2(NAN);
		},
			Curve.t0, Curve.t1, Curve.p(Curve.t0), Curve.p(Curve.t1),
			12, 0.1, 0.001, 12);

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
const int colspan = 2;  // number of blocks in a row
int blockCount = 0;  // number of blocks writed
void initSVG(const char* filename) {
	fp = fopen(filename, "wb");
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='%d' height='%d'>\n", width*colspan, 180 * width);
	fprintf(fp, "<defs><clipPath id='viewbox'><rect x='%d' y='%d' width='%d' height='%d'/></clipPath></defs>\n", 0, 0, width, width);
	fprintf(fp, "<style>text{font-family:Consolas;font-size:14px;}</style>\n");
}
void writeBlock(const std::vector<segment_sample> &points) {
	int vn = points.size();

	fprintf(fp, "<g transform='translate(%d,%d)' clip-path='url(#viewbox)'>\n",
		width*(blockCount%colspan), width*(blockCount / colspan));
	fprintf(fp, "<rect x='0' y='0' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", width, width);
	fprintf(fp, "<text x='10' y='20'>%d segments, %d samples</text>\n", vn, Parametric_callCount);  // should work

	// calculate arc length average and standard deviation
	double avrL = 0.;
	for (int i = 0; i < vn; i++)
		avrL += length(points[i].q.p - points[i].p.p) / vn;
	double stdL = 0.;
	for (int i = 0; i < vn; i++)
		stdL += pow(length(points[i].q.p - points[i].p.p) - avrL, 2.);
	stdL = sqrt(stdL / (vn - 1));
	fprintf(fp, "<text x='10' y='40'>avrL=%lf, stdL=%lf</text>\n", avrL, stdL);

	fprintf(fp, "<g transform='matrix(%lg,0,0,%lg,%lg,%lg)'>\n", scale, -scale, .5*width, .5*width);


	// segments (colored)
	fprintf(fp, "<g style='stroke-width:%lg'>", 1. / scale);
	for (int i = 0; i < vn; i++) {
		fprintf(fp, "<line x1='%lg' y1='%lg' x2='%lg' y2='%lg' stroke='%s'/>",
			points[i].p.p.x, points[i].p.p.y, points[i].q.p.x, points[i].q.p.y, i == 0 ? "green" : i & 1 ? "red" : "blue");
	}
	fprintf(fp, "</g>\n");

	// starting points
	vec2 p0(NAN);
	for (int i = 0; i < vn; i++) {
		if (points[i].p.p != p0)
			fprintf(fp, "<circle cx='%lf' cy='%lf' r='%lf' style='stroke:none;fill:black;opacity:0.3;'/>\n",
				points[i].p.p.x, points[i].p.p.y, 3.0 / scale);
		p0 = points[i].q.p;
	}

	fprintf(fp, "</g></g>");
	fflush(fp);
	blockCount++;
}
void endSVG() {
	fprintf(fp, "</svg>");
	fclose(fp);
}









// not-working code

#if 0

// use a binary search tree for interpolating parameter samples
// To-do: avoid worst case runtime (happens with certain functions)
struct BST_node {
	double t0, t1;  // starting and ending parameters
	double t; vec2 p;  // parameter and value
	BST_node *parent = 0, *c1 = 0, *c2 = 0;  // parent and children
	int size = 0;  // not actually used
	double t_min, t_max;  // minimum and maximum t in the tree

	static void initNode(BST_node* &T, double t, vec2 p, double t0, double t1, BST_node* parent = nullptr) {
		if (!T) T = new BST_node;
		T->t = t, T->p = p;
		T->c1 = T->c2 = nullptr;
		T->t0 = t0, T->t1 = t1;
		T->parent = parent;
		T->size = 1;
		T->t_min = T->t_max = t;
	}
	void addNode(double t, vec2 p) {
		if (this->t == t) return;  // no duplicates!!!
		size++;
		double tc = .5*(t0 + t1);  // where to split
		if (c1 == 0 && c2 == 0) {  // move the value at the current node
			if (this->t < tc && t < tc)  // all in c1
				initNode(c1, this->t, this->p, t0, tc, this),
				c1->addNode(t, p);
			else if (this->t >= tc && t >= tc)  // all in c2
				initNode(c2, this->t, this->p, tc, t1, this),
				c2->addNode(t, p);
			else if (this->t < tc)  // this in c1 and t in c2
				initNode(c1, this->t, this->p, t0, tc, this),
				initNode(c2, t, p, tc, t1, this);
			else if (t < tc)  // t in c1 and this in c2
				initNode(c1, t, p, t0, tc, this),
				initNode(c2, this->t, this->p, tc, t1, this);
			else throw("bug");
			this->t = NAN, this->p = vec2(NAN);
		}
		if (t < tc) {  // add to c1
			if (c1) c1->addNode(t, p);
			else initNode(c1, t, p, t0, tc, this);
		}
		else {  // add to c2
			if (c2) c2->addNode(t, p);
			else initNode(c2, t, p, tc, t1, this);
		}
		t_min = min(t_min, t);
		t_max = max(t_max, t);
	}

	const BST_node* getBefore(double t) const {
		if (t <= t_min) return nullptr;
		if (!c1 && !c2) return t <= this->t ? nullptr : this;
		if (!c2 || t <= c2->t_min) return c1->getBefore(t);
		return c2->getBefore(t);
	}
	const BST_node* getAfter(double t) const {
		if (t >= t_max) return nullptr;
		if (!c1 && !c2) return t >= this->t ? nullptr : this;
		if (!c1 || t >= c1->t_max) return c2->getAfter(t);
		return c1->getAfter(t);
	}

	static vec2 getInterpolateLinear(const BST_node *N, double t) {
		const BST_node* n0 = N->getBefore(t);
		const BST_node* n1 = N->getAfter(t);
		double t0 = n0 ? n0->t : n1->t;
		double t1 = n1 ? n1->t : n0->t;
		vec2 p0 = n0 ? n0->p : n1->p;
		vec2 p1 = n1 ? n1->p : n0->p;
		return mix(p0, p1, (t - t0) / (t1 - t0));
	}
	static vec2 getInterpolate(const BST_node *N, double t, vec2 *grad = nullptr) {
		const BST_node* n0 = N->getBefore(t);
		const BST_node* n1 = N->getAfter(t);
		double t0 = n0 ? n0->t : n1->t;
		double t1 = n1 ? n1->t : n0->t;
		const BST_node* n00 = N->getBefore(t0);
		const BST_node* n11 = N->getAfter(t1);
		double t00 = n00 ? n00->t : t0;
		double t11 = n11 ? n11->t : t1;
		vec2 p0 = n0 ? n0->p : n1->p;
		vec2 p1 = n1 ? n1->p : n0->p;
		vec2 p00 = n00 ? n00->p : p0;
		vec2 p11 = n11 ? n11->p : p1;
		if (t0 == t1) {
			if (t1 != t11) t1 = t11, p1 = p11;
			if (t0 != t00) t0 = t00, p0 = p00;
		}
		vec2 d0 = (p1 - p00) / (t1 - t00) * (t1 - t0);
		vec2 d1 = (p11 - p0) / (t11 - t0) * (t1 - t0);
		t = (t - t0) / (t1 - t0);
		double t2 = t * t, t3 = t2 * t;
		if (grad) *grad = (6.*t2 - 6.*t)*p0 + (3.*t2 - 4.*t + 1.)*d0 + (-6.*t2 + 6.*t)*p1 + (3.*t2 - 2.*t)*d1;
		return (2.*t3 - 3.*t2 + 1.)*p0 + (t3 - 2.*t2 + t)*d0 + (-2.*t3 + 3.*t2)*p1 + (t3 - t2)*d1;
	}

};


template<typename Fun>
std::vector<segment> discretizeParametricCurve_BST(Fun F,
	double t0, double t1, vec2 P0, vec2 P1,
	int min_dif, double reqLength, double reqError, int recurse_remain,
	BST_node *BST = nullptr) {

	std::vector<segment> res;
	if (recurse_remain < 0) return res;


	// user call
	if (!BST) {
		// map parameter to [0,1]
		std::function<vec2(double)> _F = [&](double t) { return F(t); };
		if (!(t0 == 0 && t1 == 1)) {
			if (abs(t1 - t0) > 1e4) {  // INF case
				_F = [&](double t) { return F(tan(t)); };
				t0 = atan(t0), t1 = atan(t1);
			}
			double _t0 = t0, _dt = t1 - t0;
			_F = [&](double t) { return F(_t0 + _dt * t); };
			t0 = 0., t1 = 1.;
		}

		reqLength *= 2.0 * 1.3;  // hmmm...
		if (!(reqLength > 0. && reqError > 0.)) return res;  // ...

		// take samples
		double dt = (t1 - t0) / min_dif;
		param_sample *samples = new param_sample[min_dif + 1];
		for (int i = 1; i < min_dif; i++) {
			double t = t0 + (i + 0.01*sin(123.456*i)) * dt;
			samples[i] = param_sample(t, _F(t));
		}
		samples[0] = param_sample(t0, P0);
		samples[min_dif] = param_sample(t1, P1);

		BST_node::initNode(BST, t0, P0, 0., 1.);
		for (int i = 1; i <= min_dif; i++)
			BST->addNode(samples[i].t, samples[i].p);

		//BST_node *BST_REF = new BST_node;
		//BST_node::initNode(BST_REF, t0, P0, 0., 1.);
		//for (int i = 1; i <= min_dif; i++)
		//	BST_REF->addNode(samples[i].t, samples[i].p);
		//_F = [&](double t) { return BST_REF->getInterpolate(BST_REF, t); };

		// recursive calls
		for (int i = 0; i < min_dif; i++) {
			std::vector<segment> app = discretizeParametricCurve_BST(_F,
				samples[i].t, samples[i + 1].t, samples[i].p, samples[i + 1].p,
				0, reqLength, reqError, recurse_remain - 1, BST);
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
	}
	else if (isNAV(P1)) {
		if (!bisectNAV(F, t1, t0, P1, P0, t1, P1)) return res;
	}

	// continue subdivision until error is small enough
	vec2 dP = P1 - P0;
	double dPL = length(dP);
	if (dPL == 0.0) return res;
	if (dPL < reqLength || min_dif == -1) {

		// split at the point(s) that produce the maximum value
		vec2 n = (P1 - P0).rot();
		vec2 p0, p1, d0, d1;
		p0 = BST->getInterpolate(BST, t0, &d0); p0 = P0;
		p1 = BST->getInterpolate(BST, t1, &d1); p1 = P1;
		vec2 tcp = cubicInterpolation_max(dot(p0, n), dot(p1, n), dot(d0, n), dot(d1, n));
		tcp = vec2(t0) + (t1 - t0)*tcp;
		//tcp = vec2(NAN);
		if (isnan(tcp.x)) tcp.x = 0.5*(t0 + t1);

		// split into 2
		if (isnan(tcp.y)) {
			double tc = tcp.x; vec2 pc = F(tc);
			BST->addNode(tc, pc);
			if (distSegment(P0, P1, pc) < reqError) {
				res.push_back(segment{ P0, pc });
				res.push_back(segment{ pc, P1 });
			}
			else {
				std::vector<segment> app0 = discretizeParametricCurve_BST(F,
					t0, tc, P0, pc, 0, reqLength, reqError, recurse_remain - 1, BST);
				std::vector<segment> app1 = discretizeParametricCurve_BST(F,
					tc, t1, pc, P1, 0, reqLength, reqError, recurse_remain - 1, BST);
				res.insert(res.end(), app0.begin(), app0.end());
				res.insert(res.end(), app1.begin(), app1.end());
			}
			return res;
		}

		// divide into 3
		else {
			double tc0 = tcp.x, tc1 = tcp.y;
			vec2 pc0 = F(tc0), pc1 = F(tc1);
			BST->addNode(tc0, pc0); BST->addNode(tc1, pc1);
			if (distSegment(P0, P1, pc0) < reqError && distSegment(P0, P1, pc1) < reqError) {
				res.push_back(segment{ P0, pc0 });
				res.push_back(segment{ pc0, pc1 });
				res.push_back(segment{ pc1, P1 });
			}
			else {
				// P_0, P_1: may have better guesses
				std::vector<segment> app0 = discretizeParametricCurve_BST(F,
					t0, tc0, P0, pc0, 0, reqLength, reqError, recurse_remain - 1, BST);
				std::vector<segment> appc = discretizeParametricCurve_BST(F,
					tc0, tc1, pc0, pc1, 0, reqLength, reqError, recurse_remain - 1, BST);
				std::vector<segment> app1 = discretizeParametricCurve_BST(F,
					tc1, t1, pc1, P1, 0, reqLength, reqError, recurse_remain - 1, BST);
				res.insert(res.end(), app0.begin(), app0.end());
				res.insert(res.end(), appc.begin(), appc.end());
				res.insert(res.end(), app1.begin(), app1.end());
			}
			return res;
		}

	}

	// divide into 3
	double Th_low = 1.9, Th_high = 2.9;  // experimental values
	if (dPL > Th_low*reqLength && dPL < Th_high*reqLength) {
		double tc0 = t0 + (t1 - t0) / 3.;  // experimental value
		double tc1 = t1 - (t1 - t0) / 3.;  // experimental value

		vec2 pc0 = F(tc0), pc1 = F(tc1);
		BST->addNode(tc0, pc0), BST->addNode(tc1, pc1);
		std::vector<segment> app0 = discretizeParametricCurve_BST(F,
			t0, tc0, P0, pc0, 0, reqLength, reqError, recurse_remain - 1, BST);
		std::vector<segment> appc = discretizeParametricCurve_BST(F,
			tc0, tc1, pc0, pc1, 0, reqLength, reqError, recurse_remain - 1, BST);
		std::vector<segment> app1 = discretizeParametricCurve_BST(F,
			tc1, t1, pc1, P1, 0, reqLength, reqError, recurse_remain - 1, BST);
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), appc.begin(), appc.end());
		res.insert(res.end(), app1.begin(), app1.end());
	}

	// divide into 2
	else {
		double tc = 0.5*(t0 + t1);  // experimental value
		vec2 pc = F(tc);
		BST->addNode(tc, pc);
		double t0c = tc, t1c = tc;
		vec2 p0c = pc, p1c = pc;
		std::vector<segment> app0 = discretizeParametricCurve_BST(F,
			t0, t0c, P0, p0c, 0, reqLength, reqError, recurse_remain - 1, BST);
		std::vector<segment> app1 = discretizeParametricCurve_BST(F,
			t1c, t1, p1c, P1, 0, reqLength, reqError, recurse_remain - 1, BST);
		res.insert(res.end(), app0.begin(), app0.end());
		res.insert(res.end(), app1.begin(), app1.end());
	}

	return res;
}

#endif
