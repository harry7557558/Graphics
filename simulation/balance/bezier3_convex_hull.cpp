// find the convex hull of a path enclosed by cubic Bezier curve
// (almost) exact solution to the 2d balance problem

// *INCOMPLETE*

#include "numerical/geometry.h"
#include "numerical/rootfinding.h"
#include "numerical/random.h"

#include "path/svg_path_read.h"
typedef svg_path_read::bezier3 bezier3;
typedef std::vector<bezier3> object;
std::vector<object> Objs, Objs_CH;


void printBezier(bezier3 P) {
	printf("(1-t)^{3}*(%lg,%lg)+3*t*(1-t)^{2}*(%lg,%lg)+3*t^{2}*(1-t)*(%lg,%lg)+t^{3}*(%lg,%lg)\n", P.A.x, P.A.y, P.B.x, P.B.y, P.C.x, P.C.y, P.D.x, P.D.y);
}
void printPath(object u) {
	int un = u.size();
	for (int i = 0; i < un; i++) {
		bezier3 b = u[i];
		printf("M%lg,%lgC%lg,%lg %lg,%lg %lg,%lg", b.A.x, b.A.y, b.B.x, b.B.y, b.C.x, b.C.y, b.D.x, b.D.y);
	}
	printf("\n");
}

// solve det(P(t)-P0,P'(t))=0 for t
double clipPointToBezier(vec2 P0, bezier3 P, double t0 = 0., double t1 = 1.) {
	vec2 p0 = P.A - P0, p1 = -3.*P.A + 3.*P.B, p2 = 3.*P.A - 6.*P.B + 3.*P.C, p3 = -P.A + 3.*P.B - 3.*P.C + P.D;
	vec2 dp0 = P.B - P.A, dp1 = 2.*P.A - 4.*P.B + 2.*P.C, dp2 = -P.A + 3.*P.B - 3.*P.C + P.D;

	// det(p1,dp0), det(p2,dp1), det(p3,dp2) are zero
	// the product is a quartic polynomial
	double k[5];
	k[0] = det(p0, dp0);
	k[1] = det(p0, dp1);
	k[2] = det(p0, dp2) + det(p2, dp0) + det(p1, dp1);
	k[3] = det(p3, dp0) + det(p1, dp2);
	k[4] = det(p3, dp1) + det(p2, dp2);

	double r[4];
	int _rn = solveQuartic_dg(k[4], k[3], k[2], k[1], k[0], r);
	int rn = 0;
	for (int i = 0; i < _rn; i++)
		if (r[i] > 0 && r[i] < 1) r[rn++] = r[i];
	double minr = NAN;
	for (int i = 0; i < rn; i++)
		if (!(r[i] > minr)) minr = r[i];
	return minr;
}

// return true iff dot(P(t)-P0,n) is always non-negative
bool isCompletelyIndir(bezier3 P, vec2 n, vec2 P0) {
	double th = dot(n, P0) - 1e-8;
	if (isnan(th)) return false;
	double a = dot(P.A, n), b = dot(P.B, n), c = dot(P.C, n), d = dot(P.D, n);
	if (a < th || d < th) return false;
	double c0 = b - a, c1 = a - 2.*b + c, c2 = -a + 3.*b - 3.*c + d;
	double delta = c1 * c1 - c0 * c2;
	if (delta < 0.) return true;
	delta = sqrt(delta);
	double t = (delta - c1) / c2;
	if (t > 0. && t < 1.) {
		double u = a + t * (-3.*a + 3.*b + t * (3.*a - 6.*b + 3.*c + t * (-a + 3.*b - 3.*c + d)));
		if (u < th) return false;
	}
	t = (-delta - c1) / c2;
	if (t > 0. && t < 1.) {
		double u = a + t * (-3.*a + 3.*b + t * (3.*a - 6.*b + 3.*c + t * (-a + 3.*b - 3.*c + d)));
		if (u < th) return false;
	}
	return true;
}

// d
object clipSinglePath(bezier3 &c) {
	object res;
	if (isCompletelyIndir(c, (c.D - c.A).rot(), c.A)) {
		res.push_back(svg_path_read::fromSegment(c.A, c.D));
		return res;
	}
#if 0
	double t = clipPointToBezier(c.A, c);
	if (!isnan(t)) {
		vec2 p = c.eval(t);
		if (isCompletelyIndir(c, (p - c.A).rot(), c.A)) {
			res.push_back(svg_path_read::fromSegment(c.A, p));
			res.push_back(c.clipr(t));
			return res;
		}
	}
	t = clipPointToBezier(c.D, c);
	if (!isnan(t)) {
		vec2 p = c.eval(t);
		if (isCompletelyIndir(c, (c.D - p).rot(), c.D)) {
			res.push_back(c.clipl(t));
			res.push_back(svg_path_read::fromSegment(p, c.D));
			return res;
		}
	}
#endif
	res.push_back(c);
	return res;
}

// assume c1.D==c2.A
// return true if it is possible to close more curves behind
bool clipPath(bezier3 c1, bezier3 c2, object &res) {
	using namespace svg_path_read;

	printBezier(c1); printBezier(c2);

	// check if clip at endpoints
	vec2 n = (c2.D - c1.A).rot();
	bool cid1 = isCompletelyIndir(c1, n, c1.A);
	bool cid2 = isCompletelyIndir(c2, n, c2.D);

	// connect using a segment
	if (cid1 && cid2) {
		bezier3 s = fromSegment(c1.A, c2.D);
		res.push_back(s); return true;
	}

	double t1 = clipPointToBezier(c2.D, c1);
	double t2 = clipPointToBezier(c1.A, c2);
	vec2 p1 = isnan(t1) ? vec2(NAN) : c1.eval(t1);
	vec2 p2 = isnan(t2) ? vec2(NAN) : c2.eval(t2);


	// cut one
	if (isCompletelyIndir(c1, (p2 - c1.A).rot(), c1.A)) {
		double t = clipPointToBezier(c1.A, c2);
		if (isnan(t)) throw(__LINE__);  // bug
		res.push_back(fromSegment(c1.A, c2.eval(t)));
		res.push_back(c2.clipr(t));
		return true;
	}
	if (isCompletelyIndir(c2, (p1 - c2.D).rotr(), c2.D)) {
		double t = clipPointToBezier(c2.D, c1);
		if (isnan(t)) throw(__LINE__);  // bug
		res.push_back(c1.clipl(t));
		res.push_back(fromSegment(c1.eval(t), c2.D));
		return false;
	}

	// cut two (doesn't always work)
	if (isnan(t1) && isnan(t2)) {
		res.push_back(c1);
		res.push_back(c2);
		return false;
	}
	if (isnan(t1)) t1 = clipPointToBezier(p2, c1), p1 = c1.eval(t1);
	if (isnan(t2)) t2 = clipPointToBezier(p1, c2), p2 = c1.eval(t2);
	for (int i = 0; i < 20; i++) {
		t1 = clipPointToBezier(p2, c1);
		p1 = c1.eval(t1);
		t2 = clipPointToBezier(p1, c2);
		p2 = c2.eval(t2);
	}
	if (isnan(t1) || isnan(t2)) {
		//throw(__LINE__);
		//return false;
		res.push_back(c1);
		res.push_back(c2);
	}
	else {
		res.push_back(c1.clipl(t1));
		res.push_back(fromSegment(p1, p2));
		res.push_back(c2.clipr(t2));
		return false;
	}
}


void clipBack(object &C) {
	while (C.size() >= 2) {
		bezier3 c2 = C[C.size() - 1];
		bezier3 c1 = C[C.size() - 2];
		object app;
		if (!clipPath(c1, c2, app)) {
			C.pop_back(); C.pop_back();
			C.insert(C.end(), app.begin(), app.end());
			break;
		}
		if (app.empty()) {
			break;
		}
		else {
			C.pop_back(); C.pop_back();
			if (!C.empty()) {
				for (int i = 0, l = app.size(); i < l; i++) {
					C.push_back(app[i]);
					clipBack(C);
				}
			}
			else {
				C.insert(C.end(), app.begin(), app.end());
			}
			break;
		}
	}
}


// assume obj is a good conditioned shape
// (connected, enclosed, counter-clockwise, no self-intersection)
object convexHull(object P) {
	object C;
	for (int i = 0, Pn = P.size(); i < Pn; i++) {
		bezier3 Pi = P[i];
		object app = clipSinglePath(Pi);

		for (int i = 0, l = app.size(); i < l; i++) {
			C.push_back(app[i]);
			clipBack(C);
		}

		printPath(C);
	}
	for (int i = 0; i < 2; i++) {
		C.push_back(C[0]);
		C.erase(C.begin());
		clipBack(C);
	}

	return C;
}









int loadObjs(const char* filename);  // load test objects from file
void writeObjs(const char* filename);  // write calculation results to file

int main(int argc, char* argv[]) {
	int N = loadObjs(argv[1]);
	Objs_CH = Objs;

	for (int i = 0; i < 24; i++) {
		//for (int i = 16; i < 17; i++) {
		object v = Objs[i];
		int vn = v.size();

		for (int i = 0; i < vn; i++) {
			printBezier(v[i]);
		}
		object u = convexHull(v);
		Objs_CH[i] = u;
	}

	writeObjs(argv[2]);
	return 0;
}



std::vector<std::string> shape_str;
std::vector<vec2> shape_translate;

// maximize the dot product of a point on the shape
double maxDot(const object &obj, vec2 n) {
	double maxE = -INFINITY;
	for (int i = 0, N = obj.size(); i < N; i++) {
		double a = dot(obj[i].A, n), b = dot(obj[i].B, n), c = dot(obj[i].C, n), d = dot(obj[i].D, n);
		maxE = max(maxE, max(a, d));
		double c0 = b - a, c1 = a - 2.*b + c, c2 = -a + 3.*b - 3.*c + d;
		double delta = c1 * c1 - c0 * c2;
		if (delta >= 0.) {
			delta = sqrt(delta);
			double t = (delta - c1) / c2;
			if (t > 0. && t < 1.) maxE = max(maxE, a + t * (3.*b - 3.*a + t * (3.*a - 6.*b + 3.*c + t * (-a + 3.*b - 3.*c + d))));
			t = (-delta - c1) / c2;
			if (t > 0. && t < 1.) maxE = max(maxE, a + t * (3.*b - 3.*a + t * (3.*a - 6.*b + 3.*c + t * (-a + 3.*b - 3.*c + d))));
		}
	}
	return maxE;
}

// read objects from text file (well-formatted SVG)
#include <fstream>
int loadObjs(const char* filename) {
	std::ifstream ifs(filename, std::ios_base::in);
	std::string s; getline(ifs, s);
	while (true) {
		getline(ifs, s);
		int b1 = s.find('"', 0), b2 = s.find('"', b1 + 1);
		if (b1 == -1) break;
		shape_str.push_back(s = s.substr(b1 + 1, b2 - b1 - 1));
		object obj; svg_path_read::parse_path(s, obj);
		vec2 MaxVal(maxDot(obj, vec2(1, 0)), maxDot(obj, vec2(0, 1)));
		vec2 MinVal(maxDot(obj, vec2(-1, 0)), maxDot(obj, vec2(0, -1)));
		vec2 C = 0.5*(MaxVal - MinVal);
		for (int i = 0, n = obj.size(); i < n; i++)
			obj[i].translate(-C);
		if (svg_path_read::calcArea(obj) < 0.) svg_path_read::reversePath(obj);
		Objs.push_back(obj);
		shape_translate.push_back(C);
	}
	ifs.close();
	return Objs.size();
}

void writeObjs(const char* filename) {
	FILE* fp = fopen(filename, "wb");
	const int colspan = 1;  // colspan of the graph
	const int W = 400;  // with and height of sub-graphs (square)
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>\n", colspan*W, ((Objs_CH.size() - 1) / colspan + 1)*W + 80);
	fprintf(fp, "<style>rect{stroke-width:1px;stroke:black;fill:white;}path{stroke-width:1px;fill:none;}</style>\n");
	for (int T = 0, L = 24; T < L; T++) {
		object P = Objs_CH[T]; int PN = P.size();
		vec2 Tr = shape_translate[T];
		fprintf(fp, "<g transform='translate(%d,%d)'>\n", W*(T%colspan), W*(T / colspan));
		fprintf(fp, "<rect x='0' y='0' width='%d' height='%d'/>\n", W, W);
		fprintf(fp, "<path transform='translate(%.1lf %.1lf)' d='%s' stroke='gray'/>\n", .5*W - Tr.x, .5*W - Tr.y, &shape_str[T][0]);
		fprintf(fp, "<path transform='translate(%.1lf %.1lf)' d='\n", .5*W, .5*W);
		fprintf(fp, "M%lg,%lg", P[0].A.x, P[0].A.y);
		for (int i = 0; i < PN; i++) {
			bezier3 Vi = P[i];
			fprintf(fp, "C%lg,%lg %lg,%lg %lg,%lg", Vi.B.x, Vi.B.y, Vi.C.x, Vi.C.y, Vi.D.x, Vi.D.y);
		}
		fprintf(fp, "Z' stroke='black'/>\n");
		fprintf(fp, "<circle cx='%lg' cy='%lg' r='5' style='fill:red;'/>\n", P[0].A.x + .5*W, P[0].A.y + .5*W);
		fprintf(fp, "</g>\n");
	}
	fprintf(fp, "</svg>");
	fclose(fp);
}

