// find the convex hull of a path enclosed by cubic Bezier curve

// as a temporary experiment to find the exact solution to the 2d balance problem,
// (use to compare numerical solutions,)
// does not work for all cases

#include "numerical/geometry.h"
#include "numerical/rootfinding.h"
#include "numerical/random.h"

#include "path/svg_path_read.h"
typedef svg_path_read::bezier3 bezier3;
typedef std::vector<bezier3> object;
std::vector<object> Objs, Objs_CH;

#define _REPORT_BUG throw(__LINE__)

// debug outputs
void printBezier(bezier3 P) {
	printf("(1-t)^3*(%lg,%lg)+3*t*(1-t)^2*(%lg,%lg)+3*t^2*(1-t)*(%lg,%lg)+t^3*(%lg,%lg)\n", P.A.x, P.A.y, P.B.x, P.B.y, P.C.x, P.C.y, P.D.x, P.D.y);
}
void printPath(bezier3 P) {
	printf("M%lg,%lgC%lg,%lg %lg,%lg %lg,%lg\n", P.A.x, P.A.y, P.B.x, P.B.y, P.C.x, P.C.y, P.D.x, P.D.y);
}
void printPath(object u, const char* end = "\n", FILE* fp = stdout) {
	int un = u.size();
	for (int i = 0; i < un; i++) {
		bezier3 b = u[i];
		fprintf(fp, "M%lg,%lgC%lg,%lg %lg,%lg %lg,%lg", b.A.x, b.A.y, b.B.x, b.B.y, b.C.x, b.C.y, b.D.x, b.D.y);
	}
	fprintf(fp, end);
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
	//int _rn = solveQuartic_dg(k[4], k[3], k[2], k[1], k[0], r);
	int _rn = solvePolynomial_bisect(4, k, r, 0., 1., 1e-12);  // numerically stable
	int rn = 0;
	for (int i = 0; i < _rn; i++) {
		if (r[i] > 1e-4 && r[i] < 1. - 1e-4) r[rn++] = r[i];
	}
	if (rn == 0) return NAN;
	return r[0];
}

// return true iff dot(P(t)-P0,n) is always non-negative
// do not normalize n to get the (expected?) result
bool isCompletelyIndir(bezier3 P, vec2 n, vec2 P0) {
	double th = dot(n, P0) - 1e-6 * n.sqr();
	if (isnan(th)) return false;
	if (!(n.sqr() > 1e-16)) return false;  // what

	double a = dot(P.A, n), b = dot(P.B, n), c = dot(P.C, n), d = dot(P.D, n);
	if (a < th || d < th) return false;
	double c0 = b - a, c1 = a - 2.*b + c, c2 = -a + 3.*b - 3.*c + d;
	if (abs(c2) < 1e-8) {
		double t = -c0 / (2.*c1);
		if (t > 0. && t < 1.) {
			double u = a + t * (-3.*a + 3.*b + t * (3.*a - 6.*b + 3.*c + t * (-a + 3.*b - 3.*c + d)));
			if (u < th) return false;
		}
		return true;
	}
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

// give a cubic Bezier curve, find the shortest ccw convex path from start to end
object clipSinglePath(bezier3 &c) {
	// not sure if this always works

	object res;
	if (isCompletelyIndir(c, (c.D - c.A).rot(), c.A)) {
		res.push_back(svg_path_read::fromSegment(c.A, c.D));
		return res;
	}

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

	res.push_back(c);
	return res;
}

// give two ccw cubic Bezier curves, find the shortest ccw convex path from the start of the first curve to the end of the second curve
// return value: the lowest bit is true if the start of c1 changed and the second bit is true if the end of c2 changed
int clipPath(bezier3 c1, bezier3 c2, object &res) {
	using namespace svg_path_read;

	//printBezier(c1); printBezier(c2);

	// check if clip at endpoints
	vec2 n = (c2.D - c1.A).rot();
	bool cid1 = isCompletelyIndir(c1, n, c1.A);
	bool cid2 = isCompletelyIndir(c2, n, c2.D);

	// connect using a segment
	if (cid1 && cid2) {
		bezier3 s = fromSegment(c1.A, c2.D);
		res.push_back(s); return 0b11;
	}

	// check endpoint tangents, expected to work because c1 and c2 are ccw
	if (ndet(c1.D - c1.C, c2.B - c2.A) > -1e-4) {
		res.push_back(c1); res.push_back(c2);
		return 0b00;
	}

	double t1 = clipPointToBezier(c2.D, c1);
	double t2 = clipPointToBezier(c1.A, c2);
	vec2 p1 = isnan(t1) ? vec2(NAN) : c1.eval(t1);
	vec2 p2 = isnan(t2) ? vec2(NAN) : c2.eval(t2);

	// cut one
	if (isCompletelyIndir(c1, (p2 - c1.A).rot(), c1.A)) {
		res.push_back(fromSegment(c1.A, p2));
		res.push_back(c2.clipr(t2));
		return 0b01;
	}
	if (isCompletelyIndir(c2, (p1 - c2.D).rotr(), c2.D)) {
		res.push_back(c1.clipl(t1));
		res.push_back(fromSegment(p1, c2.D));
		return 0b10;
	}

	// cut two
	// no guarantee this will always work
	t1 = 0.95, p1 = c1.eval(t1);
	t2 = 0.05, p2 = c2.eval(t2);
	double _t1 = t1, _t2 = t2, _dt = NAN;
	for (int i = 0; i < 10; i++) {
		t1 = clipPointToBezier(p2, c1);
		p1 = c1.eval(t1);
		if (isnan(t1)) break;
		t2 = clipPointToBezier(p1, c2);
		p2 = c2.eval(t2);
		if (isnan(t2)) break;
		double dt = max(abs(t1 - _t1), abs(t2 - _t2));
		//printf("%lf\n", log10(_dt / dt));  // quadratic convergence looks like
		if (dt < 1e-12) break;
		_t1 = t1, _t2 = t2, _dt = dt;
	}
	if (isnan(t1) || isnan(t2)) {
		// most due to floating-point inaccuracy
		// note that the test cases come from SVG path that some decimal points are truncated
		res.push_back(c1);
		res.push_back(c2);
		return 0b00;
	}
	else {
		res.push_back(c1.clipl(t1));
		res.push_back(fromSegment(p1, p2));
		res.push_back(c2.clipr(t2));
		return 0b00;
	}
}


// check the back of a path and make it go counter-clockwise
void clipBack(object &C) {
	if (C.size() >= 2) {
		bezier3 c2 = C[C.size() - 1];
		bezier3 c1 = C[C.size() - 2];
		object app;
		if (!(clipPath(c1, c2, app) & 0b01)) {
			C.pop_back(); C.pop_back();
			C.insert(C.end(), app.begin(), app.end());
			return;
		}

		C.pop_back(); C.pop_back();
		if (C.empty()) {
			C = app; return;
		}

		for (int i = 0, l = app.size(); i < l; i++) {
			C.push_back(app[i]);
			clipBack(C);
		}
	}
}


// path stored as a linked list
struct obj_node {
	bezier3 c;
	obj_node *lst = 0, *nxt = 0;
};
obj_node* object2list(const object &C) {
	obj_node *p = new obj_node, *q = p;
	for (int i = 0, l = C.size(); i < l; i++) {
		q->c = C[i];
		q->nxt = i + 1 == l ? p : new obj_node;
		q->nxt->lst = q;
		q = q->nxt;
	}
	return p;
}
object list2object(obj_node* p) {
	// list will be destroyed
	object C;
	obj_node* p0 = p;
	do {
		C.push_back(p->c);
		obj_node *q = p;
		p = p->nxt;
		delete q;
	} while (p != p0);
	return C;
}

// check the vertex between p and p->lst and make all neighborhood paths go ccw
// returns any node in the result list (the parameter node p may be deleted)
obj_node* clipList(obj_node *p, std::vector<obj_node*> *delete_list = nullptr) {

	// store the "garbage" elements and "recycle" them later
	// this is necessary because a recursive call checks whether a node is deleted by checking if its lst or nxt is nullptr
	// seems like there is still memory leak somewhere... :(
	if (delete_list == nullptr) {
		delete_list = new std::vector<obj_node*>;
		p = clipList(p, delete_list);
		for (int i = 0, l = delete_list->size(); i < l; i++)
			delete delete_list->at(i);
		delete delete_list;
		return p;
	}
	auto delete_node = [&](obj_node* p) {
		p->lst = p->nxt = 0;
		delete_list->push_back(p);
	};

	// clip the neighbor around
	bezier3 c1 = p->lst->c, c2 = p->c;
	object ins;
	int u = clipPath(c1, c2, ins);  // store the information about whether it can be clipped further

	// insert the array to the linked list
	obj_node *p0 = p->lst->lst, *p1 = p->nxt;
	delete_node(p->lst); delete_node(p);  // to be deleted later
	p = p0;
	for (int i = 0, l = ins.size(); i < l; i++) {
		p->nxt = new obj_node;
		p->nxt->lst = p;
		p = p->nxt; p->c = ins[i];
	}
	p->nxt = p1, p1->lst = p;

	// check further
	if (u == 0b01) {
		return clipList(p0->nxt, delete_list);
	}
	if (u == 0b10) {
		return clipList(p1, delete_list);
	}
	if (u == 0b11) {
		obj_node* p_new = clipList(p0->nxt, delete_list);
		if (p1->lst && p1->nxt) return clipList(p1, delete_list);
		return p_new;
	}
	return p;
}



// calculate the angle turned of a ccw path, assume each individual piece are non-self-intersecting
double angleTurned(const object &C) {
	//for (int i = 0, l = C.size(); i < l; i++) printBezier(C[i]);
	double a = 0;
	for (int i = 0, l = C.size(); i < l; i++) {
		vec2 d1 = normalize(C[i].B - C[i].A);
		if (isnan(d1.sqr())) d1 = normalize(C[i].eval(1e-6) - C[i].A);
		vec2 d2 = normalize(C[(i + 1) % l].B - C[(i + 1) % l].A);
		if (isnan(d2.sqr())) d2 = normalize(C[(i + 1) % l].eval(1e-6) - C[(i + 1) % l].A);
		double da = atan2(ndet(d1, d2), ndot(d1, d2));  // no nan please
		a += da + (da < -1e-3 ? 2.*PI : 0.);
	}
	//printf("%lf\n", a);
	return a;
}



#include "path/convex_hull_2d.h"

// assume obj is a good conditioned shape
// (connected, enclosed, counter-clockwise, no self-intersection)
object convexHull(object P) {

	//for (int i = 0; i < P.size(); i++) printBezier(P[i]);

	// check ccw-ness for every single path
	// note that a non-self-intersecting path may become self-intersecting after this
	object C;
	for (int i = 0, Pn = P.size(); i < Pn; i++) {
		bezier3 Pi = P[i];
		object app = clipSinglePath(Pi);
		C.insert(C.end(), app.begin(), app.end());
	}
	if (C.size() <= 1) return C;  // no kidding

	// make the path go counter-clockwise between endpoints
	P = C;
	C.clear();
	for (int i = 0, Pn = P.size(); i < Pn; i++) {
		C.push_back(P[i]);
		clipBack(C);
	}

	// check non-ccw at endpoints
	if (C.size() == 2) {
		bezier3 c1 = C[1], c2 = C[0];
		C.clear();
		clipPath(c1, c2, C);
		return C;
	}
	else {
		// convert array to linked list and clip
		obj_node* Pl = object2list(C);
		Pl = clipList(Pl);
		C = list2object(Pl);
	}

	if (abs(angleTurned(C) - 2.*PI) < .01)
		return C;

	// path is self-intersecting

	// first find the convex hull of a polygon
	struct vertex : vec2 {
		int cl, cr;  // index of neighborhood curves
		vertex(vec2 p, int cl, int cr) :cl(cl), cr(cr) {
			x = p.x, y = p.y;
		}
	};
	P = C;
	std::vector<vertex> Pv;
	for (int i = 0, l = P.size(); i < l; i++)
		Pv.push_back(vertex{ P[i].D, i, (i + 1) % l });
	std::vector<vertex> Cv = convexHull_2d(Pv);

	// does not always work; this test case brings it into an endless recursion:
	// M242,182 C223,124 156,112 108,151 C60,190 64,266 153,275 C235,282 234,169 155,178 C75,187 129,257 165,234 C201,211 155,189 141,203 C128,217 148,222 155,219 C162,216 152,212 151,208 C151,204 174,207 167,218 C159,230 138,232 131,214 C123,197 155,186 177,194 C199,201 196,249 158,252 C120,254 99,239 98,203 C98,167 168,137 204,169 C239,200 249,254 206,290 C163,326 76,304 55,241 C34,178 68,128 112,99 C156,70 255,89 255,89 C255,89 262,103 260,134 C258,166 242,182 242,182

	C.clear();
	if (Cv[0].cl != Cv.back().cr) {
		int i0 = Cv.back().cr, i1 = Cv[0].cl;  // between i0 and i1 are cut
		object app; clipPath(P[i0], P[i1], app);
		if (i0 < i1) i0 += P.size();
		for (int i = i1 + 1; i < i0; i++) C.push_back(P[i%P.size()]);
		C.insert(C.end(), app.begin(), app.end());
		return convexHull(C);
	}
	int breakcount = 0;  // only allow 1 to avoid certain issue
	for (int i = 0, l = Cv.size(); i < l; i++) {
		vertex v0 = Cv[i], v1 = Cv[(i + 1) % l];
		if (breakcount || v0.cr == v1.cl) C.push_back(P[v0.cr]);
		else {
			object app; clipPath(P[v0.cr], P[v1.cl], app);
			C.insert(C.end(), app.begin(), app.end());
			breakcount++;
		}
	}
	return convexHull(C);
}








// maximize the dot product of a point on the shape
double maxDot(const object &obj, vec2 n);

int loadObjs(const char* filename);  // load test objects from file
void writeObjs(const char* filename);  // write calculation results to file

int main(int argc, char* argv[]) {
	int N = loadObjs(argv[1]);
	Objs_CH = Objs;

	for (int i = 0; i < Objs.size(); i++) {
		//for (int i = 4; i < 5; i++) {
		object v = Objs[i];
		int vn = v.size();

		//for (int i = 0; i < vn; i++) printBezier(v[i]);

		printf("%d ", i);

		object u = convexHull(v);
		Objs_CH[i] = u;


		//Objs_CH[i] = clipSinglePath(Objs[i][0]);

		//object ch;
		//clipPath(v[0], v[1], ch);
		//Objs_CH[i] = ch;
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
		s = s.substr(b1 + 1, b2 - b1 - 1);
		while (1) {
			int d = min((unsigned)s.find('M', 1), (unsigned)s.find('m', 1));
			if (s[0] == 'm') s[0] = 'M';
			if (d < s.size()) {
				shape_str.push_back(s.substr(0, d) + 'Z');
				s = s.substr(d, s.length() - d);
			}
			else {
				shape_str.push_back(s + 'Z'); break;
			}
		}
	}
	//shape_str = std::vector<std::string>(&shape_str[90], &shape_str[100]);

	for (int i = 0, l = shape_str.size(); i < l; i++) {
		std::string s = shape_str[i];
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
	const int colspan = 2;  // colspan of the graph
	const int W = 600;  // with and height of sub-graphs (square)
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>\n", colspan*W, ((Objs_CH.size() - 1) / colspan + 1)*W + 80);
	fprintf(fp, "<style>rect{stroke-width:1px;stroke:black;fill:white;}path{stroke-width:1px;fill:none;}polyline{stroke:gray;fill:gray;}</style>\n");
	fprintf(fp, "<defs><marker id='EA' viewBox='0 0 10 10' refX='10' refY='5' markerUnits='strokeWidth' orient='auto' markerWidth='10' markerHeight='10'><polyline points='10,5 0,10 4,5 0,0 10,5'></polyline></marker><marker id='SA' viewBox='0 0 10 10' refX='0' refY='5' markerUnits='strokeWidth' orient='auto' markerWidth='10' markerHeight='10'><polyline points='0,5 10,0 6,5 10,10 0,5'></polyline></marker></defs>\n");
	for (int T = 0, L = Objs_CH.size(); T < L; T++) {
		fprintf(fp, "<g transform='translate(%d,%d)'>\n", W*(T%colspan), W*(T / colspan));
		//fprintf(fp, "<g transform='translate(%d,%d) translate(0,%d) scale(1,-1)'>\n", W*(T%colspan), W*(T / colspan), W);
		fprintf(fp, "<rect x='0' y='0' width='%d' height='%d'/>\n", W, W);
		fprintf(fp, "<path transform='translate(%.1lf %.1lf) scale(1,1)' d='%s' stroke='gray' vector-effect='non-scaling-stroke'/>\n", \
			.5*W - shape_translate[T].x, .5*W - shape_translate[T].y, &shape_str[T][0]);
		//for (int i = 0, PN = Objs[T].size(); i < PN; i++) \
			fprintf(fp, "<path transform='translate(%.1lf %.1lf) scale(1,1)' d='M%lf,%lf C%lf,%lf %lf,%lf %lf,%lf' stroke='gray' vector-effect='non-scaling-stroke'/>", \
				.5*W, .5*W, Objs[T][i].A.x, Objs[T][i].A.y, Objs[T][i].B.x, Objs[T][i].B.y, Objs[T][i].C.x, Objs[T][i].C.y, Objs[T][i].D.x, Objs[T][i].D.y);
		//for (int i = 0, PN = Objs_CH[T].size(); i < PN; i++) \
			fprintf(fp, "<line transform='translate(%.1lf %.1lf)' x1='%lf' y1='%lf' x2='%lf' y2='%lf' stroke='#f22' marker-end='url(#EA)'/>", \
				.5*W, .5*W, Objs_CH[T][i].A.x, Objs_CH[T][i].A.y, Objs_CH[T][i].D.x, Objs_CH[T][i].D.y);
		for (int i = 0, PN = Objs_CH[T].size(); i < PN; i++) \
			fprintf(fp, "<path transform='translate(%.1lf %.1lf) scale(1,1)' d='M%lf,%lf C%lf,%lf %lf,%lf %lf,%lf' stroke='%s' vector-effect='non-scaling-stroke'/>", \
				.5*W, .5*W, Objs_CH[T][i].A.x, Objs_CH[T][i].A.y, Objs_CH[T][i].B.x, Objs_CH[T][i].B.y, Objs_CH[T][i].C.x, Objs_CH[T][i].C.y, Objs_CH[T][i].D.x, Objs_CH[T][i].D.y, \
				i == 0 ? "green" : i & 1 ? "red" : "blue");
		//fprintf(fp, "<circle cx='%lg' cy='%lg' r='5' style='fill:red;'/>\n", Objs[T][0].A.x + .5*W, Objs[T][0].A.y + .5*W);
		//fprintf(fp, "<text x='0' y='0' transform='scale(1,-1)'>%d</text>\n", T);
		fprintf(fp, "</g>\n");
	}
	fprintf(fp, "</svg>");
	fclose(fp);
}

