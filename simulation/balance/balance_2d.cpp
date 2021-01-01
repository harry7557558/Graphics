// try to balance a 2d object placed on a plane by minimizing its gravitational potential energy
// compare the numerical solution with the exact solution

#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "numerical/geometry.h"
#include "numerical/random.h"
#include "numerical/optimization.h"


// object whose gravitational potential energy is to be minimized
// centered at the origin, represented as points that defined its convex hull
typedef std::vector<vec2> object;
std::vector<object> Objs;
std::vector<double> Objs_R;  // size of each object, not used

// rotation of the object, same index as Objs
// rotate the object counter-clockwise by the number of radians should minimize the grav potential energy
std::vector<double> rotang_exact;  // exact solution
std::vector<double> rotang_sa;  // numerical solution


// read/write data from/to file
int loadObjs(const char* filename);  // load test objects from file
void writeObjs(const char* filename);  // write calculation results to file

// optimization solvers
vec2 minimizeGravPotential_exact(const object &obj);
vec2 minimizeGravPotential_SA(const object &obj);



// calculate the gravitational potential energy of the object when laying on a plane
// with given acceleration due to gravity (usually unit vector, interestingly)
double calcGravPotential(const object &obj, vec2 g) {
	double maxE = -INFINITY;
	for (int i = 0, N = obj.size(); i < N; i++)
		maxE = max(maxE, dot(g, obj[i]));
	return maxE;
}



// the "proper" way to minimize the energy (exact solution as reference)
// this solution works in O(NlogN) even though O(N^2) brute-force solution should have no problem with the test cases

// calculate the convex hull of an object, the result is automatically counter-clockwise
object convexHull(object P) {
	std::sort(P.begin(), P.end(), [](vec2 p, vec2 q) { return p.x == q.x ? p.y < q.y : p.x < q.x; });
	object C;
	C.push_back(P[0]);
	for (int i = 1; i < (int)P.size();) {
		int Cn = C.size();
		if (Cn == 1) C.push_back(P[i]);
		else {
			if (det(C[Cn - 1] - C[Cn - 2], P[i] - C[Cn - 2]) <= 0) {
				C[Cn - 1] = P[i];
				while (Cn > 2 && det(C[Cn - 2] - C[Cn - 3], C[Cn - 1] - C[Cn - 3]) <= 0)
					C.pop_back(), Cn--, C[Cn - 1] = P[i];
			}
			else C.push_back(P[i]);
		}
		do { i++; } while (i < (int)P.size() && P[i].x == P[i - 1].x);
	}
	for (int i = P.size() - 1; i >= 0;) {
		int Cn = C.size();
		if (i == P.size() - 1) {
			if (!(C[Cn - 1] == P[i])) C.push_back(P[i]);
		}
		else {
			if (det(C[Cn - 1] - C[Cn - 2], P[i] - C[Cn - 2]) < 0) {
				C[Cn - 1] = P[i];
				while (det(C[Cn - 2] - C[Cn - 3], C[Cn - 1] - C[Cn - 3]) < 0)
					C.pop_back(), Cn--, C[Cn - 1] = P[i];
			}
			else C.push_back(P[i]);
		}
		do { i--; } while (i >= 0 && P[i].x == P[i + 1].x);
	}
	if (C.back() == C[0]) C.pop_back();
	return C;
}
// exact solution of the direction of acceleration due to gravity that minimizes the gravitational potential energy
vec2 minimizeGravPotential_exact(const object &obj) {
	double minp = INFINITY; vec2 ming;
	object C = convexHull(obj);
	for (int i = 0, l = C.size(); i < l; i++) {
		vec2 p = C[i], q = C[(i + 1) % l];
		vec2 n = normalize(q - p).rotr();
		double pt = calcGravPotential(C, n);
		if (pt < minp) minp = pt, ming = n;
	}
	return ming;
}



// when N is large: the simulated-annealing solution is expected to work in O(N)
// should work for objects with boundaries defined by splines

vec2 minimizeGravPotential_SA(const object &Obj) {
	// calculate the "radius" of the object
	double R = 0;
	for (int i = 0, n = Obj.size(); i < n; i++)
		R = max(R, length(Obj[i]));

	// function to be minimized
	int evalCount = 0;
	auto Fun = [&](double a) {
		evalCount++;
		return calcGravPotential(Obj, vec2(cos(a), sin(a)));
	};

	// simulated annealing
	uint32_t seed1 = 0, seed2 = 1, seed3 = 2;  // random number seeds
	double rand;
	double a = 0.;  // configuration
	double T = 100.0;  // temperature
	double E = Fun(a);  // energy (gravitational potential)
	double min_a = a, min_E = E, min_T = T;  // record minimum value encountered
	const int max_iter = 180;  // number of iterations
	const int max_try = 10;  // maximum number of samples per iteration
	double T_decrease = 0.9;  // multiply temperature by this each time
	for (int iter = 0; iter < max_iter; iter++) {
		for (int ty = 0; ty < max_try; ty++) {
			rand = (int32_t(seed1 = seed1 * 1664525u + 1013904223u) + .5) / 2147483648.;  // -1<rand<1
			double da = T * erfinv(rand);  // change of configuration
			double a_new = a + da;
			double E_new = Fun(a_new);
			double prob = exp(-(E_new - E) / (T*R));  // probability, note that E is approximately between -2 and 2
			rand = (seed2 = seed2 * 1664525u + 1013904223u) / 4294967296.;  // 0<=rand<1
			if (prob > rand) {  // jump
				a = a_new, E = E_new;
				if (E < min_E) {
					min_a = a, min_E = E, min_T = T;
				}
				break;
			}
		}
		// jump to the minimum point encountered
		double prob = tanh((E - min_E) / R);
		rand = (seed3 = seed3 * 1664525u + 1013904223u) / 4294967296.;
		if (prob > rand) {
			//T = min(min_T, max(T, 0.5 * (E - min_E) / R));
			a = min_a, E = min_E;
		}
		// decrease temperature
		T *= T_decrease;
		if (T < 0.0003) break;
	}
	//if (min_E < E) a = min_a;

	printf("%d; ", evalCount);

	// golden section search optimize further
	double pd = 0.02*PI;
	double a0 = a - pd, a1 = a + pd, E0 = Fun(a0), E1 = Fun(a1);
	a = GoldenSectionSearch_1d(Fun, a0, a1, E0, E1, 1e-6);

	printf("%d evals\n", evalCount);
	return vec2(cos(a), sin(a));
}





int main(int argc, char* argv[]) {
	int N = loadObjs(argv[1]);
	for (int i = 0; i < N; i++) {
		vec2 n = minimizeGravPotential_exact(Objs[i]);
		double a = atan2(n.x, -n.y);
		rotang_exact.push_back(a);
		n = minimizeGravPotential_SA(Objs[i]);
		rotang_sa.push_back(atan2(n.x, -n.y));
	}
	writeObjs(argv[2]);
	return 0;
}





// read objects from text file (well-formatted SVG)
int loadObjs(const char* filename) {
	std::ifstream ifs(filename, std::ios_base::in);
	std::string s; getline(ifs, s);
	while (true) {
		// read line
		getline(ifs, s);
		int b1 = s.find('"', 0), b2 = s.find('"', b1 + 1);
		if (b1 == -1) break;
		s = s.substr(b1 + 1, b2 - b1 - 1);

		// split string
		std::stringstream ss(s);
		object poly;
		while (ss.good()) {
			vec2 p; ss >> p.x >> p.y;
			poly.push_back(.6 * vec2(p.x, -p.y));
		}
		if (poly.back() == poly[0])
			poly.pop_back();

		// calculate the center of mass using the divergence theorem
		vec2 C(0); double A(0);
		for (int i = 0, n = poly.size(); i < n; i++) {
			vec2 p = poly[i], q = poly[(i + 1) % n];
			double dA = .5*det(p, q);
			vec2 dC = (dA / 3.)*(p + q);
			A += dA, C += dC;
		}
		Objs_R.push_back(sqrt(abs(A) / PI));
		C *= 1. / A;

		// translate the shape so its center of mass is the origin
		for (int i = 0, n = poly.size(); i < n; i++)
			poly[i] -= C;
		Objs.push_back(poly);

	}
	ifs.close();
	return Objs.size();
}

void writeObjs(const char* filename) {
	FILE* fp = fopen(filename, "wb");
	const int colspan = 4;  // colspan of the graph
	const int W = 300;  // with and height of sub-graphs (square)
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>\n", colspan*W, ((Objs.size() - 1) / colspan + 1)*W + 80);
	fprintf(fp, "<style>rect{stroke-width:1px;stroke:black;fill:white;}polygon{stroke:black;stroke-width:1px;fill:none;}line{stroke:black;stroke-width:1px;}path{stroke:gray;stroke-width:1px;fill:none;}</style>\n");
	for (int T = 0, L = Objs.size(); T < L; T++) {
		object P = Objs[T]; int PN = P.size();
		double a_ex = rotang_exact[T], a_sa = rotang_sa[T];
		fprintf(fp, "<g transform='translate(%d,%d)'>\n", W*(T%colspan), W*(T / colspan));
		fprintf(fp, "<rect x='0' y='0' width='%d' height='%d'/>\n", W, W);

		// graph of potential function
		fprintf(fp, "<path d='");
		for (double a = 0.; a < 2.*PI; a += .01*PI)
			fprintf(fp, "%c%.1lf,%.1lf", a == 0. ? 'M' : 'L', (.5*W / PI)*a, W - calcGravPotential(P, vec2(sin(a), cos(a))));
		fprintf(fp, "'/>\n");
		// values
		fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='3' style='fill:#66f;'/>\n", (.5*W / PI)*(PI - a_ex + 2.*PI*(a_ex > PI)), W - calcGravPotential(P, vec2(sin(a_ex), -cos(a_ex))));
		fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='3' style='fill:#f66;'/>\n", (.5*W / PI)*(PI - a_sa + 2.*PI*(a_sa > PI)), W - calcGravPotential(P, vec2(sin(a_sa), -cos(a_sa))));

		// reference shape (blue)
		double miny = calcGravPotential(P, vec2(sin(a_ex), -cos(a_ex)));
		fprintf(fp, "<polygon transform='translate(0 %d) scale(1 -1) translate(%.1lf %.1lf) rotate(%.3lf)' points=\"", W, .5*W, miny, -180. / PI * a_ex);
		for (int i = 0; i < PN; i++) fprintf(fp, "%.1lf %.1lf ", P[i].x, P[i].y);
		fprintf(fp, "%.1lf %.1lf\" style='stroke-dasharray:5;stroke:blue;'/>\n", P[0].x, P[0].y);
		// center of mass
		fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='2' style='fill:blue;'/>\n", .5*W, W - miny);

		// numerically optimized shape (red)
		miny = calcGravPotential(P, vec2(sin(a_sa), -cos(a_sa)));
		fprintf(fp, "<polygon transform='translate(0 %d) scale(1 -1) translate(%.1lf %.1lf) rotate(%.3lf)' points=\"", W, .5*W, miny, -180. / PI * a_sa);
		for (int i = 0; i < PN; i++) fprintf(fp, "%.1lf %.1lf ", P[i].x, P[i].y);
		fprintf(fp, "%.1lf %.1lf\" style='stroke:red;'/>\n", P[0].x, P[0].y);
		// center of mass
		fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='2' style='fill:red;'/>\n", .5*W, W - miny);

		fprintf(fp, "</g>\n");
	}

	fprintf(fp, "</svg>");
	fclose(fp);
}

