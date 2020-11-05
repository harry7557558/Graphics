// try to balance 2d objects placed on a plane by minimizing the gravitational potential energy
// objects are defined by cubic Bezier spline boundaries

#include <stdio.h>
#include <fstream>
#include <algorithm>

#include "numerical/geometry.h"
#include "numerical/random.h"
#include "numerical/optimization.h"
#include "numerical/integration.h"
#include "numerical/rootfinding.h"

#include "path/svg_path_read.h"
typedef svg_path_read::bezier3 bezier3;

namespace exact_solution {
#include "bezier3_convex_hull.cpp"
}



// object whose gravitational potential energy is to be minimized
// centered at the origin, represented as points that defined its convex hull
typedef std::vector<bezier3> object;
std::vector<object> Objs;

// calculated rotation of the object, same index as Objs
// rotate the object counter-clockwise by the number of radians should minimize its grav potential energy
std::vector<double> rotangs;
std::vector<double> rotangs_exact;  // exact reference solution


// read/write data from/to file
int loadObjs(const char* filename);  // load test objects from file
void writeObjs(const char* filename);  // write calculation results to file




// calculate the gravitational potential energy of the object when laying on a plane
// with given acceleration due to gravity (usually unit vector, interestingly)
double calcGravPotential(const object &obj, vec2 g) {
	double maxE = -INFINITY;
	for (int i = 0, N = obj.size(); i < N; i++) {
		double a = dot(obj[i].A, g), b = dot(obj[i].B, g), c = dot(obj[i].C, g), d = dot(obj[i].D, g);
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




// simulated annealing, same as the one in balance_2d.cpp
// (ironically, a brute-force search takes less samples to get more accurate estimates)

vec2 minimizeGravPotential_SA(const object &Obj) {
	// estimate the "radius" of the object
	double R = 0;
	for (int i = 0, n = Obj.size(); i < n; i++)
		R = max(R, length(Obj[i].A));

	// function to be minimized
	int evalCount = 0;
	auto Fun = [&](double a) {
		evalCount++;
		return calcGravPotential(Obj, vec2(cos(a), sin(a)));
	};

	// simulated annealing
	uint32_t seed1 = 0, seed2 = 1, seed3 = 2;  // random number seeds
	double rand;
	double a = 0.;  // configulation
	double T = 100.0;  // temperature
	double E = Fun(a);  // energy (gravitational potential)
	double min_a = a, min_E = E, min_T = T;  // record minimum value encountered
	const int max_iter = 180;  // number of iterations
	const int max_try = 10;  // maximum number of samples per iteration
	double T_decrease = 0.9;  // multiply temperature by this each time
	for (int iter = 0; iter < max_iter; iter++) {
		for (int ty = 0; ty < max_try; ty++) {
			rand = (int32_t(seed1 = seed1 * 1664525u + 1013904223u) + .5) / 2147483648.;  // -1<rand<1
			double da = T * erfinv(rand);  // change of configulation
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
	double pd = 0.05*PI;
	double a0 = a - pd, a1 = a + pd, E0 = Fun(a0), E1 = Fun(a1);
	a = GoldenSectionSearch_1d(Fun, a0, a1, E0, E1, 1e-6);

	printf("%d evals\n", evalCount);
	return vec2(cos(a), sin(a));
}



// reference solution
vec2 minimizeGravPotential_exact(const object &Obj) {
	object C = *(object*)&exact_solution::convexHull(*(exact_solution::object*)&Obj);
	int Cn = C.size();
	double miny = INFINITY; vec2 maxn(NAN);
	for (int i = 0; i < Cn; i++) {
		bezier3 b = C[i];
		auto Fun = [&](double t) {
			return calcGravPotential(object({ b }), normalize(b.eval(t)));
		};
		double t0 = 0., t1 = 1., y0 = Fun(t0), y1 = Fun(t1);
		double t = GoldenSectionSearch_1d(Fun, t0, t1, y0, y1);
		double y = Fun(t);
		if (y < miny) {
			miny = y;
			maxn = normalize(b.eval(t));
		}
	}
	return maxn;
}



int main(int argc, char* argv[]) {
	int N = loadObjs(argv[1]);
	for (int i = 0; i < N; i++) {
		vec2 n = minimizeGravPotential_SA(Objs[i]);
		rotangs.push_back(atan2(n.x, -n.y));
		n = i == 12 || i == 17 || i == 18 ? n :  // failed test cases
			minimizeGravPotential_exact(Objs[i]);
		rotangs_exact.push_back(atan2(n.x, -n.y));
	}
	writeObjs(argv[2]);
	return 0;
}





// intended for writing but not actually used
std::vector<std::string> shape_str, shape_attr;


// read objects from text file (well-formatted SVG)
int loadObjs(const char* filename) {
	std::ifstream ifs(filename, std::ios_base::in);
	std::string s; getline(ifs, s);
	while (true) {
		// read line
		getline(ifs, s);
		int b1 = s.find('"', 0), b2 = s.find('"', b1 + 1);
		if (b1 == -1) break;
		shape_attr.push_back(s.substr(b2 + 1, s.find('/', b2 + 1) - b2 - 1));
		shape_str.push_back(s = s.substr(b1 + 1, b2 - b1 - 1));

		// parse string
		object obj;
		if (!svg_path_read::parse_path(s, obj)) {
			shape_attr.pop_back(), shape_str.pop_back();
			fprintf(stderr, "Error parsing string: %s\n", &s[0]);
			continue;
		}

		// calculate the center of mass using divergence theorem
		// convert to line integral
		vec2 Ctr(0); double S(0);
		for (int i = 0, n = obj.size(); i < n; i++) {
			vec2 A = obj[i].A, B = obj[i].B, C = obj[i].C, D = obj[i].D;
			double dS = 0.5 * (.6*det(A, B) + .3*det(A, C) + .1*det(A, D) + .3*det(B, C) + .3*det(B, D) + .6*det(C, D));
			// doable analytically but definitely not a fun work
			vec2 dC = NIntegrate_GL4<vec2>([&](double t) {
				vec2 p = A + t * (-3.*A + 3.*B + t * (3.*A - 6.*B + 3.*C + t * (-A + 3.*B - 3.*C + D)));
				vec2 dp = -3.*A + 3.*B + t * (6.*A - 12.*B + 6.*C + t * (-3.*A + 9.*B - 9.*C + 3.*D));
				return .5 * det(p, dp) * (2. / 3.*p);
			}, 0, 1);
			// CAS solution
			// vec2 dC = (1. / 840.)*vec2(A.x*A.x*(105.*B.y + 30.*C.y + 5.*D.y) + A.x*(-105.*A.y*B.x - 30.*A.y*C.x - 5.*A.y*D.x + 45.*B.x*B.y + 45.*B.x*C.y + 15.*B.x*D.y - 3.*B.y*D.x + 18.*C.x*C.y + 12.*C.x*D.y + 3.*C.y*D.x + 5.*D.x*D.y) + B.x*B.x*(-45.*A.y + 27.*C.y + 18.*D.y) + B.x*(-45.*A.y*C.x - 12.*A.y*D.x - 27.*B.y*C.x - 18.*B.y*D.x + 27.*C.x*C.y + 45.*C.x*D.y + 30.*D.x*D.y) + C.x*C.x*(-18.*A.y - 27.*B.y + 45.*D.y) + C.x*(-15.*A.y*D.x - 45.*B.y*D.x - 45.*C.y*D.x + 105.*D.x*D.y) + D.x*D.x*(-5.*A.y - 30.*B.y - 105.*C.y), A.y*A.y*(-105.*B.x - 30.*C.x - 5.*D.x) + A.y*(105.*A.x*B.y + 30.*A.x*C.y + 5.*A.x*D.y - 45.*B.x*B.y + 3.*B.x*D.y - 45.*B.y*C.x - 15.*B.y*D.x - 18.*C.x*C.y - 3.*C.x*D.y - 12.*C.y*D.x - 5.*D.x*D.y) + B.y*B.y*(45.*A.x - 27.*C.x - 18.*D.x) + B.y*(45.*A.x*C.y + 12.*A.x*D.y + 27.*B.x*C.y + 18.*B.x*D.y - 27.*C.x*C.y - 45.*C.y*D.x - 30.*D.x*D.y) + C.y*C.y*(18.*A.x + 27.*B.x - 45.*D.x) + C.y*(15.*A.x*D.y + 45.*B.x*D.y + 45.*C.x*D.y - 105.*D.x*D.y) + D.y*D.y*(5.*A.x + 30.*B.x + 105.*C.x));
			S += dS, Ctr += dC;
		}
		vec2 C = Ctr / S;

		// translate the shape so its center of mass is the origin
		for (int i = 0, n = obj.size(); i < n; i++)
			obj[i].translate(-C), obj[i].scale(0.6*vec2(1, -1));
		if (svg_path_read::calcArea(obj) < 0.)
			svg_path_read::reversePath(obj);
		Objs.push_back(obj);

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
	int T_max = std::min({ Objs.size(), rotangs.size(), rotangs_exact.size() });
	for (int T = 0; T < T_max; T++) {
		object P = Objs[T]; int PN = P.size();
		double a = rotangs[T];
		fprintf(fp, "<g transform='translate(%d,%d)'>\n", W*(T%colspan), W*(T / colspan));
		fprintf(fp, "<rect x='0' y='0' width='%d' height='%d'/>\n", W, W);

		// graph of potential function
		fprintf(fp, "<path d='");
		double minE = INFINITY, mina = NAN;  // record minimum value
		for (double a = 0.; a < 2.*PI; a += .01*PI) {
			double E = calcGravPotential(P, vec2(sin(a), cos(a)));
			fprintf(fp, "%c%.1lf,%.1lf", a == 0. ? 'M' : 'L', (.5*W / PI)*a, W - E);
			if (E < minE) minE = E, mina = atan2(sin(a), -cos(a));
		}
		if (!(T == 12 || T == 17 || T == 18)) {
			mina = rotangs_exact[T];
			minE = calcGravPotential(P, vec2(sin(mina), -cos(mina)));
		}
		fprintf(fp, "'/>\n");
		// value
		fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='3' style='fill:#888;'/>\n", (.5*W / PI)*(PI - a + 2.*PI*(a > PI)), W - calcGravPotential(P, vec2(sin(a), -cos(a))));

		// untransformed shape
		std::string pathd;
		vec2 P0(NAN);
		for (int i = 0; i < PN; i++) {
			bezier3 Vi = P[i];
			char buf[64];
			if (!((Vi.A - P0).sqr() < 1e-12))
				sprintf(buf, "M%lg,%lg", Vi.A.x, Vi.A.y),
				pathd += buf;
			sprintf(buf, "C%lg,%lg %lg,%lg %lg,%lg", Vi.B.x, Vi.B.y, Vi.C.x, Vi.C.y, Vi.D.x, Vi.D.y);
			pathd += buf;
			P0 = Vi.D;
		}
		if (length(P0 - P[0].A) < 1e-6)  // should be true
			pathd.push_back('Z');
		fprintf(fp, "<path transform='translate(0 %d) scale(1 -1) translate(%.1lf %.1lf)' d=\"", W, .5*W, .7*W);
		fprintf(fp, &pathd[0]);
		fprintf(fp, "\" style='stroke:#bbb;stroke-dasharray:5;fill:none;'/>\n");

		// optimal value
		double miny = calcGravPotential(P, vec2(sin(a), -cos(a)));
		if (minE < miny) {
			fprintf(fp, "<path transform='translate(0 %d) scale(1 -1) translate(%.1lf %.1lf) rotate(%.3lf)' d='%s' style='stroke:#88f;stroke-dasharray:8;fill:none;'/>\n",
				W, .5*W, minE, -180. / PI * mina, &pathd[0]);
			fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='3' style='fill:#88f;'/>\n", (.5*W / PI)*(PI - mina + 2.*PI*(mina > PI)), W - minE);
		}

		// transformed shape
		fprintf(fp, "<path transform='translate(0 %d) scale(1 -1) translate(%.1lf %.1lf) rotate(%.3lf)' d=\"", W, .5*W, miny, -180. / PI * a);
		fprintf(fp, &pathd[0]);
		fprintf(fp, "\" style='stroke:black;fill:#00000020;'/>\n");

		// center of mass
		fprintf(fp, "<circle cx='%.1lf' cy='%.1lf' r='2' style='fill:black;'/>\n", .5*W, W - miny);

		fprintf(fp, "</g>\n");
	}

	fprintf(fp, "</svg>");
	fclose(fp);
}

