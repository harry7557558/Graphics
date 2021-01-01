// simulating the elastic collision of 16 polygons in 2d
// 16 = 1<<4 is a good number ;)

#define OBJECT_N 16

/* INCOMPLETE */
// currently only implemented numerical path


#include <stdio.h>
#include <vector>
#include <string>
#include <chrono>

#include "numerical/geometry.h"  // vec2
#include "numerical/ode.h"  // RungeKuttaMethod
#include "numerical/rootfinding.h"  // solveTrigQuadratic_refine
#include "numerical/random.h"

// debug output
#define printerr(s, ...) fprintf(stderr, s, __VA_ARGS__)


struct state {
	double t = 0;  // time
	vec2 x = vec2(0), v = vec2(0);  // position and linear velocity
	double r = 0, w = 0;  // rotation and angular velocity
};
typedef state keyFrame;
typedef struct { vec2 c, d; } splineTo;  // path as quadratic bezier curve
typedef std::vector<keyFrame> sequence;
typedef std::vector<vec2> polygon;
struct object {
	polygon P;  // with center of mass at the origin
	double inv_m;  // reciprocal of mass
	double inv_I;  // reciprocal of the moment of inertia
	double r, r2;  // radius of bounding sphere and its square
};


// constants/parameters/conditions
const double W = 6.0, H = 4.0;  // width and height of simulation box;  unit: m
const vec2 BMax = vec2(0.5*W, 0.5*H), BMin = vec2(-0.5*W, -0.5*H);  // bounding box
const double SC = 100.0;  // output scaling
const vec2 g = vec2(0, -9.8);  // gravitational acceleration, zero horizontal component
const double t_max = 10.0;  // end time
const double SM = 2.0;  // slow motion

object Body[OBJECT_N];  // main objects
state State0[OBJECT_N];  // initial states

#define MAX_VTX 20  // maximum number of vertices in a polygon

void init() {
	// content: words "Hello, World!"
	// there is a highly elongated shape that foils the bounding sphere trick
	polygon Test_Scene[OBJECT_N] = {
		{ vec2(9.5,105.5), vec2(0.5,302.5), vec2(34.5,301.5), vec2(46.5,228.5), vec2(105.5,228.5), vec2(96.5,294.5), vec2(148.5,290.5), vec2(135.5,91.5), vec2(101.5,90.5), vec2(98.5,180.5), vec2(48.5,190.5), vec2(44.5,91.5) },
		{ vec2(246.2,213), vec2(205.1,218.3), vec2(207.3,235.7), vec2(264.2,231.3), vec2(261.6,176.8), vec2(189.4,165), vec2(168.6,232.8), vec2(200.2,288.2), vec2(257,274.2), vec2(259.8,250.1), vec2(206.9,263.3), vec2(190.1,224), vec2(204.2,184.5), vec2(249.2,192) },
		{ vec2(294.7,103.7), vec2(292.7,288.7), vec2(330.7,276.7), vec2(312.7,247.7), vec2(312.7,99.7) },
		{ vec2(339.5,96.4), vec2(344.5,277.4), vec2(374.5,280.4), vec2(381.5,233.4), vec2(358.5,233.4), vec2(355.5,97.4) },
		{ vec2(440.4,276.7), vec2(477.6,275), vec2(499.8,226.4), vec2(467.7,175.2), vec2(416.5,181), vec2(396.7,229.7), vec2(419.8,274.2), vec2(435.5,258.5), vec2(414,227.2), vec2(426.4,199.1), vec2(458.6,197.5), vec2(480.9,232.1), vec2(464.4,258.5), vec2(440.4,257.7) },
		{ vec2(522.2,263.2), vec2(513.2,248.2), vec2(537.2,241.2), vec2(555.2,259.2), vec2(514.2,306.2), vec2(507.2,292.2), vec2(534.2,265.2) },
		{ vec2(588.5,124.5), vec2(619.5,123.5), vec2(645.5,240.5), vec2(669.5,123.5), vec2(685.5,119.5), vec2(706.5,230.5), vec2(723.5,119.5), vec2(756.5,115.5), vec2(728.5,276.5), vec2(695.5,277.5), vec2(676.5,185.5), vec2(654.5,282.5), vec2(626.5,279.5) },
		{ vec2(822.5,200.5), vec2(767.5,204.5), vec2(769.5,266.5), vec2(835.5,267.5), vec2(834.5,200.5), vec2(823.5,217.5), vec2(820.5,251.2), vec2(783.5,253.5), vec2(784.2,221.2), vec2(817.9,217.9) },
		{ vec2(862.5,191.5), vec2(865.5,264.5), vec2(891.5,265.5), vec2(886.5,218.5), vec2(920.5,194.5), vec2(904.5,184.5), vec2(881.5,206.5), vec2(880.5,183.5) },
		{ vec2(944.5,114.5), vec2(942.5,265.5), vec2(975.5,265.5), vec2(976.5,240.5), vec2(951.5,245.5), vec2(964.5,107.5) },
		{ vec2(1040.5,264.5), vec2(1001.5,266.5), vec2(992.5,205.5), vec2(1056.9,201.2), vec2(1049.5,107.5), vec2(1072.5,99.5), vec2(1069.5,245.5), vec2(1078.5,244.5), vec2(1069.5,269.5), vec2(1053.5,265.5), vec2(1057.5,211.2), vec2(1008.5,219.5), vec2(1012.5,253.9), vec2(1039.5,251.5) },
		{ vec2(1126.5,105.5), vec2(1153.5,102.5), vec2(1149.5,237.5), vec2(1134.5,239.5) },
		{ vec2(1132.5,256.5), vec2(1135.5,275.5), vec2(1149.5,276.5), vec2(1150.5,252.5) },
		{ vec2(415.8,310.5), vec2(497,321.2), vec2(620.4,303.2), vec2(732.8,323.2), vec2(899.3,290.5), vec2(1076.9,303.2), vec2(1075.2,314.5), vec2(898.5,303.2), vec2(732,337.9), vec2(615.3,314.5), vec2(497,336.5), vec2(395.5,313.2) },
		{ vec2(175.5,75.5), vec2(383.5,76.5), vec2(386.5,34.5), vec2(344.5,20.5), vec2(321.5,45.5), vec2(307.5,0.5), vec2(249.5,4.5), vec2(230.5,44.5), vec2(203.5,34.5) },
		{ vec2(744.3,47.3), vec2(730.8,48.3), vec2(727.8,63.5), vec2(735.9,64.5), vec2(736,71.2), vec2(717.5,67.2), vec2(717.2,70), vec2(735.2,76.2), vec2(732.2,113.5), vec2(736.9,113.5), vec2(739.5,94.5), vec2(744.5,112.5), vec2(749.9,110.9), vec2(741.9,75.7), vec2(759.7,70.2), vec2(758.4,67.4), vec2(740.5,71.2), vec2(740.5,65.2), vec2(747.5,62.3) },
	};
	//State0[0].v = vec2(5, 2.5), State0[1].v = vec2(-5, 1);
	vec2 Min = vec2(INFINITY), Max = -Min;  // bounding box of the overall shape
	const double sc = 0.005;
	for (int _ = 0; _ < OBJECT_N; _++) {
		polygon S = Test_Scene[_]; int N = S.size();
		for (int i = 0; i < N; i++) S[i] = S[i] * vec2(1, -1) + rand2(0.1);  // avoid degenerated cases
		// calculate mass, center of mass, and moment of inertia
		// treat the shape as a solid (divergence theorem)
		const double density = 1.0;
		double M(0.0); vec2 C(0.0); double I(0.0);
		for (int i = 0; i < N; i++) {
			vec2 p = S[i], q = S[(i + 1) % N];
			double dM = .5*density*det(p, q);
			vec2 dC = (dM / 3.)*(p + q);
			double dI = (dM / 6.)*(dot(p, p) + dot(p, q) + dot(q, q));
			M += dM, C += dC, I += dI;
			Min = pMin(Min, p), Max = pMax(Max, p);
		}
		C /= M;
		// make sure the shape is counter-clockwise
		if (M < 0.) {
			M = -M, I = -I;
			for (int i = 0, l = S.size(); i < l / 2; i++) {
				std::swap(S[i], S[l - i - 1]);
			}
		}
		// translate the shape to the origin
		C *= sc, M *= sc * sc, I *= sc * sc * sc * sc;
		for (int i = 0; i < N; i++) S[i] = S[i] * sc - C;
		I -= M * C.sqr();
		// calculate bounding sphere
		double r2 = 0;
		for (int i = 0; i < N; i++) {
			double r2t = S[i].sqr();
			r2 = max(r2, r2t);
		}
		// update calculation
		Body[_].P = S;
		Body[_].inv_m = 1. / M;
		Body[_].inv_I = 1. / I;
		Body[_].r2 = r2, Body[_].r = sqrt(r2);
		State0[_].x = C;
	}
	// set the initial state
	vec2 C = 0.5*sc*(Min + Max);
	for (int _ = 0; _ < OBJECT_N; _++) State0[_].x -= C;
}



// simulating by numerically integrating collision force
sequence IntPath[OBJECT_N];
void calcIntPath() {
	// debug code for tracking energy
	// directly write to SVG, should not be visible when placed in HTML <image>
	printf("<text id='ei' x='400' y='300'></text><script>({i:0,u:function(){var that=this;document.getElementById('ei').textContent=[");
	int EN = 0;

	state st[OBJECT_N];
	for (int i = 0; i < OBJECT_N; i++) st[i] = State0[i];
	auto pushState = [&]() {
		for (int i = 0; i < OBJECT_N; i++) {
			IntPath[i].push_back(st[i]);
		}
	};
	const unsigned SL = sizeof(state) / sizeof(double);  // 7
	pushState();

	double temp0[SL << 4], temp1[SL << 4], temp2[SL << 4];
	double dt = 0.00005, t = 0;  // not too large
	for (int i = 0, tN = int(t_max / dt); i < tN; i++) {
		// with some constant optimization
		RungeKuttaMethod([](const double *x, double t, double* dxdt) {
			const double k = 1e6;  // not too small
			const state* st = (state*)x;
			state* ds = (state*)dxdt;
			vec2 F[OBJECT_N]; double T[OBJECT_N];  // force and torque
			for (int i = 0; i < OBJECT_N; i++) F[i] = vec2(0), T[i] = 0;
			// calculate collision force between objects and boundary
			for (int _ = 0; _ < OBJECT_N; _++) {
				vec2 cx = st[_].x; double cr = Body[_].r;
				if (cx.x - cr < BMin.x || cx.x + cr > BMax.x || cx.y - cr < BMin.y || cx.y + cr > BMax.y) {  // bounding sphere check
					vec2 dF(0.0); double dT(0.0);  // force and torque change
					double a = st[_].r, sa = sin(a), ca = cos(a);
					for (int i = 0, l = Body[_].P.size(); i < l; i++) {  // go through all vertices
						vec2 pr = Body[_].P[i];
						pr = vec2(ca * pr.x - sa * pr.y, sa * pr.x + ca * pr.y);
						vec2 p = cx + pr;
						// calculate collision force
						vec2 ddF(0.);
						if (p.x > BMax.x) ddF += vec2(k*(BMax.x - p.x), 0);
						if (p.x < BMin.x) ddF += vec2(k*(BMin.x - p.x), 0);
						if (p.y > BMax.y) ddF += vec2(0, k*(BMax.y - p.y));
						if (p.y < BMin.y) ddF += vec2(0, k*(BMin.y - p.y));
						dF += ddF, dT += det(p - cx, ddF);
					}
					F[_] += dF, T[_] += dT;
				}
			}
			// calculate collision force between objects
			for (int u = 0; u < OBJECT_N; u++) {
				int un = Body[u].P.size();
				// calculate the position of each vertex
				vec2 V[MAX_VTX], Vd[MAX_VTX]; double Vd2[MAX_VTX];
				double ua = st[u].r, ca = cos(ua), sa = sin(ua);
				for (int ui = 0; ui < un; ui++) {
					vec2 p = Body[u].P[ui];
					V[ui] = st[u].x + vec2(ca*p.x - sa * p.y, sa*p.x + ca * p.y);
				}
				for (int ui = 0; ui < un; ui++) {
					Vd[ui] = V[(ui + 1) % un] - V[ui];
					Vd2[ui] = 1.0 / Vd[ui].sqr();
				}
				// check the vertices of all other objects
				for (int v = 0; v < OBJECT_N; v++) if (v != u) {
					double rs = Body[u].r + Body[v].r;
					if ((st[v].x - st[u].x).sqr() < rs * rs) {  // bounding sphere check
						int vn = Body[v].P.size();
						for (int vi = 0; vi < vn; vi++) {  // go through vertices
							vec2 vp = Body[v].P[vi];
							double va = st[v].r, ca = cos(va), sa = sin(va);
							vp = st[v].x + vec2(ca*vp.x - sa * vp.y, sa*vp.x + ca * vp.y);
							if ((vp - st[u].x).sqr() < Body[u].r2) {  // bounding sphere check
								// calculate the signed distance to a polygon
								// also records the edge that produces the minimum absolute distance
								// may be optimized as the exact distance is not required
								int id = -1;
								double sd = INFINITY;
								bool sgn = false;
								for (int i = 0; i < un; i++) {
									vec2 e = Vd[i];
									vec2 w = vp - V[i];
									double h = dot(w, e) * Vd2[i];
									h = h<0. ? 0. : h>1. ? 1. : h;
									vec2 b = w - e * h;
									h = b.sqr();
									if (h < sd) sd = h, id = i;
									if (e.y < 0.0) e.y = -e.y, w.y = -w.y;
									if (w.y > 0.0 && w.y < e.y && (w.y*e.x > w.x*e.y)) sgn ^= 1;
								}
								// penetration: calculate force and torque
								if (sgn) {
									vec2 n = normalize(Vd[id]).rotr();
									vec2 dF = k * sqrt(sd) * n;
									F[u] -= dF;
									F[v] += dF;
									T[u] -= det(vp - st[u].x, dF);
									T[v] += det(vp - st[v].x, dF);
								}
							}
						}
					}
				}
			}
			// update
			for (int i = 0; i < OBJECT_N; i++) {
				ds[i].t = 1;
				ds[i].x = st[i].v;  // linear velocity
				ds[i].v = F[i] * Body[i].inv_m + g;  // linear acceleration
				ds[i].r = st[i].w;  // angular velocity
				ds[i].w = T[i] * Body[i].inv_I;  // angular acceleration
			}
		}, (double*)&st, SL * OBJECT_N, t, dt, temp0, temp1, temp2);

		// check energy conservation
		if (i % (int(0.01 / dt)) == 0) {
			double E = 0;
			for (int _ = 0; _ < OBJECT_N; _++) {
				double El = (.5*st[_].v.sqr() - dot(g, st[_].x)) / Body[_].inv_m;
				double Ea = (.5*st[_].w*st[_].w) / Body[_].inv_I;
				E += El + Ea;
			}
			//printerr("%lf %lf\n", t, E);
			printf("%lf,", E); EN++;
		}

		// time update and output
		if ((t += dt) + 1e-6 >= t_max) break;
		if (i % (int(0.02 / dt)) == 0) {
			pushState();
		}
	}
	for (int i = 0; i < OBJECT_N; i++) st[i].t = t_max;
	pushState();

	// sometimes doesn't sync quite well
	printf("][(this.i++)%%%d];setTimeout(function(){that.u()},%d);}}).u();</script>\n", EN, int(SM * 10));
}


// simulating using collision law
sequence AnalyticPath[OBJECT_N];
void calcAnalyticPath() {
	state st[OBJECT_N];
	for (int i = 0; i < OBJECT_N; i++) st[i] = State0[i];
	auto pushState = [&]() {
		for (int i = 0; i < OBJECT_N; i++) {
			AnalyticPath[i].push_back(st[i]);
		}
	};
	pushState();

	double t = 0;
	while (t < t_max) {
		// calculate the next collision
		double dt = t_max - t;
		int id[2] = { -1, -1 };  // index of colliding objects
		vec2 dV[2];  // changes of linear velocity
		double dW[2];  // changes of angular velocity

		// calculate collisions between objects and boundary
		for (int _ = 0; _ < OBJECT_N; _++) {
			double a = st[_].r, sa = sin(a), ca = cos(a);

			for (int i = 0, l = Body[_].P.size(); i < l; i++) {
				vec2 r = Body[_].P[i]; r = vec2(ca*r.x - sa * r.y, sa*r.x + ca * r.y);
				vec2 x = st[_].x, v = st[_].v;
				double w = st[_].w;
				vec2 n = vec2(0.0);

				// find the intersection
				const double eps = 1e-6;
				double tt = solveTrigQuadratic_refine(0, v.x, x.x - BMax.x, r.x, -r.y, w, eps);
				if (tt < dt) dt = tt, n = vec2(-1, 0);
				tt = solveTrigQuadratic_refine(.5*g.y, v.y, x.y - BMin.y, r.y, r.x, w, eps);
				if (tt < dt) dt = tt, n = vec2(0, 1);
				tt = solveTrigQuadratic_refine(0, v.x, x.x - BMin.x, r.x, -r.y, w, eps);
				if (tt < dt) dt = tt, n = vec2(1, 0);
				tt = solveTrigQuadratic_refine(.5*g.y, v.y, x.y - BMax.y, r.y, r.x, w, eps);
				if (tt < dt) dt = tt, n = vec2(0, -1);

				// calculate collision reaction
				if (n != vec2(0.0)) {
					double s = det(r.rotate(w*dt), n);  // unit torque direction maybe
					double dp = -2 * (dot(v + g * dt, n) + w * s) / (Body[_].inv_m + Body[_].inv_I*s*s);  // change of linear momentum (scalar)
					dV[0] = (Body[_].inv_m*dp)*n;  // change of linear velocity
					dW[0] = (Body[_].inv_I*dp*s);  // change of angular velocity
					id[0] = _;
				}
			}
		}

		// the collision between objects doesn't seem can be solved analytically
		/* INCOMPLETE */

		//printerr("%d %lf %lf %lf %lf\n", id[0], t + dt, dV[0].x, dV[0].y, dW[0]);

		// output
		int tN = int(ceil(dt / 0.04)); double ddt = dt / tN;
		for (int _t = 1; _t <= tN; _t++) {
			for (int _ = 0; _ < OBJECT_N; _++) {
				st[_].t += ddt;
				st[_].x += st[_].v*ddt + g * (.5*ddt*ddt);
				st[_].v += g * ddt;
				st[_].r += st[_].w*ddt;
			}
			pushState();
		}

		// collision respond
		if (id[0] != -1) {
			st[id[0]].v += dV[0], st[id[0]].w += dW[0];
		}
		if (id[1] != -1) {
			st[id[1]].v += dV[1], st[id[0]].w += dW[1];
		}
		t += dt;

		// check energy conservation
		double E = 0;
		for (int _ = 0; _ < OBJECT_N; _++)
			E += (.5*st[_].v.sqr() - dot(st[_].x, g)) / Body[_].inv_m + (.5*st[_].w*st[_].w) / Body[_].inv_I;
		printerr("A %lf %lf\n", t, E);
	}
}



// visualization
int main(int argc, char** argv) {
	// output format: animated svg
	freopen(argv[1], "w", stdout);

	// decimal to string optimized for size
	auto str = [](double x, int d = 3)->std::string {
		char format[16]; sprintf(format, "%%.%dlf", d);
		char st[256]; sprintf(st, format, x);
		std::string s = st;
		while (s.back() == '0') s.pop_back();
		if (s.back() == '.') s.pop_back();
		if (s[0] == '0') s = s.erase(0, 1);
		return s == "" ? "0" : s;
	};
	// world coordinate to SVG coordinate
	auto transform = [](vec2 p)->vec2 {
		return SC * vec2(p.x - BMin.x, BMax.y - p.y);
	};

	// svg header
	printf("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='%d' height='%d'>\n",
		int((BMax.x - BMin.x)*SC + 1), int((BMax.y - BMin.y)*SC + 1));

	// boundary
	int IMG_W = int(W * SC + .5), IMG_H = int(H * SC + .5);
	printf("<rect width='%d' height='%d' style='fill:white;stroke-width:1;stroke:black'/>\n", IMG_W, IMG_H);

	// simulation
	init();
	auto t0 = std::chrono::high_resolution_clock::now();
	//calcIntPath();
	calcAnalyticPath();
	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
	printerr("%lfsecs\n", time_elapsed);

	// objects
	printf("<defs>\n");
	for (int _ = 0; _ < OBJECT_N; _++) {
		printf(" <polygon id='body%d' points='", _);
		for (int i = 0, l = Body[_].P.size(); i <= l; i++) {
			vec2 p = SC * vec2(1, -1)*Body[_].P[i%l];
			printf("%s %s%c", &str(p.x, 1)[0], &str(p.y, 1)[0], i == l ? '\'' : ' ');
		}
		printf("/>\n");
	}
	printf("</defs>\n");

	// animation
	for (int _ = 0; _ < OBJECT_N; _++) {
		auto printPath = [&](const sequence &p, const char* style) {
			if (p.empty()) return;
			static int id = 0; id++;
			printf("<use id='obj%d' xlink:href='#body%d' style='%s' />\n", id, _, style);
			printf("<style>\n");
			printf("#obj%d{animation:path%d %ss infinite; animation-timing-function:linear;}\n", id, id, &str(SM * t_max)[0]);
			printf("@keyframes path%d {\n", id);
			for (int i = 0, N = p.size(); i < N; i++) {
				vec2 q = transform(p[i].x);
				printf(" %s%%{transform:translate(%spx,%spx)rotate(%sdeg)}\n", &str(100 * p[i].t / t_max, 2)[0], &str(q.x, 1)[0], &str(q.y, 1)[0], &str(p[i].r*(-180 / PI), 1)[0]);
			}
			printf("} </style>\n");
		};
		printPath(IntPath[_], "stroke:black;fill:none");
		printPath(AnalyticPath[_], "stroke:black;fill:none");

	}

	printf("</svg>");
	return 0;
}
