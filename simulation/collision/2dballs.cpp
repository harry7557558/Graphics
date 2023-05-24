// simulating the elastic collision of some balls in 2d
// perfect ball: no angular motion

// animated svg output:
// red: analytical solution
// pink: numerical solution

#include <stdio.h>
#include <vector>
#include <string>

#include "numerical/geometry.h"  // vec2
#include "numerical/ode.h"  // RungeKuttaMethod
#include "numerical/rootfinding.h"  // solveQuartic

// debug output
#define printerr(s, ...) fprintf(stderr, s, __VA_ARGS__)


// simulation boundary
// 0: rectangle; 1: circle
#define BOUNDARY 1

struct keyFrame {
	double t;
	vec2 x;
};
typedef std::vector<keyFrame> sequence;


// parameters / small functions

#if BOUNDARY==0
const double W = 6.0, H = 4.0;  // width and height of simulation box;  unit: m
const vec2 BMax = vec2(0.5*W, 0.5*H), BMin = vec2(-0.5*W, -0.5*H);  // bounding box
#elif BOUNDARY==1
const double W = 6.0, H = 6.0;  // W is the diameter of the circle
const vec2 BMax = vec2(0.5*W, 0.5*H), BMin = vec2(-0.5*W, -0.5*W);  // bounding box
#endif
const double SC = 100.0;  // output scaling
const double SM = 2.0;  // slowing motion

const vec2 g = vec2(0, -9.8);  // gravitational acceleration
const double t_max = 10.0;  // end time of simulation

// static and dynamic parameters of a ball
struct ball {
	double r;  // radius
	double inv_m;  // reciprocal of mass
	vec2 x0, v0;  // initial position and velocity
};
struct ball_state {
	vec2 x;  // position
	vec2 v;  // velocity
};

#define Ball_N 4
ball Balls[Ball_N];
ball_state BallState[Ball_N];  // can be passed to ode solver

void init() {
	for (int i = 0; i < Ball_N; i++) {
		double r = 0.1 + 0.05*i;
		Balls[i].r = r;
		Balls[i].inv_m = 1.0 / (100.*r*r);
		Balls[i].x0 = vec2(-2. + 0.8*i, 0.2*i);
		Balls[i].v0 = vec2(5, 4 - 2 * i);
	}
	if (Ball_N == 2) {
		// test scene for the correctness of collision calculation
		Balls[0] = ball{ 0.2, 25.0, vec2(-2, -0.5), vec2(5, 3) };
		Balls[1] = ball{ 0.5, 4.0, vec2(1,1), vec2(-5, 0) };
	}
	for (int i = 0; i < Ball_N; i++) {
		BallState[i].x = Balls[i].x0;
		BallState[i].v = Balls[i].v0;
	}
}

auto pushState = [](sequence *s, double t) {
	for (int i = 0; i < Ball_N; i++)
		s[i].push_back(keyFrame{ t, BallState[i].x });
};


// path calculated by numerically integrating collision force
sequence IntPath[Ball_N];
void calcIntPath() {
	init();
	pushState(IntPath, 0);
	double temp0[4 * Ball_N], temp1[4 * Ball_N], temp2[4 * Ball_N];
	double dt = 0.0001, t = 0;
	for (int u = 0, tN = int(t_max / dt); u < tN; u++) {
		// ode solver
		RungeKuttaMethod([](const double *x, double t, double* dxdt) {
			ball_state* B = (ball_state*)x;
			ball_state* dB = (ball_state*)dxdt;
			const double k = 1e6;  // a large number
			for (int i = 0; i < Ball_N; i++) {
				dB[i].x = B[i].v;
				dB[i].v = g;
				vec2 p = B[i].x;
				double R = Balls[i].r;

				// calculate collision force with boundary
				vec2 F(0.0);
				if (BOUNDARY == 0) {  // rectangle
					double sd = p.y - R - BMin.y; if (sd < 0.) F = -sd * vec2(0, k);
					sd = BMax.x - p.x - R; if (sd < 0.) F = -sd * vec2(-k, 0);
					sd = p.x - R - BMin.x; if (sd < 0.) F = -sd * vec2(k, 0);
					sd = BMax.y - p.y - R; if (sd < 0.) F = -sd * vec2(0, -k);
				}
				else if (BOUNDARY == 1) {  // circle
					double sd = length(p) + R - 0.5*W;
					if (sd > 0.) F = -k * sd * normalize(p);
				}

				// collision force between balls
				for (int j = 0; j < Ball_N; j++) if (j != i) {
					vec2 pq = p - B[j].x;
					double d = length(pq) - (R + Balls[j].r);
					if (d < 0.) {
						F -= k * d * normalize(pq);
					}
				}

				dB[i].v += Balls[i].inv_m * F;
			}
		}, (double*)&BallState[0], 4 * Ball_N, t, dt, temp0, temp1, temp2);

		if ((t += dt) >= t_max) break;
		if (u % (int(0.02 / dt)) == 0) {
			pushState(IntPath, t);

			// check energy conservation
			/*double E = 0.0;
			for (int i = 0; i < Ball_N; i++)
				E += (.5*BallState[i].v.sqr() - dot(BallState[i].x, g)) / Balls[i].inv_m;
			printerr("%lf %lf\n", t, E);*/
		}
	}
	pushState(IntPath, t_max);
}


// path calculated using collision law
sequence AnalyticPath[Ball_N];
void calcAnalyticPath() {
	init();
	pushState(AnalyticPath, 0);
	double t = 0;
	while (t < t_max) {
		// calculate the next event
		double dt = t_max - t;
		int eventI[2] = { -1, -1 };  // index of collided objects, -1 for boundary
		vec2 dv[2] = { vec2(0), vec2(0) };  // change of velocities during the collision

		// check collisions with boundary
		for (int i = 0; i < Ball_N; i++) {
			vec2 p = BallState[i].x;
			vec2 v = BallState[i].v;
			double R = Balls[i].r;
			if (BOUNDARY == 0) {  // rectangle
				vec2 n(0.0);
				// right
				double tt = ((BMax.x - R) - p.x) / v.x;
				if (tt > 1e-6 && tt < dt) dt = tt, eventI[0] = i, n = vec2(-1, 0);
				// left
				tt = ((R + BMin.x) - p.x) / v.x;
				if (tt > 1e-6 && tt < dt) dt = tt, eventI[0] = i, n = vec2(1, 0);
				// bottom
				double a = .5*g.y, b = v.y, c = p.y - (BMin.y + R);
				tt = ((a < 0. ? -1 : 1)*sqrt(b*b - 4 * a*c) - b) / (2.*a);
				if (tt > 1e-6 && tt < dt) dt = tt, eventI[0] = i, n = vec2(0, 1);
				// top
				a = .5*g.y, b = v.y, c = p.y - (BMax.y - R);
				tt = ((a < 0. ? 1 : -1)*sqrt(b*b - 4 * a*c) - b) / (2.*a);
				if (tt > 1e-6 && tt < dt) dt = tt, eventI[0] = i, n = vec2(0, -1);
				// reflection
				if (eventI[0] == i) dv[0] = (-2.*dot(v + g * dt, n))*n;
			}
			else if (BOUNDARY == 1) {  // circle
				double c4 = .25*g.sqr(), c3 = dot(g, v), c2 = dot(g, p) + v.sqr(), c1 = 2.*dot(p, v), c0 = p.sqr() - (.5*W - R)*(.5*W - R);
				double rt[4]; int N = solveQuartic(c4, c3, c2, c1, c0, rt);
				for (int u = 0; u < N; u++) {
					double tt = refineRoot_quartic(c4, c3, c2, c1, c0, rt[u]);
					if (tt > 1e-6 && tt < dt) {
						dt = tt;
						eventI[0] = i, eventI[1] = -1;
						vec2 n = normalize(p + v * dt + g * (.5*dt*dt));
						dv[0] = (-2.*dot(v + g * dt, n))*n;
					}
				}
			}
		}

		// check collisions between objects
		for (int i = 0; i < Ball_N; i++) {
			double r1 = Balls[i].r, im1 = Balls[i].inv_m;
			vec2 p = BallState[i].x, u = BallState[i].v;
			for (int j = 0; j < i; j++) {
				double r2 = Balls[j].r, im2 = Balls[j].inv_m;
				vec2 q = BallState[j].x, v = BallState[j].v;
				vec2 pq = q - p, uv = v - u;
				// find collision time
				double a = uv.sqr(), b = dot(pq, uv), c = pq.sqr() - (r1 + r2)*(r1 + r2);
				double delta = sqrt(b*b - a * c);
				if (!isnan(delta)) {
					double tt = (-delta - b) / a;
					if (!(tt > 1e-6 && tt < dt)) tt = (delta - b) / a;  // not necessary
					if (tt > 1e-6 && tt < dt) {
						dt = tt;
						eventI[0] = i, eventI[1] = j;
						// collision reaction
						vec2 n = normalize(pq + uv * tt);  // force normal
						vec2 dp = (2.*dot(uv, n) / (im1 + im2)) * n;  // change of momentum
						dv[0] = im1 * dp;
						dv[1] = -im2 * dp;
					}
				}
			}
		}

		// output
		int tN = int(ceil(dt / 0.04)); double ddt = dt / tN;
		for (int i = 1; i <= tN; i++) {
			double ct = i * ddt;
			for (int d = 0; d < Ball_N; d++) {
				keyFrame f; f.t = t + ct, f.x = BallState[d].x + BallState[d].v * ct + g * (.5*ct*ct);
				AnalyticPath[d].push_back(f);
			}
		}

		// collision response
		for (int i = 0; i < Ball_N; i++) {
			vec2 p = BallState[i].x, v = BallState[i].v;
			BallState[i].x = p + v * dt + g * (.5*dt*dt);
			BallState[i].v = v + g * dt;
			for (int u = 0; u < 2; u++) if (i == eventI[u]) {
				BallState[i].v += dv[u];
			}
		}
		t += dt;

		// check energy conservation
		/*double E = 0.0;
		for (int i = 0; i < Ball_N; i++)
			E += (.5*BallState[i].v.sqr() - dot(BallState[i].x, g)) / Balls[i].inv_m;
		printerr("%d %d\t%lf %lf\n", eventI[0], eventI[1], t, E);*/
	}
}


// visualization
int main(int argc, char** argv) {
	// export format: animated svg
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

	// svg header
	printf("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='%d' height='%d'>\n",
		int((BMax.x - BMin.x)*SC + 1), int((BMax.y - BMin.y)*SC + 1));
	if (BOUNDARY == 0)
		printf("<rect width='%s' height='%s' style='fill:white;stroke-width:1;stroke:black'/>\n", &str(W*SC, 1)[0], &str(H*SC, 1)[0]);
	else if (BOUNDARY == 1)
		printf("<circle r='%s' cx='%s' cy='%s' style='fill:white;stroke-width:1;stroke:black'/>\n", &str(.5*W*SC, 1)[0], &str(.5*W*SC, 1)[0], &str(.5*H*SC, 1)[0]);

	// path calculation
	calcIntPath();
	calcAnalyticPath();

	auto transform = [](vec2 p)->vec2 {
		return SC * vec2(p.x - BMin.x, BMax.y - p.y);
	};

	// animation
	auto printPath = [&](const sequence &s, double R, const char* style) {
		if (s.empty()) return;
		vec2 q = transform(s[0].x);
		printf("<circle r='%s' cx='%d' cy='%d' style='%s'>\n", &str(R*SC, 1)[0], int(q.x + .5), int(q.y + .5), style);
		std::string values_cx, values_cy, keyTimes;
		int N = s.size();
		for (int i = 0; i < N; i++) {
			q = transform(s[i].x);
			values_cx += str(q.x, 1);
			values_cy += str(q.y, 1);
			keyTimes += str(s[i].t / t_max, 4);
			if (i + 1 != N) {
				values_cx += ';', values_cy += ';';
				keyTimes += ';';
			}
		}
		printf("<animate attributeType='CSS' attributeName='cx' repeatCount='indefinite'\n values='");
		printf(&values_cx[0]);
		printf("'\n keyTimes='");
		printf(&keyTimes[0]);
		printf("'\n begin='%ss' dur='%ss'/>\n", "0", &str(SM*t_max)[0]);
		printf("<animate attributeType='CSS' attributeName='cy' repeatCount='indefinite'\n values='");
		printf(&values_cy[0]);
		printf("'\n keyTimes='");
		printf(&keyTimes[0]);
		printf("'\n begin='%ss' dur='%ss'/>\n", "0", &str(SM*t_max)[0]);
		printf("</circle>\n");
	};
	for (int i = 0; i < Ball_N; i++)
		printPath(IntPath[i], Balls[i].r, "fill:rgba(240,220,180,0.6)");
	for (int i = 0; i < Ball_N; i++)
		printPath(AnalyticPath[i], Balls[i].r, "fill:red");

	printf("</svg>");
	return 0;
}
