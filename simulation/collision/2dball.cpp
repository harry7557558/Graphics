// simulating the elastic collision of a perfect ball in 2d
// perfect ball: no angular motion

#include <stdio.h>
#include <vector>
#include <string>

#include "numerical/geometry.h"  // vec2
#include "numerical/ode.h"  // RungeKuttaMethod
#include "numerical/rootfinding.h"  // solveQuartic

// simulation boundary
// 0: rectangle; 1: circle
#define BOUNDARY 1

struct keyFrame {
	double t;
	vec2 p;
};
struct splineTo {
	vec2 c, d;
};
typedef std::vector<keyFrame> sequence;


// constants/parameters/conditions
#if BOUNDARY==0
const double W = 6.0, H = 4.0;  // width and height of simulation box;  unit: m
const vec2 BMax = vec2(0.5*W, 0.5*H), BMin = vec2(-0.5*W, -0.5*H);  // bounding box
#elif BOUNDARY==1
const double W = 6.0, H = 5.0;  // W is the diameter of the circle
const vec2 BMax = vec2(0.5*W, 0.5*H), BMin = vec2(-0.5*W, -0.5*W);  // bounding box
#endif
const double SC = 100.0;  // output scaling
const double R = 0.1;  // radius of the ball
vec2 P0 = vec2(-1, 0.5), V0 = vec2(5, 2);  // initial state
const vec2 g = vec2(0, -9.8);  // acceleration, or force for unit mass
const double t_max = 10.0;  // end time


// path calculated by numerically integrating collision force
sequence intPath;
void calcIntPath() {
	intPath.push_back(keyFrame{ 0, P0 });
	vec2 x[2] = { P0, V0 };
	double temp0[4], temp1[4], temp2[4];
	double dt = 0.0001, t = 0;
	for (int i = 0, tN = int(t_max / dt); i < tN; i++) {
		RungeKuttaMethod([](const double *x, double t, double* dxdt) {
			const double k = 1e6;  // a large number
			const vec2 *p = (const vec2*)x; vec2 *dp = (vec2*)dxdt;
			dp[0] = p[1];
			dp[1] = g;
			// calculate collision force acceleration
#if BOUNDARY==0
			vec2 gd(0.0);
			double sd = p[0].y - R - BMin.y; if (sd < 0.) gd = -sd * vec2(0, 1);
			sd = BMax.x - p[0].x - R; if (sd < 0.) gd = -sd * vec2(-1, 0);
			sd = p[0].x - R - BMin.x; if (sd < 0.) gd = -sd * vec2(1, 0);
			dp[1] += k * gd;
#elif BOUNDARY==1
			double sd = length(p[0]) + R - 0.5*W;
			if (sd > 0.) dp[1] -= k * sd * normalize(p[0]);
#endif
		}, (double*)&x[0], 4, t, dt, temp0, temp1, temp2);
		if ((t += dt) >= t_max) break;
		if (i % (int(0.02 / dt)) == 0) intPath.push_back(keyFrame{ t, x[0] });
	}
	intPath.push_back(keyFrame{ t_max, x[0] });
}

// path calculated using collision law
sequence analyticPath;
std::vector<splineTo> qPath;
void calcAnalyticPath() {
	analyticPath.push_back(keyFrame{ 0, P0 });
	vec2 p = P0, v = V0; double t = 0;
	while (t < t_max) {
		// calculate the next collision
		double dt = t_max - t;
		vec2 n = vec2(0);
#if BOUNDARY==0
		double tt = ((BMax.x - R) - p.x) / v.x;
		if (tt > 1e-6 && tt < dt) dt = tt, n = vec2(-1, 0);
		tt = ((R + BMin.x) - p.x) / v.x;
		if (tt > 1e-6 && tt < dt) dt = tt, n = vec2(1, 0);
		double a = .5*g.y, b = v.y, c = p.y - (BMin.y + R);
		tt = ((a < 0. ? -1 : 1)*sqrt(b*b - 4 * a*c) - b) / (2.*a);
		if (tt > 1e-6 && tt < dt) dt = tt, n = vec2(0, 1);
#elif BOUNDARY==1
		double c4 = .25*g.sqr(), c3 = dot(g, v), c2 = dot(g, p) + v.sqr(), c1 = 2.*dot(p, v), c0 = p.sqr() - (.5*W - R)*(.5*W - R);
		double R[4]; int N = solveQuartic(c4, c3, c2, c1, c0, R);
		for (int i = 0; i < N; i++) {
			double tt = refineRoot_quartic(c4, c3, c2, c1, c0, R[i]);
			if (tt > 1e-6 && tt < dt) dt = tt;
		}
		n = -normalize(p + v * dt + g * (.5*dt*dt));
#endif
		vec2 p1 = p + v * dt + g * (.5*dt*dt);
		vec2 v1 = v + g * dt;

		// output
		int tN = int(ceil(dt / 0.04)); double ddt = dt / tN;
		for (int i = 1; i <= tN; i++) {
			double ct = i * ddt;
			analyticPath.push_back(keyFrame{ t + ct, p + v * ct + g * (.5*ct*ct) });
		}
		qPath.push_back(splineTo{ p + .5*v*dt, p1 });

		// collision respond
		p = p1, v = v1;
		v -= 2.*dot(v, n)*n;
		t += dt;
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
#if BOUNDARY==0
	printf("<rect width='%s' height='%s' style='fill:white;stroke-width:1;stroke:black'/>\n", &str(W*SC, 1)[0], &str(H*SC, 1)[0]);
#elif BOUNDARY==1
	printf("<circle r='%s' cx='%s' cy='%s' style='fill:white;stroke-width:1;stroke:black'/>\n", &str(.5*W*SC, 1)[0], &str(.5*W*SC, 1)[0], &str(.5*H*SC, 1)[0]);
#endif

	// path initialization
	calcIntPath();
	calcAnalyticPath();

	auto transform = [](vec2 p)->vec2 {
		return SC * vec2(p.x - BMin.x, BMax.y - p.y);
	};

	// analytical path
	printf("<path stroke-dasharray='3,4' style='fill:none;stroke-width:1;stroke:gray'\n d='");
	vec2 q0 = transform(P0); printf("M%s,%s", &str(q0.x, 1)[0], &str(q0.y, 1)[0]);
	for (int i = 0, l = qPath.size(); i < l; i++) {
		vec2 q = transform(qPath[i].c);
		printf("Q%s,%s", &str(q.x, 1)[0], &str(q.y, 1)[0]);
		q = transform(qPath[i].d);
		printf(" %s,%s", &str(q.x, 1)[0], &str(q.y, 1)[0]);
	}
	printf("'/>\n");

	// animation
	auto printPath = [&](const sequence &p, const char* style) {
		vec2 q = transform(p[0].p);
		printf("<circle r='%s' cx='%d' cy='%d' style='%s'>\n", &str(R*SC, 1)[0], int(q.x + .5), int(q.y + .5), style);
		std::string values_cx, values_cy, keyTimes;
		int N = p.size();
		for (int i = 0; i < N; i++) {
			q = transform(p[i].p);
			values_cx += str(q.x, 1);
			values_cy += str(q.y, 1);
			keyTimes += str(p[i].t / t_max, 4);
			if (i + 1 != N) {
				values_cx += ';', values_cy += ';';
				keyTimes += ';';
			}
		}
		printf("<animate attributeType='CSS' attributeName='cx' repeatCount='indefinite'\n values='");
		printf(&values_cx[0]);
		printf("'\n keyTimes='");
		printf(&keyTimes[0]);
		printf("'\n begin='%ss' dur='%ss'/>\n", "0", &str(t_max)[0]);
		printf("<animate attributeType='CSS' attributeName='cy' repeatCount='indefinite'\n values='");
		printf(&values_cy[0]);
		printf("'\n keyTimes='");
		printf(&keyTimes[0]);
		printf("'\n begin='%ss' dur='%ss'/>\n", "0", &str(t_max)[0]);
		printf("</circle>\n");
	};
	printPath(analyticPath, "fill:blue");
	printPath(intPath, "fill:red");

	printf("<text x='20' y='30'>Blue ball: path is calculated using collision law;</text>\n");
	printf("<text x='20' y='50'>Red ball: path is calculated by integrating collision force numerically;</text>\n");
	//printf("<text x='20' y='30'>Blue ball: path is calculated using projectile motion and collision law;</text>\n");
	//printf("<text x='20' y='50'>Red ball: path is calculated by integrating gravitational and collision forces numerically;</text>\n");
	printf("</svg>");
	return 0;
}
