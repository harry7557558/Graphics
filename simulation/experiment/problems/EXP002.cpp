// Batch test
// Euler:       30 ⚠ 0 ⛔ 66 ❗ 0 ✅
// Midpoint:    1 ⚠ 0 ⛔ 57 ❗ 38 ✅
// Runge-Kutta: 3 ⚠ 0 ⛔ 59 ❗ 34 ✅
// Biggest issue: losing accuracy at certain point

// Note:
// "⚠" is considered to be unstable as it is caused by "overflow", a difference from EXP001
// Runge-Kutta's accuracy is highest compare to other methods; this is obvious when setting BatchTest to 0
// This may not be a good experiment as the reference solution calculates y and t from x instead of x and y from t

#define BatchTest 1


#include <stdio.h>
#include "numerical/ode.h"
#include "numerical/geometry.h"


// parameters
double vf = 1.26;
double v = 3.24;
double L = 1.5;
double r = vf / v;  // γ ??

// simulation parameters
double dt = 0.01;
double t_max = 10.0 - 1e-8;


// derivative of state
void dxdt(const double* x, double t, double* dxdt) {
	vec2 p = *(vec2*)x;
	vec2 d = normalize(vec2(L, vf*t) - p);
	*(vec2*)dxdt = v * d;
}

// given x, calculate analytic y and t
vec2 analytic_sol(double x) {
	double y = .5*L*(1. / (1 + r)*pow(1 - x / L, 1 + r) - 1 / (1 - r)*pow(1 - x / L, 1 - r)) + L * r / (1 - r * r);
	double t = L / v * (1 / (1 - r * r) - .5*(1 / (1 + r)*pow(1 - x / L, 1 + r) + 1 / (1 - r)*pow(1 - x / L, 1 - r)));
	return vec2(y, t);
}


// simulation and testing
int ALERT = 0, BAR = 0, EXAM = 0, OK = 0;
#define ALERT_SIGN "\u26a0"
#define BAR_SIGN "\u26d4"
#define EXAM_SIGN "\u2757"
#define OK_SIGN "\u2705"
void simulate() {
	vec2 s(0), yt, ayt;
	double temp0[2], temp1[2], temp2[2];
	double t = 0.0;
	vec2 emle(0), maxe(-INFINITY); int count = 0;  // hehe

	t_max = L / (v*(1 - r * r));
	while (t < t_max) {
		//for (int i = 0; i < 4; i++) EulersMethod(dxdt, (double*)&s, 2, t, .25*dt, temp0), t += .25*dt;
		//for (int i = 0; i < 2; i++) MidpointMethod(dxdt, (double*)&s, 2, t, .5*dt, temp0, temp1), t += .5*dt;
		RungeKuttaMethod(dxdt, (double*)&s, 2, t, dt, temp0, temp1, temp2), t += dt;
		yt = vec2(s.y, t);
		ayt = analytic_sol(s.x);

		// check
		vec2 e = log(abs(yt / ayt - vec2(1)));  // compare size, log scale
		emle += e, maxe = pMax(maxe, e); count++;
		if (!(e.x < 16 && e.y < 16)) break;
#if !BatchTest
		printf("<tr><td>%lf</td><td>%lf, %lf</td><td>%lf, %lf</td><td>%.3le, %.3le</td></tr>", s.x, ayt.x, ayt.y, yt.x, yt.y, exp(e.x), exp(e.y));
#endif
	}
	emle = exp(emle / count);
	maxe = exp(maxe);

#if 1
	// print result
	printf("<td>");
	if (!(ayt.x < 1e6 && ayt.y < 1e6)) printf(ALERT_SIGN), ALERT++;  // unstable
	else if (!(t >= t_max)) printf(BAR_SIGN), BAR++;  // aborted
	else {
		if (max(maxe.x, maxe.y) > 0.01) printf(EXAM_SIGN), EXAM++;  // high error
		else printf(OK_SIGN), OK++;  // ok
	}
	printf("</td>");
	printf("<td>%.2le, %.2le</td>", emle.x, emle.y);
	printf("<td>%.2le, %.2le</td>", maxe.x, maxe.y);
	printf("<td>%d</td>", count);
	printf("<td>%lg%s</td>", t, t >= t_max ? "" : " (aborted)");
	printf("<td>%lg, %lg</td>", ayt.x, ayt.y);
	printf("<td>%lg, %lg</td>", yt.x, yt.y);
#endif
}


// main
int main(int argc, char** argv) {
	// output format: HTML
	freopen(argv[1], "w", stdout);
	printf("<style>td{padding:0px 10px;white-space:nowrap;}</style>");
	printf("<table>");

#if BatchTest
	// test different coefficients
	const int vN = 4;
	double _vf[vN] = { 0.1, 0.5, 0.95, 10.0 };
	double _v[vN] = { 1.0, 1.0, 1.0, 20.0 };
	const int LN = 4; double _L[LN] = { 1.0, 2.0, 5.0, 10.0 };
	const int dtN = 6; double _dt[dtN] = { 0.00001, 0.0001, 0.001, 0.01, 0.1, 1 };
	printf("<tr><td>Parameters</td><td>Status</td><td>EMLE (y, t)</td><td>MAXE (y, t)</td><td>Count</td><td>End x</td><td>Analytic (y, t)</td><td>Numeric (y, t)</td></tr>");
	for (int vi = 0; vi < vN; vi++) {
		vf = _vf[vi], v = _v[vi];
		r = vf / v;
		for (int Li = 0; Li < LN; Li++) {
			L = _L[Li];
			for (int dti = 0; dti < dtN; dti++) {
				dt = _dt[dti];
				printf("<tr>");
				printf("<td>vf=%lg, v=%lg, γ=%lg, dt=%lg:</td>", vf, v, r, dt);
				simulate();
				printf("</tr>");
			}
		}
	}
	printf("</table>");
	printf("<div style='margin:20px;'>%d %s %d %s %d %s %d %s</div><div><br></div>", ALERT, ALERT_SIGN, BAR, BAR_SIGN, EXAM, EXAM_SIGN, OK, OK_SIGN);
#else
	// test using default parameters
	printf("<tr><td>x</td><td>y(x), t(x)</td><td>y, t</td><td>Ey, Et</td></tr>");
	simulate();
	printf("</table>");
#endif

	return 0;
}
