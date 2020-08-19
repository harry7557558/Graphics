// Batch test
// Euler:       140 ⛔ 101 ❗ 420 ✅
// Midpoint:    143 ⛔ 55 ❗ 463 ✅
// Runge-Kutta: 153 ⛔ 25 ❗ 483 ✅
// Biggest issue: stability

#define BatchTest 1


#include <stdio.h>
#include "numerical/ode.h"
#include "numerical/geometry.h"

typedef struct { double x, v; } state;


// parameters
double v0 = 1.26;
double k = 3.24;
double n = 1.5;

// simulation parameters
double dt = 0.1;
double t_max = 10.0 - 1e-8;


// derivative of state
void dxdt(const double* x, double t, double* dxdt) {
	dxdt[0] = x[1];  // dx/dt = v
	dxdt[1] = -k * pow(x[1], n);  // dv/dt = -k v^n
}

// analytical solution
state analytic_sol(double t) {
	double v, x;
	if (n == 1.0) {
		v = v0 * exp(-k * t);
		x = v0 / k * (1.0 - exp(-k * t));
	}
	else if (n == 2.0) {
		v = 1.0 / (k*t + 1. / v0);
		x = (1. / k) * log(k*v0*t + 1);
	}
	else {
		v = pow(k*(n - 1)*t + pow(v0, 1 - n), 1. / (1 - n));
		x = (1. / (k*(n - 2))) * (pow(k*(n - 1)*t + pow(v0, 1 - n), (2 - n) / (1 - n)) - pow(v0, 2 - n));
	}
	if (x < 0) x = NAN;
	if (v < 0) v = NAN;
	return state{ x, v };
}


// simulation and testing
int ALERT = 0, BAR = 0, QUESTION = 0, EXAM = 0, OK = 0;
#define ALERT_SIGN "\u26a0"
#define BAR_SIGN "\u26d4"
#define QUESTION_SIGN "\u2753"
#define EXAM_SIGN "\u2757"
#define OK_SIGN "\u2705"
void simulate() {
	state s = state{ 0, v0 }, sa;
	double temp0[2], temp1[2], temp2[2];
	double t = 0.0;
	state rmse = state{ 0, 0 }, maxe = state{ 0, 0 }; int count = 0;

	while (t < t_max) {
		//for (int i = 0; i < 4; i++) EulersMethod(dxdt, (double*)&s, 2, t, .25*dt, temp0), t += .25*dt;
		//for (int i = 0; i < 2; i++) MidpointMethod(dxdt, (double*)&s, 2, t, .5*dt, temp0, temp1), t += .5*dt;
		RungeKuttaMethod(dxdt, (double*)&s, 2, t, dt, temp0, temp1, temp2), t += dt;
		sa = analytic_sol(t);

		// check
		double ev = sa.v - s.v, ex = sa.x - s.x;  // this may not be a good measure of error
		rmse.v += ev * ev, rmse.x += ex * ex;
		maxe.v = max(maxe.v, abs(ev)), maxe.x = max(maxe.x, abs(ex));
		count++;
		if (isnan(ev) || isnan(ex) || abs(ev) > 1e8 || abs(ex) > 1e8) break;
#if !BatchTest
		printf("<tr><td>%lg</td><td>%lf, %lf</td><td>%lf, %lf</td><td>%.3le, %.3le</td></tr>", t, sa.v, sa.x, s.v, s.x, abs(ev), abs(ex));
#endif
	}

#if BatchTest
	// print result
	printf("<td>");
	if (!(sa.v < 1e6 && sa.x < 1e6)) printf(ALERT_SIGN), ALERT++;  // unable to judge
	else if (!(t >= t_max)) {
		if (count == 0 || !(maxe.v < 1e6 && maxe.x < 1e6)) printf(BAR_SIGN), BAR++;  // unstable
		else printf(QUESTION_SIGN), QUESTION++;  // aborted for other reason
	}
	else {
		if (max(maxe.x, maxe.v) > 0.01) printf(EXAM_SIGN), EXAM++;  // high error
		else printf(OK_SIGN), OK++;  // ok
	}
	printf("</td>");
	printf("<td>%.2le, %.2le</td>", sqrt(rmse.v / count), sqrt(rmse.x / count));
	printf("<td>%.2le, %.2le</td>", maxe.v, maxe.x);
	printf("<td>%d</td>", count);
	printf("<td>%lg%s</td>", t, t >= t_max ? "" : " (aborted)");
	printf("<td>%lg, %lg</td>", sa.v, sa.x);
	printf("<td>%lg, %lg</td>", s.v, s.x);
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
	const int v0N = 4; double _v0[v0N] = { 0.01, 0.5, 1.0, 1000.0 };
	const int kN = 4; double _k[kN] = { 0.0001, 0.1, 1.0, 100.0 };
	const int nN = 6; double _n[nN] = { 0.5, 1.0, 2.0, 3.0, 7.5, 20.0 };
	const int dtN = 8; double _dt[dtN] = { 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0 };
	printf("<tr><td>Parameters</td><td>Status</td><td>RMSE (Ev, Ex)</td><td>MAXE (Ev, Ex)</td><td>Count</td><td>End time</td><td>Analytic (v, x)</td><td>Numeric (v, x)</td></tr>");
	for (int ni = 0; ni < nN; ni++) {
		n = _n[ni];
		for (int ki = 0; ki < kN; ki++) {
			k = _k[ki];
			for (int v0i = 0; v0i < v0N; v0i++) {
				v0 = _v0[v0i];
				if (k*(n - 1)*t_max + pow(v0, 1 - n) > 0) {  // otherwise invalid
					for (int dti = 0; dti < dtN; dti++) {
						dt = _dt[dti];
						printf("<tr>");
						printf("<td>n=%lg, k=%lg, v0=%lg, dt=%lg:</td>", n, k, v0, dt);
						simulate();
						printf("</tr>");
					}
				}
			}
		}
	}
	printf("</table>");
	printf("<div style='margin:20px;'>%d %s %d %s %d %s %d %s %d %s</div><div><br></div>", ALERT, ALERT_SIGN, QUESTION, QUESTION_SIGN, BAR, BAR_SIGN, EXAM, EXAM_SIGN, OK, OK_SIGN);
#else
	// test using default parameters
	printf("<tr><td>Time</td><td>Analytic (v, x)</td><td>Numeric (v, x)</td><td>Absolute error (v, x)</td></tr>");
	simulate();
	printf("</table>");
#endif

	return 0;
}
