// Batch test
// Euler:       70 ⚠ 169 ❗ 145 ✅
// Midpoint:    62 ⚠ 144 ❗ 178 ✅
// Runge-Kutta: 61 ⚠ 162 ❗ 161 ✅

// Issues: similar to EXP002
// May not be a good experiment as the variable of the analytic solution is not t
// The error is magnified significantly near the end of the simulation
// I may have a bug

#define BatchTest 1


#include <stdio.h>
#include "numerical/ode.h"
#include "numerical/geometry.h"


// parameters
vec2 A = vec2(-1, 1);
double v0 = 1.0;
double v = 2.0;
double r0 = length(A);
double phi0 = atan2(A.y, A.x);
double k = v / v0;

// simulation parameters
double dt = 0.01;
double t_max = 10.0 - 1e-8;


// derivative of state
void dxdt(const double* x, double t, double* dxdt) {
	vec2 p = *(vec2*)x;
	vec2 d = -v * normalize(p);
	*(vec2*)dxdt = d + vec2(v0, 0);
}

// given φ, calculate r and t
vec2 analytic_sol(double phi) {
	double r = r0 * pow(tan(.5*phi) / tan(.5*phi0), k) * sin(phi0) / sin(phi);
	auto Int = [](double phi) {
		return (r0 / v0) * sin(phi0)*pow(tan(.5*phi), k - 1) * ((k - 1)*pow(tan(.5*phi), 2) + k + 1) / (2 * (1 - k * k)*pow(tan(.5*phi0), k));
	};
	double t = Int(phi) - Int(phi0);
	return vec2(r, t);
}


// simulation and testing
int ALERT = 0, BAR = 0, QUESTION = 0, EXAM = 0, OK = 0;
#define ALERT_SIGN "\u26a0"
#define BAR_SIGN "\u26d4"
#define QUESTION_SIGN "\u2753"
#define EXAM_SIGN "\u2757"
#define OK_SIGN "\u2705"
void simulate() {
	vec2 p(A), rt, art;
	double temp0[2], temp1[2], temp2[2];
	double t = 0.0;
	vec2 rmse(0), maxe(0); int count = 0;

	while (t < t_max) {
		//for (int i = 0; i < 4; i++) EulersMethod(dxdt, (double*)&p, 2, t, .25*dt, temp0), t += .25*dt;
		//for (int i = 0; i < 2; i++) MidpointMethod(dxdt, (double*)&p, 2, t, .5*dt, temp0, temp1), t += .5*dt;
		RungeKuttaMethod(dxdt, (double*)&p, 2, t, dt, temp0, temp1, temp2), t += dt;
		double phi = atan2(p.y, p.x);
		double r = length(p); //if (!(r > v*dt)) break;
		rt = vec2(r, t);
		art = analytic_sol(phi);

		// check
		vec2 err = abs(rt - art);
		rmse += err * err, maxe = pMax(maxe, err);
		count++;
		if (!(err.x < 1e8 && err.y < 1e8)) break;
#if !BatchTest
		printf("<tr><td>%lf</td><td>%lf, %lf</td><td>%lf, %lf</td><td>%.3le, %.3le</td></tr>", phi, art.x, art.y, rt.x, rt.y, err.x, err.y);
#endif
	}
	rmse = sqrt(rmse / count);

#if BatchTest
	// print result
	printf("<td>");
	if (!(art.x < 1e6 && art.y < 1e6)) printf(ALERT_SIGN), ALERT++;  // unable to judge
	else if (!(t >= t_max || rt.x < v*dt)) {
		if (count == 0 || !(maxe.x < 1e6 && maxe.y < 1e6)) printf(BAR_SIGN), BAR++;  // unstable
		else printf(QUESTION_SIGN), QUESTION++;  // aborted for other reason
	}
	else {
		if (max(maxe.x, maxe.y) > 0.01) printf(EXAM_SIGN), EXAM++;  // high error
		else printf(OK_SIGN), OK++;  // ok
	}
	printf("</td>");
	printf("<td>%.2le, %.2le</td>", rmse.x, rmse.y);
	printf("<td>%.2le, %.2le</td>", maxe.x, maxe.y);
	printf("<td>%d</td>", count);
	printf("<td>%lf</td>", atan2(p.y, p.x));
	printf("<td>%lg, %lg</td>", art.x, art.y);
	printf("<td>%lg, %lg</td>", rt.x, rt.y);
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
	// values are chosen to avoid degenerated cases
	const int v0N = 4; double _v0[v0N] = { 0.01, 0.5, 1.2, 100.0 };
	const int vN = 4; double _v[vN] = { 0.1, 1.0, 5.0, 20.0 };
	const int AN = 4; vec2 _A[AN] = { vec2(-1,1), vec2(1,1), vec2(-0.1,2), vec2(-5,1) };
	const int dtN = 6; double _dt[dtN] = { 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1 };
	printf("<tr><td>Parameters</td><td>Status</td><td>RMSE (r, t)</td><td>MAXE (r, t)</td><td>Count</td><td>φ</td><td>r(φ), t(φ)</td><td>r, t</td></tr>");
	for (int Ai = 0; Ai < AN; Ai++) {
		A = _A[Ai];
		for (int v0i = 0; v0i < v0N; v0i++) {
			v0 = _v0[v0i];
			for (int vi = 0; vi < vN; vi++) {
				v = _v[vi];
				r0 = length(A), phi0 = atan2(A.y, A.x), k = v / v0;

				double t1 = analytic_sol(0.0).y;
				if (t1 > 0. && t1 < t_max) t_max = t1 - 1e-8;
				else t_max = 10.0 - 1e-8;

				for (int dti = 0; dti < dtN; dti++) {
					dt = _dt[dti];
					printf("<tr>");
					printf("<td>A=(%lg, %lg), v0=%lg, v=%lg, dt=%lg:</td>", A.x, A.y, v0, v, dt);
					simulate();
					printf("</tr>");
				}
			}
		}
	}
	printf("</table>");
	printf("<div style='margin:20px;'>%d %s %d %s %d %s %d %s %d %s</div><div><br></div>", ALERT, ALERT_SIGN, QUESTION, QUESTION_SIGN, BAR, BAR_SIGN, EXAM, EXAM_SIGN, OK, OK_SIGN);
#else
	// test using default parameters
	printf("<tr><td>φ</td><td>r(φ), t(φ)</td><td>r, t</td><td>Absolute error</td></tr>");
	simulate();
	printf("</table>");
#endif

	return 0;
}
