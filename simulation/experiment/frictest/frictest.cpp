// A block initially resting on a plane is pulled horizontally by a Hookean spring at a constant velocity.
// The coefficient of friction between the block and the plane is a function of the block's velocity μ=μ(v).
// Find the relationship between the block's velocity and time.

// Write animated SVG graph to stdout
// May not work in old browsers/viewers

// In the output graph:
// there is a blue curve and a pink curve
// the blue curve is the relation between the block's position and the block's velocity
// the pink curve is the relation between the position of the spring end (red dot) and the block's velocity
// the red dot is moving at a constant velocity
// the pulling velocity is set to 1m/s, the v-t graph should coincide with the pink curve (depend on the unit of the scale)
// the lowest horizontal dashed line indicates the unit velocity (pulling velocity)

// To-do: analytical solution for all linear cases


#include <stdio.h>
#include <string>
#include "numerical/ode.h"
#include "numerical/geometry.h"


#define sign(x) ((x)==0.?0.:(x)>0.?1.:-1.)


// object parameters - all in SI unit
namespace param {
	const double m = 0.2;  // mass of the block
	const double v = 1.0;  // pulling velocity, 1.0m/s
	const double k = 1.0;  // spring constant: tension=k*length
	const double N = 9.8*m;   // normal force
	// friction models
	const double fs = 0.5*N;  // constant static friction
	const double fk = 0.3*N;  // constant kinetic friction
	const auto f_reciprocal = [](double v) { return 0.5 / (abs(v) + 1.) * N; };
	const auto f_rational = [](double v) { return (0.5 / (abs(v) + 1.) + .1*abs(v)) * N; };
	const auto f_rational2 = [](double v) { return (0.5 / (abs(v) + 1.) + .1*v*v) * N; };
	const auto f_linear = [](double v) { return 0.2*v*N; };
	const auto f_quadratic = [](double v) { return 0.2*v*v*N; };
	const auto f_invquadratic = [](double v) { return 0.5*N / (v*v + 1.); };
	const auto f_invquadraticlinear = [](double v) { return 0.5*N / (v*v + 1.) + 0.1*v*v; };
}

// numerical solver parameters
namespace solver {
	constexpr int N = 1000;  // number of steps taken by the numerical solver
	constexpr double t_end = 10.0;  // simulation time
	constexpr double dt = t_end / N;  // simulation time step
}


struct state {
	double x;  // position
	double v;  // velocity
};
struct statet {
	double x;  // position
	double v;  // velocity
	double t;  // time
};



// calculate net force from object state

// no friction
double dsdt_nofrict(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T;
}
// constant friction (fk)
double dsdt_constfrict(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::fk;
}
// constant static and kinetic friction (fs and fk)
double dsdt_staticfrict(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*(abs(s.v) < 0.05 ? param::fs : param::fk);
}
// other friction models
double dsdt_reciprocalfrict(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_reciprocal(s.v);
}
double dsdt_rationalfrict(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_rational(s.v);
}
double dsdt_rational2frict(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_rational2(s.v);
}
double dsdt_linear(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_linear(s.v);
}
double dsdt_quadratic(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_quadratic(s.v);
}
double dsdt_invquadratic(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_invquadratic(s.v);
}
double dsdt_invquadraticlinear(state s, double t) {
	double T = param::k * (param::v*t - s.x);
	return T - sign(s.v)*param::f_invquadraticlinear(s.v);
}



// simulation function
void solve_path(double(*dsdt)(state, double), statet S[solver::N + 1]) {
	state st; st.x = 0., st.v = 0.;
	state buf0, buf1, buf2;
	for (int i = 0; i < solver::N; i++) {
		double t = i * solver::dt;
		S[i] = statet{ st.x, st.v, t };
		RungeKuttaMethod([&](const double* vec, double t, double* vecd) {
			state *s = (state*)vec, *ds = (state*)vecd;
			ds->v = dsdt(*s, t) / param::m;
			ds->x = s->v;
		}, (double*)&st, 2, t, solver::dt, (double*)&buf0, (double*)&buf1, (double*)&buf2);
	}
	S[solver::N] = statet{ st.x, st.v, solver::t_end };
}



// analytical solution to no friction cases
void nofriction_analysol(statet S[solver::N + 1]) {
	using namespace param;
	double lambda = sqrt(k / m);
	for (int i = 0; i <= solver::N; i++) {
		double t = i * solver::dt;
		S[i].x = v * t - v * sin(lambda*t) / lambda;
		S[i].v = v - v * cos(lambda*t);
		S[i].t = t;
	}
}
// previous solution function plus a time shifting
void constfriction_analysol(statet S[solver::N + 1]) {
	using namespace param;
	double lambda = sqrt(k / m);
	double shift = fk / (k*v);
	for (int i = 0; i <= solver::N; i++) {
		double t = i * solver::dt - shift;
		if (t < 0.) S[i].x = S[i].v = 0.;
		else
			S[i].x = v * t - v * sin(lambda*t) / lambda,
			S[i].v = v - v * cos(lambda*t);
		S[i].t = t + shift;
	}
}




// write animated SVG graph to stdout
namespace SVG {
	const int W = max(solver::N, 600) + 200; // image width
	const int H = 200;  // diagram height
	const int HW = 240;  // viewbox height (include scale)
}
void writeSVGBlock(const statet S[solver::N]) {
	using namespace SVG;
	//const double MAXVAL = S[solver::N].x;
	const double MAXVAL = param::v*solver::t_end;
	const double SC_X = (W - 200.) / MAXVAL;  // x-axis scaling
	const int BOX_D = int(0.4*SC_X);  // side length of the block
	char cbuf[1024];

	// pre-computed strings
	std::string poss = "";  // block positions
	std::string times = "";  // times
	for (int i = 0; i <= solver::N; i++) {
		double tp = S[i].t / solver::t_end;
		sprintf(cbuf, "%.0lf%s", S[i].x * SC_X + BOX_D, i == solver::N ? "" : ";");
		poss += std::string(cbuf);
		sprintf(cbuf, "%.3lg%s", tp, i == solver::N ? "" : ";");
		times += std::string(cbuf);
	}

	// x-v graph and x-v0 graph
	std::string x_v = "", x_v0 = "";
	for (int i = 0; i <= solver::N; i += 5) {
		double x = S[i].x*SC_X + BOX_D;
		double x0 = S[i].t*param::v*SC_X + BOX_D;
		double v = S[i].v / param::v * (0.25*H);
		sprintf(cbuf, "%c%.0lf,%.0lf", i == 0 ? 'M' : 'L', x, H - v);
		x_v += std::string(cbuf);
		sprintf(cbuf, "%c%.0lf,%.0lf", i == 0 ? 'M' : 'L', x0, H - v);
		x_v0 += std::string(cbuf);
	}
	for (int i = 0; i < 4; i++)
		printf("<line x1='%d' x2='%d' y1='%d' y2='%d' style='stroke:#ddd;stroke-width:2px;fill:none;stroke-dasharray:10 10'/>\n"
			, BOX_D, W - 100, (int)round(H - .25*i*H), (int)round(H - .25*i*H));
	printf("<path d='%s' style='stroke:#ccf;stroke-width:2px;fill:none'/>\n", &x_v[0]);
	printf("<path d='%s' style='stroke:#fcc;stroke-width:2px;fill:none'/>\n", &x_v0[0]);

	// scale
	printf("<g id='scale' style='font-size:16px'>");
	printf("<path d='M%d,%dH%d", 0, H, W);
	std::string text = "";
	for (int i = 0, imax = (int)ceil((W - BOX_D) / SC_X); i < imax; i++) {
		double x = SC_X * i + BOX_D;
		sprintf(cbuf, "<text x='%.4lg' y='%d'>%d</text>", x, H + 20, i);
		text += std::string(cbuf);
		printf("M%.4lg,%dv10", x, H);
		for (int u = 1; u < 10; u++) {
			x += 0.1*SC_X;
			if (x < W) printf("M%.4lg,%dv%d", x, H, u == 5 ? 7 : 5);
		}
	}
	printf("' style='stroke-width:1;stroke:black;fill:none;'/>\n");
	printf("%s\n", &text[0]);
	printf("</g>\n\n");

	// block
	printf("<rect x='0' y='%d' width='%d' height='%d' style='fill:#8a69ba;' transform='translate(%d 0)'>\n", H - BOX_D, BOX_D, BOX_D, -BOX_D);
	printf("<animate attributeName='x' repeatCount='indefinite'\n values='%s'\n keyTimes='%s'\n begin='0s' dur='%lgs'/>\n"
		, &poss[0], &times[0], solver::t_end);
	printf("</rect>\n\n");

	// string
	printf("<line x1='%d' y1='%d' x2='%d' y2='%d' style='stroke:#ffb400;stroke-width:5px;fill:none;'>\n", BOX_D, H - BOX_D / 2, BOX_D, H - BOX_D / 2);
	printf("<animate attributeName='x2' repeatCount='indefinite' from='%d' to='%lf' begin='0s' dur='%lgs'/>\n"
		, BOX_D, param::v*solver::t_end*SC_X + BOX_D, solver::t_end);
	printf("<animate attributeName='x1' repeatCount='indefinite'\n values='%s'\n keyTimes='%s'\n begin='0s' dur='%lgs'/>\n"
		, &poss[0], &times[0], solver::t_end);
	printf("</line>\n");

	// node
	printf("<circle cx='%d' cy='%d' r='8' style='stroke:none;fill:red;'>\n", BOX_D, H - BOX_D / 2);
	printf("<animate attributeName='cx' repeatCount='indefinite' values='%d;%lf' keyTimes='0;1' begin='0s' dur='%lgs'/>\n"
		, BOX_D, param::v*solver::t_end*SC_X + BOX_D, solver::t_end);
	printf("</circle>\n");

}
void writeSVG() {
	statet S[solver::N + 1];

	// list of friction models
	const int CASE_N = 10;
	double(*Model[CASE_N])(state, double) = {
		dsdt_nofrict,
		dsdt_constfrict,
		dsdt_staticfrict,
		dsdt_reciprocalfrict,
		dsdt_rationalfrict,
		dsdt_rational2frict,
		dsdt_linear,
		dsdt_quadratic,
		dsdt_invquadratic,
		dsdt_invquadraticlinear
	};
	const char description[CASE_N][256] = {
		"No friction",
		"Constant friction (μ=0.3)",
		"Static and kinetic friction (μ[s]=0.5 and μ[k]=0.3; threshold: 0.05m/s)",
		"μ(v) = 0.5/(v+1)",
		"μ(v) = 0.5/(v+1) + 0.1v",
		"μ(v) = 0.5/(v+1) + 0.1v²",
		"μ(v) = 0.2v",
		"μ(v) = 0.2v²",
		"μ(v) = 0.5/(v²+1)",
		"μ(v) = 0.5/(v²+1)+0.1v²"
	};

	// batch simulating
	using namespace SVG;
	int IMG_H = HW * CASE_N + 40;
	printf("<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' version='1.1' width='%d' height='%d'>\n", W, IMG_H);
	printf("<rect width='%d' height='%d' style='fill:white;stroke:none;'/>\n\n", W, IMG_H);

	int Height = 0;
	for (int T = 0; T < CASE_N; T++) {
		printf("<g transform='translate(0 %d)'>\n", Height);
		solve_path(Model[T], S);
		writeSVGBlock(S);
		printf("<text x='10' y='20'>%s</text>\n", description[T]);
		printf("</g>\n");
		Height += HW;
	}

	printf("</svg>");
}



// test function
void testAccuracy() {
	statet S[solver::N + 1], SA[solver::N + 1];
	solve_path(dsdt_constfrict, S);
	constfriction_analysol(SA);
	for (int i = 0; i <= solver::N; i++) {
		printf("%lg\t%lg\t%lg\n", abs(S[i].x - SA[i].x), abs(S[i].v - SA[i].v), abs(S[i].t - SA[i].t));
	}
}


int main(int argc, char* argv[]) {
	freopen(argv[1], "wb", stdout);

	writeSVG();

	return 0;
}

