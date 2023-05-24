// test various numerical optimization methods

#include <stdio.h>
#include <vector>
#include "numerical/geometry.h"
#include "numerical/optimization.h"
#include "numerical/integration.h"


// a good-conditioned test curve, periodic with period 1
vec2 fun(double t) {
	t *= 2.*PI;
	return vec2(.0, .05)
		+ vec2(.6156, .2141)*cos(t) + vec2(-.3412, .4436)*sin(t)
		+ vec2(.1344, .1148)*cos(2.*t) + vec2(-.1224, -.146)*sin(2.*t)
		+ vec2(.08335, -.09506)*cos(3.*t) + vec2(.2634, -.06217)*sin(3.*t)
		+ vec2(-.07623, -.0758)*cos(4.*t) + vec2(-.09188, -.02987)*sin(4.*t)
		+ vec2(.01339, .2293)*cos(5.*t) + vec2(-.01866, .1629)*sin(5.*t)
		+ vec2(.1631, .005689)*cos(6.*t) + vec2(.006984, .07154)*sin(6.*t)
		+ vec2(.02867, -.02175)*cos(7.*t) + vec2(-.01512, .1169)*sin(7.*t)
		+ vec2(.00989, -.01123)*cos(8.*t) + vec2(.02405, .02682)*sin(8.*t)
		+ vec2(.002186, -.01068)*cos(9.*t);
}


// from parametric2bezier.cpp
struct cubicBezier {
	vec2 A, B, C, D;
};
struct cubicCurve {
	vec2 c0, c1, c2, c3;  // c0 + c1 t + c2 t² + c3 t³, 0 < t < 1
	vec2 eval(double t) { return c0 + t * (c1 + t * (c2 + t * c3)); }
};
cubicCurve bezierAlg(vec2 A, vec2 B, vec2 C, vec2 D) {
	return cubicCurve{ A, -3 * A + 3 * B, 3 * A - 6 * B + 3 * C, -A + 3 * B - 3 * C + D };
}
// calculate the square of distance to a cubic parametric curve
#include "cubicdist.h"
static uint32_t distCubic2_callCount = 0;  // a counter for testing
double distCubic2(cubicCurve c, vec2 p) {
	distCubic2_callCount++;
	return CubicCurveDistance2(&c.c0, p);
};
double shortestPointOnCurve(cubicCurve c, vec2 p) {  // return the parameter
	distCubic2_callCount++;
	double t; CubicCurveDistance2(&c.c0, p, &t);
	return t;
}


// visualization functions
struct curveInterval;
void initSVG(const char* filename);
void writeBlock(const std::vector<curveInterval> &val, const char msg[] = "");
void endSVG();





// a struct for curve fitting
struct curveInterval {
	double t0, t1;  // parametric interval
	cubicBezier fit;  // curve of best fit
};

// initial guess, split the parameter evenly
std::vector<curveInterval> initialGuess(int N) {
	std::vector<curveInterval> res;
	for (int i = 0; i < N; i++) {
		double t0 = i / (double)N, t1 = (i + 1) / (double)N;
		double dt = 1e-6*(t1 - t0);
		vec2 p0 = fun(t0), p1 = fun(t1);
		vec2 d0 = (fun(t0 + dt) - p0) / (3.*N*dt);
		vec2 d1 = (p1 - fun(t1 - dt)) / (3.*N*dt);
		res.push_back(curveInterval{ t0, t1, cubicBezier{ p0, p0 + d0, p1 - d1, p1 } });
	}
	return res;
}


// similar to fitPartCurve() function in parametric2bezier.cpp
// iterate through all curve pieces; loss function is based on the integral of the square of distance to the curve
std::vector<curveInterval> fitCurve_intError(std::vector<curveInterval> Pieces) {
	distCubic2_callCount = 0;

	int PN = Pieces.size();
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;

		// pre-computing for numerical integral
		double eps = 1e-5*dt;
		vec2 P[32]; double dL[32];
		for (int i = 0; i < 32; i++) {
			double t = t0 + NIntegrate_GL32_S[i] * dt;
			P[i] = fun(t);
			dL[i] = length(fun(t + eps) - fun(t - eps)) / (2.*eps);
			dL[i] *= NIntegrate_GL32_W[i] * dt;
		}

		// length of the curve
		double clength = 0.;
		for (int i = 0; i < 32; i++) clength += dL[i];

		// downhill simplex optimization
		// same loss function as that used in parametric2bezier.cpp
		vec2 uv(1., 1.);
		vec2 UV0[3] = {
			uv,
			uv * vec2(1.5, 1.),
			uv * vec2(1., 1.5)
		};
		int sampleCount = 0;  // counter
		uv = downhillSimplex_2d([&](vec2 uv) {
			sampleCount++;
			double penalty = (uv.x < 0. ? uv.x*uv.x : 0.) + (uv.y < 0. ? uv.y*uv.y : 0.);  // negative penalty
			cubicCurve C = bezierAlg(P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1);
			double blength = NIntegrate_GL24<double>([&](double t) { return length(C.c1 + t * (2.*C.c2 + t * 3.*C.c3)); }, 0., 1.);
			double penaltyl = (blength - clength)*(blength - clength);  // difference in curve lengths penalty
			double s(0.);
			for (int i = 0; i < 32; i++)
				s += distCubic2(C, P[i]) * dL[i];  // average error
			return s + 1.0 * penalty + penaltyl;
		}, UV0, 1e-6);
		//printf("s=%d\n", sampleCount);  // 60-80

		// output
		Pieces[i].fit = cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 };
	}

	printf("%d\n", distCubic2_callCount);
	return Pieces;
}


// iterate through all curve pieces and fit with tangent constraint
// loss function: takes equal-spaced samples and calculate the maximum error
// visually no difference but takes more samples than fitCurve_intError()
std::vector<curveInterval> fitCurve_maxError(std::vector<curveInterval> Pieces) {
	distCubic2_callCount = 0;

	int PN = Pieces.size();
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;

		// take samples
		vec2 P[32];
		for (int i = 0; i < 32; i++) {
			double t = t0 + ((i + 1) / 33.) * dt;  // assume max does not appear at endpoints
			P[i] = fun(t);
		}

		// length of the curve
		double eps = 1e-5*dt;
		double clength = NIntegrate_GL32<double>([&](double t) {
			return length(fun(t + eps) - fun(t - eps)) / (2.*eps);
		}, t0, t1);

		// downhill simplex optimization
		vec2 uv(1., 1.);
		vec2 UV0[3] = {
			uv,
			uv * vec2(1.5, 1.),
			uv * vec2(1., 1.5)
		};
		int sampleCount = 0;  // counter
		uv = downhillSimplex_2d([&](vec2 uv) {
			sampleCount++;
			double penalty = (uv.x < 0. ? uv.x*uv.x : 0.) + (uv.y < 0. ? uv.y*uv.y : 0.);  // negative penalty
			cubicCurve C = bezierAlg(P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1);
			double blength = NIntegrate_GL24<double>([&](double t) { return length(C.c1 + t * (2.*C.c2 + t * 3.*C.c3)); }, 0., 1.);
			double penaltyl = (blength - clength)*(blength - clength);  // difference in curve lengths penalty
			double maxerr(0.);
			for (int i = 0; i < 32; i++)
				maxerr = max(maxerr, distCubic2(C, P[i]));  // maximum error
			return maxerr + 1.0 * penalty + penaltyl;
		}, UV0, 1e-6);
		//printf("s=%d\n", sampleCount);  // 70-90

		// output
		Pieces[i].fit = cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 };
	}

	printf("%d\n", distCubic2_callCount);
	return Pieces;
}


// a method described in various sources found on the internet
// takes less samples; does not converge with a poor initial guess
vec2 fitCurve_minimizeParameterSumSqr(vec2 P0, vec2 P1, vec2 T0, vec2 T1, int N, vec2 P[], double t[], double w[]) {
	// this function finds u and v that minimizes Σ[w·(Bezier(t)-P)²] (parametric continuity)
	// u and v can be found analytically by solving a linear system, without distCubic2

#if 0
	// reference solution to make sure my formula is correct
	return Newton_Gradient_2d([&](vec2 uv) {
		double err = 0.;
		vec2 A = P0, B = P0 + uv.x*T0, C = P1 - uv.y*T1, D = P1;
		for (int i = 0; i < N; i++) {
			double ti = t[i];
			vec2 Ci = A + ti * (-3.*A + 3.*B + ti * (3.*A - 6.*B + 3.*C + ti * (-A + 3.*B - 3.*C + D)));
			err += w[i] * (Ci - P[i]).sqr();
		}
		return err;
	}, vec2(1., 1.));
#endif

	mat2 M(0.); vec2 b(0.);
	for (int i = 0; i < N; i++) {
		/*
			# Python 3
			from sympy import *
			P0,P1,T0,T1,u,v,t,Pi=symbols('P0,P1,T0,T1,u,v,t,Pi')
			A,B,C,D=P0,P0+u*T0,P1-v*T1,P1
			Curve = (1-t)**3*A+3*t*(1-t)**2*B+3*t**2*(1-t)*C+t**3*D
			E = expand((Curve-Pi)**2)
			dEdu = collect(diff(E,u)/6,(u,v,P0,P1,T0,T1))
			dEdv = collect(diff(E,v)/6,(u,v,P0,P1,T0,T1))
			#dEdu,dEdv = simplify(dEdu),simplify(dEdv)
			print(dEdu)
			print(dEdv)
		*/
		double ti = t[i], t2 = ti * ti, t3 = t2 * ti, t4 = t3 * ti, t5 = t4 * ti, t6 = t5 * ti;
		vec2 Pi = P[i];
		double k0 = dot(P0, T0)*(2.*t6 - 7.*t5 + 8.*t4 - 2.*t3 - 2.*t2 + ti)
			+ dot(P1, T0)*(-2.*t6 + 7.*t5 - 8.*t4 + 3.*t3)
			+ dot(Pi, T0)*(-t3 + 2.*t2 - ti);
		double u0 = dot(T0, T0)*(3.*t6 - 12.*t5 + 18.*t4 - 12.*t3 + 3.*t2);
		double v0 = dot(T0, T1)*(3.*t6 - 9.*t5 + 9.*t4 - 3.*t3);
		double k1 = dot(P0, T1)*(2.*t6 - 5.*t5 + 3.*t4 + t3 - t2)
			+ dot(P1, T1)*(-2.*t6 + 5.*t5 - 3.*t4)
			+ dot(Pi, T1)*(-t3 + t2);
		double u1 = dot(T0, T1)*(3.*t6 - 9.*t5 + 9.*t4 - 3.*t3);
		double v1 = dot(T1, T1)*(3.*t6 - 6.*t5 + 3.*t4);
		M += w[i] * mat2(u0, v0, u1, v1);
		b += w[i] * vec2(-k0, -k1);
	}
	return M.inverse()*b;
}
std::vector<curveInterval> fitCurve_reParameterize(std::vector<curveInterval> Pieces,
	bool weighted, bool chordLengthParameterization) {
	distCubic2_callCount = 0;

	int PN = Pieces.size();
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;

		// initial parameterization
		const int N = 32;
		vec2 Ps[N]; double ts[N], ws[N];
		for (int i = 0; i < N; i++) {
			double a = (i + 1.) / (N + 1.);  // ignore endpoints
			Ps[i] = fun(t0 + a * dt);
			ts[i] = a;
			ws[i] = 1.;
		}
		// calculate weights
		if (weighted) {
			for (int i = 0; i < N; i++) {
				vec2 p0 = i == 0 ? P0 : Ps[i - 1];
				vec2 p1 = i == N - 1 ? P1 : Ps[i + 1];
				ws[i] = length(p1 - p0) * .5*(N + 1);
				//ws[i] = length(fun(t0 + ts[i] * dt + .001) - fun(t0 + ts[i] * dt - .001)) / (.002 / dt);
			}
		}
		// chord-length parameterization
		if (chordLengthParameterization) {
			for (int i = 0; i < N; i++) {
				vec2 p0 = i == 0 ? P0 : Ps[i - 1];
				vec2 p1 = Ps[i];
				ts[i] = length(p1 - p0) + (i == 0 ? 0. : ts[i - 1]);
			}
			double tsN = ts[N - 1] + length(P1 - Ps[N - 1]);
			for (int i = 0; i < N; i++) ts[i] /= tsN;
		}

		// fitting and re-parametrization
		vec2 uv_old(NAN), uv;
		double err_old(NAN), err;
		int noimprove_count = 0;
		for (int iter = 0; iter < 64; iter++) {
			// optimize with current parametrization
			uv = fitCurve_minimizeParameterSumSqr(P0, P1, T0, T1, N, Ps, ts, ws);

			// reparameterize
			err = 0.;  // in practice: abort if still too big after some number of iterations
			cubicCurve C = bezierAlg(P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1);
			for (int i = 0; i < N; i++) {
				ts[i] = shortestPointOnCurve(C, Ps[i]);
				err = max(err, (C.eval(ts[i]) - Ps[i]).sqr());
			}

			// termination condition
			double de = abs(err_old - err);
			if (de < 1e-6) {
				if (++noimprove_count > 3) {
					//printf("%d\n", iter);
					break;
				}
			}
			else noimprove_count = 0;

			// linear convergence, slower when the parameters are "going down a valley"
			//if (iter) printf("(%d,%lf),", iter, log10(de));
			//if (iter) printf("(%lf,%lf),", uv.x, uv.y);  // travels on a straight line; accelerate using a line search maybe?

			uv_old = uv, err_old = err;
		}

		// output
		Pieces[i].fit = cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 };
	}

	printf("%d\n", distCubic2_callCount);
	return Pieces;
}


// minimize the integral of error without tangent constraints
// 3x+ more samples but the result is much better visually
// @useTangentRepresentation: set to true to represent control points using parallel and orthogonal components of tangent vectors in downhill simplex optimization
std::vector<curveInterval> fitCurve_intError_4(std::vector<curveInterval> Pieces, bool useTangentRepresentation) {
	distCubic2_callCount = 0;

	int PN = Pieces.size();
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;
		vec2 N0 = T0.rot(), N1 = T1.rot();

		// pre-computing for numerical integral
		double eps = 1e-4*dt;
		vec2 P[32]; double dL[32];
		for (int i = 0; i < 32; i++) {
			double t = t0 + NIntegrate_GL32_S[i] * dt;
			P[i] = fun(t);
			dL[i] = length(fun(t + eps) - fun(t - eps)) / (2.*eps);
			dL[i] *= NIntegrate_GL32_W[i] * dt;
		}

		// length of the curve
		double clength = 0.;
		for (int i = 0; i < 32; i++) clength += dL[i];

		// downhill simplex optimization
		// loss function is similar to that used in parametric2bezier.cpp
		double Ts[5][4] = { { 1, 0, 1, 0 } };
		double *Tp[5] = { Ts[0], Ts[1], Ts[2], Ts[3], Ts[4] };
		if (!useTangentRepresentation)
			*(vec2*)Tp[0] = T0, *(vec2*)&Tp[0][2] = T1;
		setupInitialSimplex_regular(4, Tp[0], Tp,
			useTangentRepresentation ? 0.2 : 0.25*pow(T0.sqr()*T1.sqr(), .25));
		double val[5];
		int sampleCount = 0;  // counter
		int id = downhillSimplex(4, [&](double T[4]) {
			sampleCount++;
			// curve
			vec2 R0 = useTangentRepresentation ? T[0] * T0 + T[1] * N0 : vec2(T[0], T[1]);
			vec2 R1 = useTangentRepresentation ? T[2] * T1 + T[3] * N1 : vec2(T[2], T[3]);
			cubicCurve C = bezierAlg(P0, P0 + R0, P1 - R1, P1);
			double penalty = 0.;  // negative penalty; use dot product between tangents instead maybe?
			double blength = NIntegrate_GL24<double>([&](double t) { return length(C.c1 + t * (2.*C.c2 + t * 3.*C.c3)); }, 0., 1.);
			double penaltyl = (blength - clength)*(blength - clength);  // difference in curve lengths penalty
			double s(0.);
			for (int i = 0; i < 32; i++)
				s += distCubic2(C, P[i]) * dL[i];  // average error
			return s + 1.0 * penalty + penaltyl;
		}, Tp, val, false, 1e-8, false, 10);
		//printf("s=%d\n", sampleCount);  // 200-300, larger with large step size

		// output
		Pieces[i].fit = cubicBezier{ P0,
			P0 + (useTangentRepresentation ? Tp[id][0] * T0 + Tp[id][1] * N0 : vec2(Tp[id][0], Tp[id][1])),
			P1 - (useTangentRepresentation ? Tp[id][2] * T1 + Tp[id][3] * N1 : vec2(Tp[id][2], Tp[id][3])),
			P1 };
	}

	printf("%d\n", distCubic2_callCount);
	return Pieces;
}



// similar to fitCurve_minimizeParameterSumSqr(), without tangent constraints
// 0.25x number of samples compare to fitCurve_intError_4() but the G0 at segment breaks are more obvious
void fitCurve_minimizeParameterSumSqr_4(vec2 P0, vec2 P1, vec2 &C0, vec2 &C1, int N, vec2 P[], double t[], double w[]) {
	// analytically find control points C0 and C1 to minimize Σ[w·(Bezier(t)-P)²]

#if 0
	// reference solution to make sure my formula is correct
	vec2 PX[2] = { C0, C1 };
	Newton_Iteration_Minimize(4, [&](double *X) {
		double err = 0.;
		vec2 A = P0, B = *(vec2*)X, C = ((vec2*)X)[1], D = P1;
		for (int i = 0; i < N; i++) {
			double ti = t[i];
			vec2 Ci = A + ti * (-3.*A + 3.*B + ti * (3.*A - 6.*B + 3.*C + ti * (-A + 3.*B - 3.*C + D)));
			err += w[i] * (Ci - P[i]).sqr();
		}
		return err;
	}, (double*)&PX[0], (double*)&PX[0]);
	C0 = PX[0], C1 = PX[1];
	return;
#endif

	mat2 M(0.), b(0.);
	for (int i = 0; i < N; i++) {
		double ti = t[i];
		double m0 = (1 - ti)*(1 - ti)*(1 - ti);
		double n0 = 3.*ti*(1 - ti)*(1 - ti);
		double n1 = 3.*ti*ti*(1 - ti);
		double m1 = ti * ti*ti;
		vec2 Pm = m0 * P0 + m1 * P1 - P[i];
		double u0 = n0 * n0, u1 = n0 * n1;
		double v0 = n0 * n1, v1 = n1 * n1;
		M += w[i] * mat2(u0, u1, v0, v1);
		b += w[i] * mat2(n0*Pm, n1*Pm);
	}
	b = -b * M.inverse();
	C0 = b.column(0);
	C1 = b.column(1);
}
std::vector<curveInterval> fitCurve_reParameterize_4(std::vector<curveInterval> Pieces) {
	distCubic2_callCount = 0;
	int PN = Pieces.size();
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;

		// initial parameterization
		const int N = 32;
		vec2 Ps[N]; double ts[N], ws[N];
		for (int i = 0; i < N; i++) {
			double a = (i + 1.) / (N + 1.);  // ignore endpoints
			Ps[i] = fun(t0 + a * dt);
			ts[i] = a;
			ws[i] = 1.;
		}

		// fitting and re-parametrization
		vec2 C0_old(P0 + T0), C1_old(P1 - T1), C0(C0_old), C1(C1_old);
		double err_old(NAN), err;
		int noimprove_count = 0;
		for (int iter = 0; iter < 64; iter++) {
			// optimize with current parametrization
			fitCurve_minimizeParameterSumSqr_4(P0, P1, C0, C1, N, Ps, ts, ws);

			// reparameterize
			err = 0.;  // in practice: abort if still too big after some number of iterations
			cubicCurve C = bezierAlg(P0, C0, C1, P1);
			for (int i = 0; i < N; i++) {
				ts[i] = shortestPointOnCurve(C, Ps[i]);
				err = max(err, (C.eval(ts[i]) - Ps[i]).sqr());
			}

			// termination condition
			double de = abs(err_old - err);
			if (de < 1e-7) {
				if (++noimprove_count > 3) break;
			}
			else noimprove_count = 0;
			//if (iter) printf("(%d,%lf),", iter, log10(de));  // linear convergence

			C0_old = C0, C1_old = C1, err_old = err;
		}

		// output
		Pieces[i].fit = cubicBezier{ P0, C0, C1, P1 };
	}

	printf("%d\n", distCubic2_callCount);
	return Pieces;
}




int main(int argc, char* argv[]) {
	initSVG(argv[1]);

	std::vector<curveInterval> pieces = initialGuess(8);
	writeBlock(pieces, "Initial guess");

	writeBlock(fitCurve_intError(pieces), "Quadrature of square error");
	writeBlock(fitCurve_maxError(pieces), "Max of square error");

	writeBlock(fitCurve_reParameterize(pieces, false, false), "Reparameterization");
	writeBlock(fitCurve_reParameterize(pieces, false, true), "Chord-length parameterization");
	writeBlock(fitCurve_reParameterize(pieces, true, false), "Weighted reparameterization");
	writeBlock(fitCurve_reParameterize(pieces, true, true), "Weighted chord-length parameterization");

	writeBlock(fitCurve_intError_4(pieces, false), "Quadrature without tangent constraint");
	writeBlock(fitCurve_intError_4(pieces, true), "Quadrature without tangent constraint (components)");

	writeBlock(fitCurve_reParameterize_4(pieces), "Reparameterization without tangent constraint");


	endSVG();

	return 0;
}



FILE* fp = 0;
const double scale = 150.;  // scaling from graph to svg
const int width = 360;  // width and height of each block
const int colspan = 4;  // number of blocks in a row
int blockCount = 0;  // number of blocks writed
void initSVG(const char* filename) {
	fp = fopen(filename, "wb");
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='%d' height='%d'>\n", width*colspan, 100 * width);
	fprintf(fp, "<defs><clipPath id='viewbox'><rect x='%d' y='%d' width='%d' height='%d' /></clipPath>\n", 0, 0, width, width);
	fprintf(fp, "<marker id='anchor-start' viewBox='0 0 10 10' refX='5' refY='5' orient='' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'><rect x='3.8' y='3.8' width='2.4' height='2.4' style='stroke:black;stroke-width:1px;fill:black'></rect></marker><marker id='anchor-end' viewBox='0 0 10 10' refX='5' refY='5' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'><ellipse cx='5' cy='5' rx='1.2' ry='1.2' style='stroke:black;stroke-width:1px;fill:black'></ellipse></marker></defs>\n");
	fprintf(fp, "<style>.anchors{opacity:0;}.points{opacity:0;}</style>\n");  // for testing
	fprintf(fp, "<style>.anchors{stroke:black;stroke-width:%lg;fill:none;opacity:0.4;}.points{stroke:none;fill:blue;opacity:0.5;}</style>\n", 1. / scale);

	vec2 p = fun(0.);
	fprintf(fp, "<defs><path id='refgraph' d='M%lg,%lg", p.x, p.y);
	for (double t = .005; t < 1.; t += .005) {
		p = fun(t);
		fprintf(fp, "L%lg,%lg", p.x, p.y);
	}
	fprintf(fp, "z' style='stroke:#ccc;stroke-width:3px;fill:none;' vector-effect='non-scaling-stroke'/></defs>\n");
}
void writeBlock(const std::vector<curveInterval> &val, const char message[]) {
	fprintf(fp, "<g transform='translate(%d,%d)' clip-path='url(#viewbox)'>\n",
		width*(blockCount%colspan), width*(blockCount / colspan));
	fprintf(fp, "<rect x='0' y='0' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", width, width);
	fprintf(fp, "<text x='10' y='20'>%s</text>\n", message);
	fprintf(fp, "<text x='10' y='40'>%d</text>\n", distCubic2_callCount);  // should work
	fprintf(fp, "<g transform='matrix(%lg,0,0,%lg,%lg,%lg)'>\n", scale, -scale, .5*width, .5*width);
	fprintf(fp, "<use x='0' y='0' xlink:href='#refgraph'/>\n");

	int vn = val.size();
	if (vn) {
		// splines
		fprintf(fp, "<path d='");
		for (int i = 0; i < vn; i++) {
			cubicBezier c = val[i].fit;
			fprintf(fp, "M%lf,%lf C%lf,%lf %lf,%lf %lf,%lf", c.A.x, c.A.y, c.B.x, c.B.y, c.C.x, c.C.y, c.D.x, c.D.y);
		}
		fprintf(fp, "' style='stroke:black;stroke-width:1px;fill:none;' vector-effect='non-scaling-stroke'/>\n");
		// anchor points
		fprintf(fp, "<g class='anchors' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)'>");
		for (int i = 0; i < vn; i++) {
			vec2 P0 = val[i].fit.A, Q0 = val[i].fit.B, Q1 = val[i].fit.C, P1 = val[i].fit.D;
			fprintf(fp, "<line x1='%lf' y1='%lf' x2='%lf' y2='%lf'/>", P0.x, P0.y, Q0.x, Q0.y);
			fprintf(fp, "<line x1='%lf' y1='%lf' x2='%lf' y2='%lf'/>", P1.x, P1.y, Q1.x, Q1.y);
		}
		fprintf(fp, "</g>\n");
		// endpoints
		fprintf(fp, "<g class='points'>");
		for (int i = 0; i < vn; i++) {
			fprintf(fp, "<circle cx='%lf' cy='%lf' r='%lg'/>", val[i].fit.A.x, val[i].fit.A.y, (i == 0 ? 5. : 3.) / scale);
		}
		fprintf(fp, "</g>\n");
	}

	fprintf(fp, "</g></g>");
	fflush(fp);
	blockCount++;
}
void endSVG() {
	fprintf(fp, "</svg>");
	fclose(fp);
}









// impractical methods go here

#if 0
std::vector<curveInterval> fitCurve_global(std::vector<curveInterval> Pieces) {
	distCubic2_callCount = 0;

	int PN = Pieces.size();

	// cheap way to improve the initial guess slightly
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;
		const int N = 3;
		vec2 Ps[N]; double ts[N], ws[N];
		for (int i = 0; i < N; i++) {
			double a = (i + 1.) / (N + 1.);
			Ps[i] = fun(t0 + a * dt);
			ts[i] = a, ws[i] = 1.;
		}
		vec2 uv = fitCurve_minimizeParameterSumSqr(P0, P1, T0, T1, N, Ps, ts, ws);
		Pieces[i].fit = cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 };
	}

	// pre-computing for numerical integral
	double eps = 1e-4;
	const int SAMPLE_N = 640;
	vec2 P[SAMPLE_N]; double dL[SAMPLE_N];
	for (int i = 0; i < SAMPLE_N; i++) {
		/*double t = NIntegrate_GL96_S[i];
		P[i] = fun(t);
		dL[i] = length(fun(t + eps) - fun(t - eps)) / (2.*eps);
		dL[i] *= NIntegrate_GL96_W[i];*/
		double t = 1.0 * (i + 1.) / (SAMPLE_N + 1.);
		P[i] = fun(t);
		dL[i] = length(fun(t + eps) - fun(t - eps)) / (2.*eps);
		dL[i] *= 1.0 / SAMPLE_N;
	}
	double clength = 0.;
	for (int i = 0; i < SAMPLE_N; i++) clength += dL[i];

	struct vec23 { vec2 A, B, C; };
	vec23 *Pk = new vec23[PN];
	for (int i = 0; i < PN; i++) {
		Pk[i] = vec23{ Pieces[i].fit.A, Pieces[i].fit.B, Pieces[i].fit.C };
	}
	vec23 **PkN = new vec23*[6 * PN + 1];
	for (int i = 0; i <= 6 * PN; i++) PkN[i] = new vec23[PN];
	double *val = new double[6 * PN + 1];
	setupInitialSimplex_regular(6 * PN, (double*)Pk, (double**)PkN, 0.05);

	auto loss = [&](double* vec) {
		// get all curve pieces
		vec23 *Pk = (vec23*)vec;
		std::vector<cubicCurve> algs;
		for (int i = 0; i < PN; i++) {
			algs.push_back(bezierAlg(Pk[i].A, Pk[i].B, Pk[i].C, Pk[(i + 1) % PN].A));
		}
		// quadrature of square distance
		double S = 0.;
		for (int i = 0; i < SAMPLE_N; i++) {
			double md2 = INFINITY;
			vec2 p = P[i];
			for (int j = 0; j < PN; j++) {
				double d2 = distCubic2(algs[j], p);
				md2 = min(md2, d2);
			}
			S += md2 * dL[i];
		}
		// length penalty
		double L = 0.;
		for (int i = 0; i < PN; i++) {
			cubicCurve C = algs[i];
			L += NIntegrate_GL24<double>([&](double t) { return length(C.c1 + t * (2.*C.c2 + t * 3.*C.c3)); }, 0., 1.);
		}
		return S + 0.5*(L - clength)*(L - clength);
	};

	int id = downhillSimplex(6 * PN, loss, (double**)PkN, val, false, 1e-8, true, 10, 100, 100000);
	vec23 *sol = PkN[id];
	for (int i = 0; i < PN; i++) {
		Pieces[i].fit = cubicBezier{ sol[i].A, sol[i].B, sol[i].C, sol[(i + 1) % PN].A };
	}

	printf("%d\n", distCubic2_callCount);
	writeBlock(Pieces, "Optimal solution");
	return Pieces;
}
#endif

