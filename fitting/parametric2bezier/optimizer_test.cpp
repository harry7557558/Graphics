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
void writeBlock(const std::vector<curveInterval> &val);
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
			uv * vec2(1.1, 1.),
			uv * vec2(1., 1.1)
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
		//printf("%d\n", sampleCount);  // 60-80

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
			uv * vec2(1.1, 1.),
			uv * vec2(1., 1.1)
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
		//printf("%d\n", sampleCount);  // 70-90

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
std::vector<curveInterval> fitCurve_reParametrize(std::vector<curveInterval> Pieces) {
	distCubic2_callCount = 0;

	int PN = Pieces.size();
	for (int i = 0; i < PN; i++) {
		double t0 = Pieces[i].t0, t1 = Pieces[i].t1, dt = t1 - t0;
		cubicBezier c = Pieces[i].fit;
		vec2 P0 = c.A, T0 = c.B - c.A, T1 = c.D - c.C, P1 = c.D;

		// initial parametrization
		const int N = 32;
		vec2 Ps[N]; double ts[N], ws[N];
		for (int i = 0; i < N; i++) {
			double a = (i + 1.) / (N + 1.);  // ignore endpoints
			Ps[i] = fun(t0 + a * dt);
			ts[i] = a;
			ws[i] = 1.;
		}

		// fitting and re-parametrization
		vec2 uv_old(NAN), uv;
		double err_old(NAN), err;
		int noimprove_count = 0;
		for (int iter = 0; iter < 64; iter++) {
			// optimize with current parametrization
			uv = fitCurve_minimizeParameterSumSqr(P0, P1, T0, T1, N, Ps, ts, ws);

			// re-parametrize
			err = 0.;  // in practice: abort if still too big after some number of iterations
			cubicCurve C = bezierAlg(P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1);
			for (int i = 0; i < N; i++) {
				ts[i] = shortestPointOnCurve(C, Ps[i]);
				err = max(err, (C.eval(ts[i]) - Ps[i]).sqr());
			}

			// termination condition
			double de = abs(err_old - err);
			if (de < 1e-6) {
				if (++noimprove_count > 3) break;
			}
			else noimprove_count = 0;
			//if (iter) printf("(%d,%lf),", iter, log10(de));  // linear convergence

			uv_old = uv, err_old = err;
		}

		// output
		Pieces[i].fit = cubicBezier{ P0, P0 + T0 * uv.x, P1 - T1 * uv.y, P1 };
	}

	printf("%d\n", distCubic2_callCount);
	return Pieces;
}







int main(int argc, char* argv[]) {
	initSVG(argv[1]);

	std::vector<curveInterval> pieces = initialGuess(10);
	writeBlock(pieces);

	writeBlock(fitCurve_intError(pieces));
	writeBlock(fitCurve_maxError(pieces));
	writeBlock(fitCurve_reParametrize(pieces));

	endSVG();

	return 0;
}



FILE* fp = 0;
const double scale = 150.;  // scaling from graph to svg
const int width = 360;  // width and height of each block
const int colspan = 3;  // number of blocks in a row
int blockCount = 0;  // number of blocks writed
void initSVG(const char* filename) {
	fp = fopen(filename, "wb");
	fprintf(fp, "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' width='%d' height='%d'>\n", width*colspan, 100 * width);
	fprintf(fp, "<defs><clipPath id='viewbox'><rect x='%d' y='%d' width='%d' height='%d' /></clipPath></defs>\n", 0, 0, width, width);
	fprintf(fp, "<defs><marker id='anchor-start' viewBox='0 0 10 10' refX='5' refY='5' orient='' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'><rect x='3.8' y='3.8' width='2.4' height='2.4' style='stroke:black;stroke-width:1px;fill:black'></rect></marker><marker id='anchor-end' viewBox='0 0 10 10' refX='5' refY='5' markerUnits='strokeWidth' markerWidth='10' markerHeight='10'><ellipse cx='5' cy='5' rx='1.2' ry='1.2' style='stroke:black;stroke-width:1px;fill:black'></ellipse></marker></defs>\n");

	vec2 p = fun(0.);
	fprintf(fp, "<defs><path id='refgraph' d='M%lg,%lg", p.x, p.y);
	for (double t = .005; t < 1.; t += .005) {
		p = fun(t);
		fprintf(fp, "L%lg,%lg", p.x, p.y);
	}
	fprintf(fp, "z' style='stroke:#ccc;stroke-width:3px;fill:none;' vector-effect='non-scaling-stroke'/></defs>\n");
}
void writeBlock(const std::vector<curveInterval> &val) {
	fprintf(fp, "<g transform='translate(%d,%d)' clip-path='url(#viewbox)'>\n",
		width*(blockCount%colspan), width*(blockCount / colspan));
	fprintf(fp, "<rect x='0' y='0' width='%d' height='%d' style='stroke-width:1px;stroke:black;fill:white;'/>\n", width, width);
	fprintf(fp, "<text x='10' y='20'>#%d</text>\n", blockCount + 1);
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
		fprintf(fp, "<g class='anchors' marker-start='url(#anchor-start)' marker-end='url(#anchor-end)' style='stroke:black;stroke-width:%lg;fill:none;opacity:0.4;'>", 1. / scale);
		for (int i = 0; i < vn; i++) {
			vec2 P0 = val[i].fit.A, Q0 = val[i].fit.B, Q1 = val[i].fit.C, P1 = val[i].fit.D;
			fprintf(fp, "<line x1='%lf' y1='%lf' x2='%lf' y2='%lf'/>", P0.x, P0.y, Q0.x, Q0.y);
			fprintf(fp, "<line x1='%lf' y1='%lf' x2='%lf' y2='%lf'/>", P1.x, P1.y, Q1.x, Q1.y);
		}
		fprintf(fp, "</g>\n");
		// endpoints
		fprintf(fp, "<g class='points'>");
		for (int i = 0; i < vn; i++) {
			fprintf(fp, "<circle cx='%lf' cy='%lf' r='%lg' style='stroke:none;fill:blue;opacity:0.5;'/>", val[i].fit.A.x, val[i].fit.A.y, (i == 0 ? 5. : 3.) / scale);
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

