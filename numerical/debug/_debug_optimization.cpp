

#include "optimization.h"
#include "random.h"
#include <stdio.h>

// test numerical differentiation
// this function helped me discover and kill 3 bugs ;)
void testNGrad() {
	// test function, one should be enough
	auto testF = [&](double x[4]) {
		double a = x[0], b = x[1], c = x[2], d = x[3];
		double res = 1.23456789;
		res += a * a*b + a * b*c + d * d* b*b + c - c * d + 2.*b*b;
		res += sin(a)*cos(d) - b * tanh(c) + sin(c) - 0.7*cosh(d);
		res += asinh(PI*a) - atan(b*c);
		res += b * exp(d) + log(abs(a)) / log(abs(c));
		res += sqrt(b*b + 1) + 3.2*log(a*a + b * b + c * c + d * d*d*d + 1);
		res += pow(c*c + d * d + 1.6, a*a + b * b + 0.7) + erf(d);
		return res;
	};

	// Thank you, WolframAlpha!
	auto gradF = [&](const double x[4], double g[4]) {
		double a = x[0], b = x[1], c = x[2], d = x[3], a2 = a * a, b2 = b * b, c2 = c * c, d2 = d * d, d3 = d2 * d, d4 = d2 * d2;
		g[0] = 2 * a*b + b * c + (6.4*a) / (1 + a2 + b2 + c2 + d4) + PI / sqrt(1 + a2 * PI*PI) + cos(a)*cos(d) + 1 / (a*log(abs(c))) + 2 * a*pow(1.6 + c2 + d2, 0.7 + a2 + b2)* log(1.6 + c2 + d2);
		g[1] = a2 + 4 * b + b / sqrt(1 + b2) + a * c - c / (1 + b2 * c2) + 2 * b*d2 + (6.4*b) / (1 + a2 + b2 + c2 + d4) + exp(d) + 2 * b*pow(1.6 + c2 + d2, 0.7 + a2 + b2)* log(1.6 + c2 + d2) - tanh(c);
		g[2] = 1 + a * b - b / (1 + b2 * c2) - d + 2 * (0.7 + a2 + b2)*c* pow(1.6 + c2 + d2, -0.3 + a2 + b2) + (6.4*c) / (1 + a2 + b2 + c2 + d4) + cos(c) - log(abs(a)) / (c*pow(log(abs(c)), 2)) - b * pow(1. / cosh(c), 2);
		g[3] = -c + 2 * b2*d + 2 * (0.7 + a2 + b2)*d* pow(1.6 + c2 + d2, -0.3 + a2 + b2) + (12.8*d3) / (1 + a2 + b2 + c2 + d4) + b * exp(d) + 2 / (exp(d2)*sqrt(PI)) - sin(a)*sin(d) - 0.7*sinh(d);
	};
	auto grad2F = [&](const double x[4], double H[4][4]) {
		double a = x[0], b = x[1], c = x[2], d = x[3], a2 = a * a, b2 = b * b, c2 = c * c, d2 = d * d, d3 = d2 * d, d4 = d2 * d2;
		H[0][0] = 2 * b - (12.8*a2) / pow(1 + a2 + b2 + c2 + d4, 2) + 6.4 / (1 + a2 + b2 + c2 + d4) - (a*PI*PI*PI) / pow(1 + a2 * PI*PI, 1.5) - 1 / (a2*log(abs(c))) + 2 * pow(1.6 + c2 + d2, 0.7 + a2 + b2)* log(1.6 + c2 + d2) + 4 * a2*pow(1.6 + c2 + d2, 0.7 + a2 + b2)* pow(log(1.6 + c2 + d2), 2) - cos(d)*sin(a);
		H[1][1] = 4 - b2 / pow(1 + b2, 1.5) + 1 / sqrt(1 + b2) + (2 * b*c*c2) / pow(1 + b2 * c2, 2) + 2 * d2 - (12.8*b2) / pow(1 + a2 + b2 + c2 + d4, 2) + 6.4 / (1 + a2 + b2 + c2 + d4) + 2 * pow(1.6 + c2 + d2, 0.7 + a2 + b2)* log(1.6 + c2 + d2) + 4 * b2*pow(1.6 + c2 + d2, 0.7 + a2 + b2)* pow(log(1.6 + c2 + d2), 2);
		H[2][2] = (2 * b2*b*c) / pow(1 + b2 * c2, 2) + 4 * (-0.3 + a2 + b2)* (0.7 + a2 + b2)*c2* pow(1.6 + c2 + d2, -1.3 + a2 + b2) + 2 * (0.7 + a2 + b2)* pow(1.6 + c2 + d2, -0.3 + a2 + b2) - (12.8*c2) / pow(1 + a2 + b2 + c2 + d4, 2) + 6.4 / (1 + a2 + b2 + c2 + d4) + (2 * log(abs(a))) / (c2*pow(log(abs(c)), 3)) + log(abs(a)) / (c2*pow(log(abs(c)), 2)) - sin(c) + 2 * b*pow(1. / cosh(c), 2)*tanh(c);
		H[3][3] = 2 * b2 + 4 * (-0.3 + a2 + b2)* (0.7 + a2 + b2)*d2* pow(1.6 + c2 + d2, -1.3 + a2 + b2) + 2 * (0.7 + a2 + b2)* pow(1.6 + c2 + d2, -0.3 + a2 + b2) - (51.2*pow(d, 6)) / pow(1 + a2 + b2 + c2 + d4, 2) + (38.4*d2) / (1 + a2 + b2 + c2 + d4) + b * exp(d) - (4 * d) / (exp(d2)*sqrt(PI)) - 0.7*cosh(d) - cos(d)*sin(a);
		H[0][1] = H[1][0] = 2 * a + c - (12.8*a*b) / pow(1 + a2 + b2 + c2 + d4, 2) + 4 * a*b*pow(1.6 + c2 + d2, 0.7 + a2 + b2)* pow(log(1.6 + c2 + d2), 2);
		H[0][2] = H[2][0] = b + 4 * a*c*pow(1.6 + c2 + d2, -0.3 + a2 + b2) - (12.8*a*c) / pow(1 + a2 + b2 + c2 + d4, 2) - 1 / (a*c*pow(log(abs(c)), 2)) + 4 * a*(0.7 + a2 + b2)*c* pow(1.6 + c2 + d2, -0.3 + a2 + b2)* log(1.6 + c2 + d2);
		H[0][3] = H[3][0] = 4 * a*d*pow(1.6 + c2 + d2, -0.3 + a2 + b2) - (25.6*a*d3) / pow(1 + a2 + b2 + c2 + d4, 2) + 4 * a*(0.7 + a2 + b2)*d* pow(1.6 + c2 + d2, -0.3 + a2 + b2)* log(1.6 + c2 + d2) - cos(a)*sin(d);
		H[1][2] = H[2][1] = a + (2 * b2*c2) / pow(1 + b2 * c2, 2) - 1 / (1 + b2 * c2) + 4 * b*c*pow(1.6 + c2 + d2, -0.3 + a2 + b2) - (12.8*b*c) / pow(1 + a2 + b2 + c2 + d4, 2) + 4 * b*(0.7 + a2 + b2)*c* pow(1.6 + c2 + d2, -0.3 + a2 + b2)* log(1.6 + c2 + d2) - pow(1. / cosh(c), 2);
		H[1][3] = H[3][1] = 4 * b*d + 4 * b*d*pow(1.6 + c2 + d2, -0.3 + a2 + b2) - (25.6*b*d3) / pow(1 + a2 + b2 + c2 + d4, 2) + exp(d) + 4 * b*(0.7 + a2 + b2)*d* pow(1.6 + c2 + d2, -0.3 + a2 + b2)* log(1.6 + c2 + d2);
		H[2][3] = H[3][2] = -1 + 4 * (-0.3 + a2 + b2)* (0.7 + a2 + b2)*c*d* pow(1.6 + c2 + d2, -1.3 + a2 + b2) - (25.6*c*d3) / pow(1 + a2 + b2 + c2 + d4, 2);
	};

	for (int i = 0; i < 20; i++) {
		_SRAND(i);
		double x[4] = { randf_n(2), randf_n(2), randf_n(2), randf_n(2) };
		double grad[4], grada[4];
		NGrad(4, testF, x, grad);
		gradF(x, grada);
		for (int i = 0; i < 4; i++) {
			printf("%lf ", abs(grad[i] / grada[i] - 1.0));
		}
		printf("\n");
		double Hessian[4][4], HessianA[4][4];
		NGrad2(4, testF, x, 0, 0, &Hessian[0][0]);
		grad2F(x, HessianA);
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				printf("%lf ", abs(Hessian[i][j] / HessianA[i][j] - 1.0));
			}
			printf("\n");
		}
		printf("\n");
	}
}

// test numerical differentiation in higher dimensions
void testNGradN(int N) {
	auto testF = [&](const double *x) {
		// not standard Rosenbrock function
		double res = 0.0;
		for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (j != i) {
			double a = 100.*(2.*hashf(i, j) - 1.), b = 2.*hashf(j, i) - 1.;
			double u = x[j] - x[i] * x[i], v = b - x[i];
			res += a * u*u + v * v;
		}
		return res;
	};
	auto gradF = [&](const double *x, double *g) {
		for (int i = 0; i < N; i++) g[i] = 0;
		for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (j != i) {
			double a = 100.*(2.*hashf(i, j) - 1.), b = 2.*hashf(j, i) - 1.;
			double u = x[j] - x[i] * x[i], v = b - x[i];
			g[j] += 2.*a * u;
			g[i] -= 4.*a*x[i] * u + 2.*v;
		}
	};
	auto grad2F = [&](const double *x, double *H) {
		for (int i = 0; i < N*N; i++) H[i] = 0;
		for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (j != i) {
			double a = 100.*(2.*hashf(i, j) - 1.), b = 2.*hashf(j, i) - 1.;
			H[j*N + j] += 2.*a;
			H[i*N + j] = H[j*N + i] -= 4.*a*x[i];
			H[i*N + i] += -4.*a*x[j] + 12.*a*x[i] * x[i] + 2.;
		}
	};

	double *x = new double[N];
	double *grad = new double[N], *grada = new double[N];
	double *Hessian = new double[N*N], *HessianA = new double[N*N];
	for (int i = 0; i < 20; i++) {
		_SRAND(i);
		for (int i = 0; i < N; i++) x[i] = randf_n(5);
		NGrad(N, testF, x, grad);
		gradF(x, grada);
		for (int i = 0; i < N; i++) {
			printf("%lf ", abs(grad[i] / grada[i] - 1.0));
		}
		printf("\n");
		NGrad2(N, testF, x, 0, 0, Hessian, 1e-3);
		grad2F(x, HessianA);
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				printf("%lf ", abs(Hessian[i*N + j] / HessianA[i*N + j] - 1.0));
			}
			printf("\n");
		}
		printf("\n");
	}
	delete x;
	delete grad; delete grada;
	delete Hessian; delete HessianA;
}



// test functions for optimization in arbitrary dimensions

// Σ ai x[i]²
void testOptimize_Sphere(int N) {
	for (int T = 0; T < 10; T++) {
		int Calls = 0;
		auto F = [&](const double *x)->double {
			Calls++;
			double res = 0.;
			for (int i = 0; i < N; i++) {
				double a = erfinv(abs(hashf(T, i)));
				res += a * x[i] * x[i];
			}
			return res;
		};
		_SRAND(T);
		double *x = new double[N];
		for (int t = 0; t < 10; t++) {
			for (int i = 0; i < N; i++) x[i] = randf_n(5.0);
			Calls = 0;
			Newton_Iteration_Minimize(N, F, x, x);
			printf("%d ", Calls);
			for (int i = 0; i < N; i++) printf("%lf ", x[i]);
			printf("\n");
		}
		delete x;
	}
}

// Σ ai(x[2i+1]-x[2i]²)²+bi(1-x[2i])²   not standard Rosenbrock
void testOptimize_Rosenbrock(int N) {
	int D = 2 * N;
	for (int T = 0; T < 10; T++) {
		int Calls = 0;
		double *a = new double[N], *b = new double[N];
		for (int i = 0; i < N; i++) {
			a[i] = 100 * (hashf(T, i) + 1.);
			b[i] = hashf(i, T) + 1.;
		}
		auto F = [&](const double *x)->double {
			Calls++;
			double res = 0;
			for (int i = 0; i < N; i++) {
				double u = x[2 * i + 1] - x[2 * i] * x[2 * i];
				double v = 1.0 - x[2 * i];
				res += a[i] * u*u + b[i] * v*v;
			}
			return res;
		};
		double *x = new double[D];
		for (int t = 0; t < 10; t++) {
			_SRAND(t);
			for (int i = 0; i < D; i++) x[i] = randf_n(1.0);
			Calls = 0;
			Newton_Iteration_Minimize(D, F, x, x);
			printf("%4d  ", Calls);
			for (int i = 0; i < D; i++) printf("%.4lf ", x[i]);
			if (abs(x[0] - 1.) > 1e-4) printf("%d %d", T, t);
			printf("\n");
		}
		delete x;
		delete a; delete b;
	}
}

// (Σxi²-r²)²+4(r²-N)Σxi   N>1, 0.75N<r²<N
// one minimum at xi=1; one maximum surrounded by a valley and one saddle point inside the valley
void testOptimize_Ring(int N) {
	for (int T = 0; T < 10; T++) {
		_SRAND(10 * N + T);
		double r = randf(sqrt(.75*N), sqrt(N)), r2 = r * r;
		int Calls = 0;
		auto F = [&](const double *x)->double {
			Calls++;
			double s2 = 0, s = 0;
			for (int i = 0; i < N; i++) {
				s2 += x[i] * x[i], s += x[i];
			}
			return (s2 - r2)*(s2 - r2) + 4 * (r2 - N) * s;
		};
		double *x = new double[N];
		for (int t = 0; t < 10; t++) {
			_SRAND(10 * N + t);
			for (int i = 0; i < N; i++) x[i] = randf_n(0.1) - (hashf(T, t) > 0. ? 1.0 : 0.0);  // close to maxima/saddle
			Calls = 0;
			Newton_Iteration_Minimize(N, F, x, x, true);
			printf("%4d  ", Calls);
			for (int i = 0; i < N; i++) printf("%.4lf ", x[i]);
			if (abs(x[0] - 1.) > 1e-4) printf("%d %d", T, t);
			printf("\n");
		}
		delete x;
	}
}


int main() {
	for (int N = 2; N <= 30; N++) testOptimize_Ring(N);
	return 0;
}

