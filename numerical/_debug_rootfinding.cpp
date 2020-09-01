
#include <stdio.h>
#include "random.h"
#include "rootfinding.h"

// compile error with GCC
#undef min
#undef max

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;


// Visual Studio: 8.07M/s
// GCC Ofast: 4.92M/s
void test_solveCubic() {
	const int N = 0x1000000;
	// set up random numbers
	double *c2 = new double[N], *c1 = new double[N], *c0 = new double[N];
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		c2[i] = randf(-20, 20), c1[i] = randf(-20, 20), c0[i] = randf(-20, 20);
	}
	// solve equation
	double *r = new double[N], *u = new double[N], *v = new double[N];
	bool *n = new bool[N];
	auto t0 = NTime::now();
	for (int i = 0; i < N; i++) {
		n[i] = solveCubic(1, c2[i], c1[i], c0[i], r[i], u[i], v[i]);
		r[i] = refineRoot_cubic(1, c2[i], c1[i], c0[i], r[i]);
		if (n[i]) {
			u[i] = refineRoot_cubic(1, c2[i], c1[i], c0[i], u[i]);
			v[i] = refineRoot_cubic(1, c2[i], c1[i], c0[i], v[i]);
		}
	}
	double time_elapsed = fsec(NTime::now() - t0).count();
	printf("%.2lfM/s\n", N / time_elapsed / 1e6);
	// check
	for (int i = 0; i < N; i++) {
		auto checkroot = [&](double c2, double c1, double c0, double r) {
			double y = c0 + r * (c1 + r * (c2 + r));
			double dy = c1 + r * (2.*c2 + r * 3.);
			if (!(abs(y / dy) < 1e-12)) {
				fprintf(stderr, "%lg\n", y / dy);
			}
		};
		checkroot(c2[i], c1[i], c0[i], r[i]);
		if (n[i]) checkroot(c2[i], c1[i], c0[i], u[i]), checkroot(c2[i], c1[i], c0[i], v[i]);
	}
	delete c2; delete c1; delete c0;
	delete r; delete u; delete v; delete n;
}

// Visual Studio: 7.08M/s
// GCC Ofast: 5.20M/s
void test_solveQuartic() {
	const int N = 0x1000000;
	// set up random numbers
	double *c3 = new double[N], *c2 = new double[N], *c1 = new double[N], *c0 = new double[N];
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		c3[i] = randf(-20, 20), c2[i] = randf(-20, 20), c1[i] = randf(-20, 20), c0[i] = randf(-20, 20);
	}
	// solve equation
	double *r = new double[4 * N]; int *n = new int[N];
	auto t0 = NTime::now();
	for (int i = 0; i < N; i++) {
		auto p = &r[4 * i];
		n[i] = solveQuartic(1, c3[i], c2[i], c1[i], c0[i], p);
		for (int u = 0; u < n[i]; u++) {
			p[u] = refineRoot_quartic(1, c3[i], c2[i], c1[i], c0[i], p[u]);
		}
	}
	double time_elapsed = fsec(NTime::now() - t0).count();
	printf("%.2lfM/s\n", N / time_elapsed / 1e6);
	// check
	for (int i = 0; i < N; i++) {
		auto checkroot = [&](double c3, double c2, double c1, double c0, double r) {
			double y = c0 + r * (c1 + r * (c2 + r * (c3 + r)));
			double dy = c1 + r * (2.*c2 + r * (3.*c3 + r * 4.));
			if (!(abs(y / dy) < 1e-10)) {
				fprintf(stderr, "%lg\n", y / dy);
			}
		};
		for (int u = 0; u < n[i]; u++) {
			checkroot(c3[i], c2[i], c1[i], c0[i], r[4 * i + u]);
		}
	}
	delete c3; delete c2; delete c1; delete c0;
	delete r; delete n;
}


// Visual Studio: 0.95M/s
// GCC Ofast: 0.53M/s
// there is no guarantee that the solver does not miss a root or find an unintended root
void test_solveQuadraticTrig() {
	const int N = 0x1000000;
	// set up random numbers
	double *a = new double[N], *b = new double[N], *c = new double[N];
	double *c1 = new double[N], *c2 = new double[N], *w = new double[N];
	double *x0 = new double[N];
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		a[i] = randf(-20, 20), b[i] = randf(-20, 20), c[i] = randf(-20, 20);
		vec2 p = rand2_n(10.0); c1[i] = p.x, c2[i] = p.y;
		w[i] = randf_n(10.); x0[i] = randf_n(10.0);
	}
	// solve equation
	double *r = new double[N];
	auto t0 = NTime::now();
	for (int i = 0; i < N; i++) {
		r[i] = solveTrigQuadratic(a[i], b[i], c[i], c1[i], c2[i], w[i], x0[i]);
		r[i] = refineRoot_TrigQuadratic(a[i], b[i], c[i], c1[i], c2[i], w[i], r[i]);
	}
	double time_elapsed = fsec(NTime::now() - t0).count();
	printf("%.2lfM/s\n", N / time_elapsed / 1e6);
	// check
	for (int i = 0; i < N; i++) {
		auto checkroot = [&](double a, double b, double c, double c1, double c2, double w, double x) {
			double y = (a*x + b)*x + c + c1 * cos(w*x) + c2 * sin(w*x);
			double dy = 2 * a*x + b - c1 * w*sin(w*x) + c2 * w*cos(w*x);
			if (!(abs(y / dy) < 1e-10)) {
				fprintf(stderr, "%lg\n", y / dy);
				return false;
			}
			return true;
		};
		if (!isnan(r[i])) {
			if (r[i] < x0[i]) fprintf(stderr, "LE%lf\n", r[i]);
			else if (!checkroot(a[i], b[i], c[i], c1[i], c2[i], w[i], r[i])) {
				//fprintf(stderr, "%d\n", i);
			}
		}
	}
	delete a; delete b; delete c;
	delete c1; delete c2; delete w;
	delete x0; delete r;
}


double solveTrigPoly_slow(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x_min) {
	double m = sqrt(c1 * c1 + c2 * c2);
	auto eval = [&](double x) {
		return k0 + x * (k1 + x * (k2 + x * (k3 + x * k4))) + c1 * cos(w*x) + c2 * sin(w*x);
	};
	auto evald = [&](double x) {
		return k1 + x * (2 * k2 + x * (3 * k3 + x * 4 * k4)) - w * c1*sin(w*x) + w * c2*cos(w*x);
	};
	double dx = 0.01;
	double ans = NAN;
	for (int i = 0; i < 10000; i++) {
		double x0 = x_min + i * dx, x1 = x0 + dx, x = .5*(x0 + x1);
		double y0 = eval(x0), y1 = eval(x1);
		if (y0*y1 < 0) {
			// bisection search
			for (int iter = 0; iter < 64; iter++) {
				double y = eval(x);
				if (y*y1 > 0) x1 = x, y1 = y;
				else x0 = x, y0 = y;
				x = .5*(x0 + x1);
				if (x1 - x0 < 1e-12) break;
			}
		}
		else if (abs(y0) < m && abs(y1) < m) {
			// Newton's iteration
			for (int iter = 0; iter < 64; iter++) {
				double y = eval(x);
				double dy = evald(x);
				double dx = y / dy;
				if (abs(dx) < 1e6) x -= dx;
				else break;
				if (abs(dx) < 1e-12) break;
			}
		}
		else x = NAN;
		if (x >= x_min) {
			if (abs(eval(x) / evald(x)) < 1e-6) {
				if (!(x > ans)) ans = x;
			}
		}
	}
	return ans;
}
void test_solveTrigPoly() {
#ifdef _DEBUG
	freopen("stdout.txt", "w", stdout);
#endif
	for (int T = 0; T < 1000; T++) {
		//T = 761;
		_SRAND(T);
		double k[5];
		for (int i = 0; i < 5; i++) k[i] = 0;
		int N = randi(0, 4) + 1;
		for (int i = 0; i <= N; i++) k[i] = randf_n(5);
		vec2 p = rand2_n(5.0);
		double w = randf_n(2.0);
		for (unsigned i = 0; i < 10; i++) {
			//i = 2;
			_SRAND((T << 16) + i);
			double t0 = randf_n(10.);
#ifdef _DEBUG
			printf("s=Calc.getState();s['expressions']['list']=[");
#endif
			//double r = solveTrigPoly(k[4], k[3], k[2], k[1], k[0], p.x, p.y, w, t0);
			double r = solveTrigPoly_smallw(k[4], k[3], k[2], k[1], k[0], p.x, p.y, w, t0);
			r = refineRoot_TrigPoly(k[4], k[3], k[2], k[1], k[0], p.x, p.y, w, r);
			double r_ref = solveTrigPoly_slow(k[4], k[3], k[2], k[1], k[0], p.x, p.y, w, t0);
			r_ref = refineRoot_TrigPoly(k[4], k[3], k[2], k[1], k[0], p.x, p.y, w, r_ref);
#ifdef _DEBUG
			printf("];Calc.setState(s);\n");
#endif
			if (!((isnan(r) && isnan(r_ref)) || abs(r - r_ref) < 1e-12)) {
				fprintf(stderr, "%d %d  %lf  %lf %lf  %lg\n", T, i, t0, r, r_ref, r - r_ref);
			}
			//return;
		}
		//return;
	}
}
void test_solveTrigQuadratic() {
#ifdef _DEBUG
	freopen("stdout.txt", "w", stdout);
#endif
	for (int T = 0; T < 10000; T++) {
		//T = 2;
		_SRAND(T);
		double k[3];
		for (int i = 0; i <= 2; i++) k[i] = randf_n(5);
		vec2 p = rand2_n(5.0);
		//double w = randf_n(100.);
		//double w = randf_n(5.0);
		double w = randf_n(0.0001);
		for (unsigned i = 0; i < 10; i++) {
			//i = 9;
			_SRAND((T << 16) + i);
			double t0 = randf_n(10.);
#ifdef _DEBUG
			printf("s=Calc.getState();s['expressions']['list']=[");
#endif
			double r = solveTrigQuadratic(k[2], k[1], k[0], p.x, p.y, w, t0);
			r = refineRoot_TrigPoly(0, 0, k[2], k[1], k[0], p.x, p.y, w, r);
			double r_ref = solveTrigPoly_slow(0, 0, k[2], k[1], k[0], p.x, p.y, w, t0);
			r_ref = refineRoot_TrigPoly(0, 0, k[2], k[1], k[0], p.x, p.y, w, r_ref);
#ifdef _DEBUG
			printf("];Calc.setState(s);\n");
#endif
			if (!((isnan(r) && isnan(r_ref)) || abs(r - r_ref) < 1e-12)) {
				fprintf(stderr, "%d %d  %lf  %lf %lf  %lg\n", T, i, t0, r, r_ref, r - r_ref);
			}
			//return;
		}
		//return;
	}
}



int main() {
#ifdef _DEBUG
	freopen("stdout.txt", "w", stdout);
#endif
	test_solveQuadraticTrig();
	system("pause");
	return 0;
}

