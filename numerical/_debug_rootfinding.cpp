
#include "random.h"
#include <stdio.h>

#include "rootfinding.h"



#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;


// 1.33s on my machine
void test_solveCubic() {
	for (int i = 0; i < 10000000; i++) {
		_SRAND(i);
		//double c2 = randf_n(10.0), c1 = randf_n(10.0), c0 = randf_n(10.0);
		double c2 = randf(-20, 20), c1 = randf(-20, 20), c0 = randf(-20, 20);
		auto testroot = [&](double r) {
			double y = c0 + r * (c1 + r * (c2 + r));
			double dy = c1 + r * (2.*c2 + r * 3.);
			if (!(abs(y / dy) < 1e-12)) {
				printf("%lg\n", y / dy);
			}
		};
		double r, u, v;
		bool n = solveCubic(1, c2, c1, c0, r, u, v);
		r = refineRoot_cubic(1, c2, c1, c0, r); testroot(r);
		if (n) {
			u = refineRoot_cubic(1, c2, c1, c0, u); testroot(u);
			v = refineRoot_cubic(1, c2, c1, c0, v); testroot(v);
		}
	}
}

// 1.48s on my machine
void test_solveQuartic() {
	for (int i = 0; i < 10000000; i++) {
		_SRAND(i);
		//double c3 = randf_n(10.0), c2 = randf_n(10.0), c1 = randf_n(10.0), c0 = randf_n(10.0);
		double c3 = randf(-20, 20), c2 = randf(-20, 20), c1 = randf(-20, 20), c0 = randf(-20, 20);
		auto testroot = [&](double r) {
			double y = c0 + r * (c1 + r * (c2 + r * (c3 + r)));
			double dy = c1 + r * (2.*c2 + r * (3.*c3 + r * 4.));
			if (!(abs(y / dy) < 1e-10)) {
				printf("%lg\n", y / dy);
			}
		};
		double r[4];
		int n = solveQuartic(1, c3, c2, c1, c0, r);
		for (int i = 0; i < n; i++) {
			r[i] = refineRoot_quartic(1, c3, c2, c1, c0, r[i]);
			testroot(r[i]);
		}
	}
}




double solveTrigPoly_slow(double k4, double k3, double k2, double k1, double k0, double c1, double c2, double w, double x_min) {
	auto eval = [&](double x) {
		return k0 + x * (k1 + x * (k2 + x * (k3 + x * k4))) + c1 * cos(w*x) + c2 * sin(w*x);
	};
	auto evald = [&](double x) {
		return k1 + x * (2 * k2 + x * (3 * k3 + x * 4 * k4)) - w * c1*sin(w*x) + w * c2*cos(w*x);
	};
	double dx = 0.01;
	double ans = NAN;
	for (int i = 0; i < 2000; i++) {
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
		else {
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


int main() {
	auto t0 = NTime::now();
	test_solveTrigPoly();
	fprintf(stderr, "%lfsecs\n", fsec(NTime::now() - t0).count());
	return 0;
}

