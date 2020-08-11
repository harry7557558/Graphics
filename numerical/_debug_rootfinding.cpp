
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



int main() {
	auto t0 = NTime::now();
	test_solveQuartic();
	printf("%lfsecs\n", fsec(NTime::now() - t0).count());
	return 0;
}

