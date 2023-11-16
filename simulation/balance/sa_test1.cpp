// simulated annealing test
// minimize a periodic function approximately between -2 and 2

#include <cmath>
#include <stdio.h>
#include <stdint.h>

#include "numerical/random.h"  // erfinv

// function to minimize
int evalCount = 0;
double Fun(double x) {
	evalCount++;
	double s = 0;
	//s += .01*abs((x - 1.)*(x - 5.)*(x + 15.)) + .01*x*x;
	//s += 100.*abs(max(abs(x) - PI, 0.));
	for (int n = 1; n <= 20; n++) {
		//s += sin(exp2(n)*x)*exp2(-n);
		s += sin(n*n*x) / (n*n);
		//s += sin(n*n*x) / n;
		//s += asin(sin(n*n*x)) / (n*n);
	}
	//s = sin(x) + sin(10.*x) + sin(100.*x);
	return s;
}

// simulated annealing process
double Simulated_Annealing() {
	uint32_t seed1 = 0, seed2 = 1, seed3 = 2;  // random number seeds
	double rand;
	double x = 0.;  // configulation
	double T = 100.0;  // temperature
	double E = Fun(x);  // energy
	double min_x = x, min_E = E;  // record minimum value encountered
	const int max_iter = 150;  // number of iterations
	const int max_try = 10;  // maximum number of samples per iteration
	double T_decrease = 0.9;  // multiply temperature by this each time
	for (int iter = 0; iter < max_iter; iter++) {
		for (int ty = 0; ty < max_try; ty++) {
			rand = int32_t(seed1 = seed1 * 1664525u + 1013904223u) / 2147483648.;  // -1<=rand<1
			double dx = T * erfinv(rand);  // change of configuration
			double x_new = x + dx;
			double E_new = Fun(x_new);
			double prob = exp(-(E_new - E) / T);  // probability, note that E is approximately between -2 and 2
			//printf("%lf\n", prob);
			rand = (seed2 = seed2 * 1664525u + 1013904223u) / 4294967296.;  // 0<=rand<1
			if (prob > rand) {  // jump
				x = x_new, E = E_new;
				if (E < min_E) {
					min_x = x, min_E = E;
				}
				break;
			}
		}
		// jump to the minimum point encountered
		double prob = tanh(E - min_E);
		rand = (seed3 = seed3 * 1664525u + 1013904223u) / 4294967296.;
		if (prob > rand) {
			x = min_x, E = min_E;
		}
		// decrease temperature
		T *= T_decrease;
		printf("%d T=%lf x=%lf E=%lf\n", iter, T, x, E);
	}
	return x;
}

int main() {
	evalCount = 0;
	double x = Simulated_Annealing();
	printf("%d evals\n", evalCount);
	x = fmod(x, 2.*PI);
	printf("(%lf,%lf)\n", x, Fun(x));
}

