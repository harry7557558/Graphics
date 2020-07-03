// Use to debug "eigensystem.h"

// To-do: random matrix generator that generates non-invertible matrixes



#include <stdio.h>

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

#include "eigensystem.h"
#include "random.h"


// a debug function that checks the correctness of eigenpair calculation
// eigvec should be normalized
void _debug_check_eigenpair_correctness(const double M[6][6], double eigv, double eigvec[6]) {
	double u[6]; matmul(M, eigvec, u);
	double v[6]; for (int i = 0; i < 6; i++) v[i] = eigv * eigvec[i];
	double e = 0; for (int i = 0; i < 6; i++) e += u[i] * u[i]; e = sqrt(e) * (eigv > 0. ? 1. : -1);
	if (abs(e / eigv - 1.) > 1e-10) {
		printf("Error! %d\n", __LINE__);
	}
	e = 0; for (int i = 0; i < 6; i++) e += u[i] * v[i]; e /= eigv * eigv;
	if (abs(e - 1.) > 1e-10) {
		printf("Error! %d\n", __LINE__);
	}
}


int main() {
	auto t0 = NTime::now();
	for (int T = 0; T < 100000; T++) {
		_IDUM = T;
		double M[6][6];
		for (int i = 0; i < 6; i++) for (int j = 0; j <= i; j++) {
			M[i][j] = M[j][i] = randf_n(10.0);
		}
		double eigv[6], eigvec[6][6];

		//EigenPairs_expand(M, eigv, eigvec);  // 2x faster ??!!
		EigenPairs_Jacobi(M, eigv, eigvec);  // more stable

		/*for (int i = 0; i < 6; i++) {
			printf("%lf\t", eigv[i]);
			for (int j = 0; j < 6; j++) printf("%lf ", eigvec[i][j]);
			printf("\n");
		}*/
		for (int i = 0; i < 6; i++) {
			_debug_check_eigenpair_correctness(M, eigv[i], eigvec[i]);
		}
	}
	printf("%lfsecs\n", fsec(NTime::now() - t0).count());
	return 0;
}

