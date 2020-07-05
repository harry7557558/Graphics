// Use to debug "eigensystem.h"

// To-do: random matrix generator that generates non-invertible matrixes



#include <stdio.h>

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

#include "eigensystem.h"
#include "random.h"



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

