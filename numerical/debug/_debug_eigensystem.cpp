// Use to debug "eigensystem.h"

// To-do: random matrix generator that generates non-invertible matrices



#include <stdio.h>

#include <chrono>
typedef std::chrono::high_resolution_clock NTime;
typedef std::chrono::duration<double> fsec;

#include "eigensystem.h"
#include "random.h"


#define N 6

int main() {
	auto t0 = NTime::now();
	for (int T = 0; T < 100000; T++) {
		_IDUM = T;
		double M[N][N];
		for (int i = 0; i < N; i++) for (int j = 0; j <= i; j++) {
			M[i][j] = M[j][i] = randf(-10.0, 10.0);
		}
		double eigv[N], eigvec[N][N];

		//EigenPairs_expand(N, &M[0][0], eigv, &eigvec[0][0]);  // faster when N isn't large
		EigenPairs_Jacobi(N, &M[0][0], eigv, &eigvec[0][0]);  // much more stable

		/*for (int i = 0; i < N; i++) {
			printf("%lf\t", eigv[i]);
			for (int j = 0; j < N; j++) printf("%lf ", eigvec[i][j]);
			printf("\n");
		}*/
		for (int i = 0; i < N; i++) {
			_debug_check_eigenpair_correctness(N, &M[0][0], eigv[i], eigvec[i]);
		}
	}
	printf("%lfsecs\n", fsec(NTime::now() - t0).count());
	return 0;
}

