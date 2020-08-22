// seems like hard-coding matrices as static 2d arrays can be 1.2x faster
// do dirty things when performance really matter


#ifndef __INC_LINEARSYSTEM_H

#define __INC_LINEARSYSTEM_H


#include <stdlib.h>
#include <cmath>


// copy a matrix
void veccpy(int N, const double *src, double *res) {
	for (int i = 0; i < N; i++) res[i] = src[i];
}
void matcpy(int N, const double *src, double *res) {
	for (int i = 0, l = N * N; i < l; i++) res[i] = src[i];
}

// matrix multiplication, C needs to be different from A B
void matmul(int N, const double *A, const double *B, double *C) {
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
		double t = 0;
		for (int k = 0; k < N; k++) t += A[i*N + k] * B[k*N + j];
		C[i*N + j] = t;
	}
}

// matrix times vector, b needs to be different from x
void matvecmul(int N, const double *A, const double *x, double *b) {
	for (int i = 0; i < N; i++) {
		b[i] = 0;
		auto Ai = &A[i*N];
		for (int j = 0; j < N; j++) b[i] += Ai[j] * x[j];
	}
}

// evaluate uᵀAv
double quamul(int N, const double *u, const double *A, const double *v) {
	double s = 0;
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
		s += u[i] * A[i*N + j] * v[j];
	}
	return s;
}

// evaluate uᵀv
double vecdot(int N, const double *u, const double *v) {
	double r = 0;
	for (int i = 0; i < N; i++) r += u[i] * v[i];
	return r;
}

// trace of a matrix
double trace(int N, const double *M) {
	double res = 0;
	for (int i = 0; i < N; i++) res += M[i*N + i];
	return res;
}

// determinant of a matrix
double determinant(int N, const double *M) {
	double *A = new double[N*N]; matcpy(N, M, A);
	double det = 1;
	for (int i = 0; i < N; i++) {
		for (int j = i + 1; j < N; j++) {
			double m = -A[j*N + i] / A[i*N + i];
			for (int k = i; k < N; k++) A[j*N + k] += m * A[i*N + k];
		}
		det *= A[i*N + i];
	}
	delete A;
	if (0.0*det != 0.0) return 0.0;
	return det;
}

// matrix inversion
void matinv(int N, const double *M, double *I) {
	double *A = new double[N*N]; matcpy(N, M, A);
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) I[i*N + j] = double(i == j);
	for (int i = 0; i < N; i++) {
		double m = 1.0 / A[i*N + i];
		for (int k = 0; k < N; k++) {
			A[i*N + k] *= m, I[i*N + k] *= m;
		}
		for (int j = 0; j < N; j++) if (j != i) {
			m = -A[j*N + i];
			for (int k = 0; k < N; k++) {
				A[j*N + k] += m * A[i*N + k];
				I[j*N + k] += m * I[i*N + k];
			}
		}
	}
	delete A;
}

// matrix transpose
void transpose(int N, const double *src, double *res) {
	for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
		res[i*N + j] = src[j*N + i];
	}
}
void transpose(int N, double *A) {
	for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
		double t = A[i*N + j]; A[i*N + j] = A[j*N + i]; A[j*N + i] = t;
	}
}


// solve linear system, assume the matrix is invertible and not too large
void solveLinear_nc(int N, double *A, double *x) {
	for (int i = 0; i < N; i++) {
		auto Ai = &A[i*N];
		double m = 1.0 / Ai[i];
		for (int k = i; k < N; k++) Ai[k] *= m;
		x[i] *= m;
		for (int j = 0; j < N; j++) if (j != i) {
			auto Aj = &A[j*N];
			double m = -Aj[i] / Ai[i];
			for (int k = i; k < N; k++) Aj[k] += m * Ai[k];
			x[j] += m * x[i];
		}
	}
}
void solveLinear(int N, const double *M, double *x) {
	double *A = new double[N*N]; for (int i = 0; i < N*N; i++) A[i] = M[i];
	solveLinear_nc(N, A, x);
	delete A;
}

// symmetric matrix only, initialize X to an initial guess
// seems to be O(N^3) for non-sparse matrices; larger constant than elimination and higher error
void solveLinear_ConjugateGradient(int N, const double* M, const double *b, double *x) {
	double *A = new double[N*N]; for (int i = 0; i < N*N; i++) A[i] = M[i];
	double *r = new double[N], *p = new double[N], *Ap = new double[N];

	matvecmul(N, A, x, Ap);
	for (int i = 0; i < N; i++) p[i] = r[i] = b[i] - Ap[i];
	double rsold = vecdot(N, r, r), rsnew, alpha;

	int MAX_ITER = N + (.005*N*N);  // :(
	int i; for (i = 0; i < MAX_ITER; i++) {
		matvecmul(N, A, p, Ap);  // O(N^2)
		alpha = rsold / vecdot(N, p, Ap);
		for (int i = 0; i < N; i++) x[i] += p[i] * alpha;
		for (int i = 0; i < N; i++) r[i] -= Ap[i] * alpha;
		rsnew = vecdot(N, r, r);
		if (rsnew < 1e-12) break;
		double t = rsnew / rsold;
		for (int i = 0; i < N; i++) p[i] = r[i] + p[i] * t;
		rsold = rsnew;
	}
	//if (i > maxIter) maxIter = i;
	//printf("%d ", i);

	delete A;
	delete r; delete p; delete Ap;
}



// debug
#include <stdio.h>
void printMatrix(int N, const double *A, const char* end = "\n") {
	putchar('{');
	for (int i = 0; i < N; i++) {
		putchar('{');
		for (int j = 0; j < N; j++) printf("%lf%c", A[i*N + j], j == N - 1 ? '}' : ',');
		putchar(i == N - 1 ? '}' : ',');
	}
	printf(end);
}
void printMatrix_latex(int M, int N, const double *A, const char end[] = "\\\\\n") {  // M rows and N columns
	printf("\\begin{bmatrix}");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (abs(A[i*N + j]) < 1e-6) printf("0");
			else printf("%.6g", A[i*N + j]);
			if (j < N - 1) putchar('&');
		}
		printf("\\\\");
	}
	printf("\\end{bmatrix}%s", end);
}
void printMatrix_latex(int N, const double *M, const char end[] = "\\\\\n") {
	printMatrix_latex(N, N, M, end);
}




#endif // __INC_LINEARSYSTEM_H

