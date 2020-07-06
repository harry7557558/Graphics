// organized from "Ellipse Fitting.cpp"
// currently all matrixes are 6x6 unpacked


#ifndef __INC_LINEARSYSTEM_H

#define __INC_LINEARSYSTEM_H




// copy a matrix
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
		for (int j = 0; j < N; j++) b[i] += A[i*N + j] * x[j];
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

// solve linear system
void solveLinear(int N, const double *M, double *x) {
	double *A = new double[N*N]; for (int i = 0; i < N*N; i++) A[i] = M[i];
	for (int i = 0; i < N; i++) {
		double m = 1.0 / A[i*N + i];
		for (int k = i; k < N; k++) A[i*N + k] *= m;
		x[i] *= m;
		for (int j = 0; j < N; j++) if (j != i) {
			double m = -A[j*N + i] / A[i*N + i];
			for (int k = i; k < N; k++) A[j*N + k] += m * A[i*N + k];
			x[j] += m * x[i];
		}
	}
	delete A;
}





// debug
#include <stdio.h>
void printMatrix(const double A[6][6], const char* end = "\n") {
	putchar('{');
	for (int i = 0; i < 6; i++) {
		putchar('{');
		for (int j = 0; j < 6; j++) printf("%lf%c", A[i][j], j == 5 ? '}' : ',');
		putchar(i == 5 ? '}' : ',');
	}
	printf(end);
}
void printMatrix_latex(const double M[6][6], const char end[] = "\\\\\n") {
	printf("\\begin{bmatrix}");
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			if (abs(M[i][j]) < 1e-6) printf("0");
			else printf("%.6g", M[i][j]);
			if (j < 5) putchar('&');
		}
		printf("\\\\");
	}
	printf("\\end{bmatrix}%s", end);
}




#endif // __INC_LINEARSYSTEM_H

