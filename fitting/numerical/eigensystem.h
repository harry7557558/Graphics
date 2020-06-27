// organized from "Ellipse Fitting.cpp"
// assume all 6x6 matrixes are positive definite

// To-do: implement more eigenvalue algorithms


#include <cmath>

#ifndef __INC_LINEARSYSTEM_H
#include "linearsystem.h"
#endif


// find the all eigenpairs of a matrix by solving its characteristic equation
// doesn't seem to be practical
void EigenPairs_expand(const double M[6][6], double eigv[6], double eigvec[6][6]) {
	double A[6][6]; matcpy(M, A);
	double msc = 1.0;
#if 1
	// avoid overflow/precison error
	msc = 1. / pow(determinant(A), 1. / 6);
	if (0.0*msc == 0.0) {
		for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) A[i][j] *= msc;
		msc = 1. / msc;
	}
	else msc = 1.0;
#endif

	// expand characteristic polynomial, works in O(n^4)
	// method discribed at https://mathworld.wolfram.com/CharacteristicPolynomial.html
	double C[7], C0[7], Ci = -1;
	double B[6][6]; matcpy(A, B);
	C0[0] = C[0] = 1;
	for (int i = 1; i <= 6; i++) {
		Ci = trace(B) / i;
		double T[6][6]; matcpy(B, T);
		for (int k = 0; k < 6; k++) T[k][k] -= Ci;
		matmul(A, T, B);
		C0[i] = C[i] = -Ci;
	}
	//printPolynomial(C, 6);
	matcpy(A, B);

	// find the roots of the characteristic polynomial
	for (int R = 6; R > 0; R--) {
		// Newton's iteration method starting at x=0
		// this should success because the matrix is positive difinite
		double x = 0;
		for (int i = 0; i < 64; i++) {
			double y = 0, dy = 0;
			for (int i = 0; i <= R; i++) {
				y = y * x + C[i];
				if (R - i) dy = dy * x + (R - i)*C[i];
			}
			double dx = y / dy;
			x -= dx;
			if (dx*dx < 1e-24) break;
		}
#if 0
		// "refine" the root using the original polynomial
		for (int i = 0; i < 3; i++) {
			double y = 0, dy = 0;
			for (int i = 0; i <= 6; i++) {
				y = y * x + C0[i];
				if (6 - i) dy = dy * x + (6 - i)*C0[i];
			}
			double dx = y / dy;
			x -= dx;
			if (dx*dx < 1e-24) break;
		}
#endif
		// divide the root from the polynomial
		for (int i = 0; i < R; i++) {
			C[i + 1] += C[i] * x;
		}
		// export the eigenvalue
		eigv[6 - R] = x * msc;

		// find the eigenvector from the eigenvalue
		double v[6];
		matcpy(B, A);
		for (int i = 0; i < 6; i++) A[i][i] -= x;
		int sp = -1;
		const double eps = 1e-6;
		for (int i = 0, d = 0; d < 6; i++, d++) {
			if (abs(A[i][d]) < eps) {
				for (int j = i + 1; j < 6; j++) {
					if (!(abs(A[j][d]) < eps)) {
						for (int k = d; k < 6; k++) {
							double t = A[i][k]; A[i][k] = A[j][k]; A[j][k] = t;
						}
						break;
					}
				}
			}
			if (abs(A[i][d]) < eps) {
				sp = d;
				i--; continue;
			}
			else if (i == 5) {  // idk why this happens
				sp = 5;
				break;
			}
			double m = 1. / A[i][d];
			for (int k = d; k < 6; k++) A[i][k] *= m;
			for (int j = 0; j < 6; j++) if (j != i) {
				double m = A[j][d];
				for (int k = d; k < 6; k++) A[j][k] -= m * A[i][k];
			}
		}
		for (int i = 0; i < sp; i++) v[i] = -A[i][sp];
		v[sp] = 1;
		for (int i = sp + 1; i < 6; i++) v[i] = 0;

		// try to "refine" the eigenvector using some iterative methods

		// export normalized eigenvector
		double m = 1;
		for (int i = 0; i < sp; i++) m += v[i] * v[i];
		m = 1. / sqrt(m);
		for (int i = 0; i < 6; i++) eigvec[6 - R][i] = v[i] * m;
	}
}

// find an eigenpair using power iteration and inverse iteration
void EigenPair_powIter(const double M[6][6], double &u, double a[6]) {
	for (int i = 0; i < 6; i++) a[i] = sqrt(1. / 6);
	double A[6][6]; matcpy(M, A);
#if 1
	// use power of matrix to make it converge faster
	// sometimes the numbers can get extremly large and cause precision/overflow error
	double m = 1. / pow(determinant(A), 1. / 6);
	if (0.*m == 0.) {
		for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) A[i][j] *= m;
		for (int i = 0; i < 2; i++) {
			double T[6][6]; matmul(A, A, T);
			matcpy(T, A);
		}
	}
#endif
	// power iteration
	for (int i = 0; i < 1024; i++) {
		double v[6]; matmul(A, a, v);
		double m = 0;
		for (int j = 0; j < 6; j++) m += v[j] * v[j];
		m = 1. / sqrt(m);
		double err = 0;
		for (int j = 0; j < 6; j++) {
			v[j] *= m;
			err += v[j] * a[j];
			a[j] = v[j];
		}
		if (abs(abs(err) - 1) < 1e-12) break;
		// warning: floatpoint precision
	}
	// calculate eigenvalue from eigenvector
	u = 0;
	double v[6]; matmul(M, a, v);
	for (int i = 0; i < 6; i++) u += v[i] * v[i];
	u = sqrt(u);
}
void EigenPair_invIter(const double M[6][6], double &u, double a[6]) {
	double A[6][6]; matinv(M, A);
	EigenPair_powIter(A, u, a);
	u = 1. / u;
}


