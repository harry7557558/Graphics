// organized from "Ellipse Fitting.cpp"
// assume all matrices are positive definite

// To-do: implement more eigenvalue algorithms



#ifndef __INC_EIGENSYSTEM_H

#define __INC_EIGENSYSTEM_H


#include <cmath>

#ifndef __INC_LINEARSYSTEM_H
#include "linearsystem.h"
#endif

#ifndef PI
#define PI 3.1415926535897932384626
#endif



// checks the correctness of eigenpair calculation
// eigvec should be normalized
void _debug_check_eigenpair_correctness(int N, const double *M, double eigv, double *eigvec) {
	double *u = new double[N]; matvecmul(N, M, eigvec, u);
	double *v = new double[N]; for (int i = 0; i < N; i++) v[i] = eigv * eigvec[i];
	double e = 0; for (int i = 0; i < N; i++) e += u[i] * u[i]; e = sqrt(e) * (eigv > 0. ? 1. : -1);
	if (abs(e / eigv - 1.) > 1e-10) {
		printf("Error! [eigensystem.h %d] %lg\n", __LINE__, abs(e / eigv - 1.));
	}
	e = 0; for (int i = 0; i < N; i++) e += u[i] * v[i]; e /= eigv * eigv;
	if (abs(e - 1.) > 1e-10) {
		printf("Error! [eigensystem.h %d] %lg\n", __LINE__, abs(e - 1.));
	}
	delete u; delete v;
}
bool _check_eigenpair_correctness(int N, const double *M, double eigv, double *eigvec) {
	double *u = new double[N]; matvecmul(N, M, eigvec, u);
	double *v = new double[N]; for (int i = 0; i < N; i++) v[i] = eigv * eigvec[i];
	double e = 0; for (int i = 0; i < N; i++) e += u[i] * u[i]; e = sqrt(e) * (eigv > 0. ? 1. : -1);
	if (abs(e / eigv - 1.) > 1e-6) return false;
	e = 0; for (int i = 0; i < N; i++) e += u[i] * v[i]; e /= eigv * eigv;
	if (abs(e - 1.) > 1e-6) return false;
	delete u; delete v;
	return true;
}






// find the all eigenpairs of a matrix by solving its characteristic equation
// eigvec: a matrix of row vectors
// due to the O(N⁴) complexity and error accumulation in Gaussian elimination, it is not recommend for N>6
void EigenPairs_expand(int N, const double *M, double *eigv, double *eigvec) {
	double *A = new double[N*N]; matcpy(N, M, A);
	double msc = 1.0;
#if 0
	// avoid overflow/precision error
	msc = 1. / pow(determinant(N, A), 1. / N);
	if (0.0*msc == 0.0) {
		for (int i = 0; i < N*N; i++) A[i] *= msc;
		msc = 1. / msc;
	}
	else msc = 1.0;
#endif

	// expand characteristic polynomial, works in O(n^4)
	// method discribed at https://mathworld.wolfram.com/CharacteristicPolynomial.html
	double *C = new double[N + 1], *C0 = new double[N + 1], Ci = -1;
	double *B = new double[N*N], *T = new double[N*N]; matcpy(N, A, B);
	C0[0] = C[0] = 1;
	for (int i = 1; i <= N; i++) {
		Ci = trace(N, B) / i;
		matcpy(N, B, T);
		for (int k = 0; k < N; k++) T[k*N + k] -= Ci;
		matmul(N, A, T, B);
		C0[i] = C[i] = -Ci;
	}
	//printPolynomial(C, N);
	matcpy(N, A, B);

	// find the roots of the characteristic polynomial
	double *v = new double[N];
	for (int R = N; R > 0; R--) {
		// Newton's iteration method starting at x=0
		// this should success because the matrix is positive definite
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
#if 1
		// "refine" the root using the original polynomial
		for (int i = 0; i < 3; i++) {
			double y = 0, dy = 0;
			for (int i = 0; i <= N; i++) {
				y = y * x + C0[i];
				if (N - i) dy = dy * x + (N - i)*C0[i];
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
		eigv[N - R] = x * msc;

		// find the eigenvector from the eigenvalue
		matcpy(N, B, A);
		for (int i = 0; i < N; i++) A[i*N + i] -= x;
		int sp = -1;
		const double eps = 1e-6;
		for (int i = 0, d = 0; d < N; i++, d++) {
			if (abs(A[i*N + d]) < eps) {
				for (int j = i + 1; j < N; j++) {
					if (!(abs(A[j*N + d]) < eps)) {
						for (int k = d; k < N; k++) {
							double t = A[i*N + k]; A[i*N + k] = A[j*N + k]; A[j*N + k] = t;
						}
						break;
					}
				}
			}
			if (abs(A[i*N + d]) < eps) {
				sp = d;
				i--; continue;
			}
			else if (i == N - 1) {  // idk why this happens
				sp = N - 1;
				break;
			}
			double m = 1. / A[i*N + d];
			for (int k = d; k < N; k++) A[i*N + k] *= m;
			for (int j = 0; j < N; j++) if (j != i) {
				double m = A[j*N + d];
				for (int k = d; k < N; k++) A[j*N + k] -= m * A[i*N + k];
			}
		}
		for (int i = 0; i < sp; i++) v[i] = -A[i*N + sp];
		v[sp] = 1;
		for (int i = sp + 1; i < N; i++) v[i] = 0;

		// try to "refine" the eigenvector using some iterative methods

		// export normalized eigenvector
		double m = 1;
		for (int i = 0; i < sp; i++) m += v[i] * v[i];
		m = 1. / sqrt(m);
		for (int i = 0; i < N; i++) eigvec[(N - R)*N + i] = v[i] * m;
	}
	delete v;

	delete T; delete B; delete C; delete C0;
	delete A;
}




// find an eigenpair using power iteration and inverse iteration
void EigenPair_powIter(int N, const double *M, double *eigv, double *eigvec) {
	for (int i = 0; i < N; i++) eigvec[i] = sqrt(1. / N);
	double *A = new double[N*N]; matcpy(N, M, A);
#if 1
	// use power of matrix to make it converge faster
	// sometimes the numbers can get extremely large and cause precision/overflow error
	double m = 1. / pow(determinant(N, A), 1. / N);
	if (0.*m == 0.) {
		for (int i = 0; i < N*N; i++) A[i] *= m;
		double *T = new double[N*N];
		for (int i = 0; i < 2; i++) {
			matmul(N, A, A, T);
			matcpy(N, T, A);
		}
		delete T;
	}
#endif
	// power iteration
	double *v = new double[N];
	for (int i = 0; i < 1024; i++) {
		matvecmul(N, A, eigvec, v);
		double m = 0;
		for (int j = 0; j < N; j++) m += v[j] * v[j];
		m = 1. / sqrt(m);
		double err = 0;
		for (int j = 0; j < N; j++) {
			v[j] *= m;
			err += v[j] * eigvec[j];
			eigvec[j] = v[j];
		}
		if (abs(abs(err) - 1) < 1e-12) break;
		// warning: float-point precision
	}
	// calculate eigenvalue from eigenvector
	matvecmul(N, M, eigvec, v);
	for (int i = 0; i < N; i++) {
		if (abs(eigvec[i]) > .2) {
			if (eigv) *eigv = v[i] / eigvec[i];
			break;
		}
	}
	delete v;
	delete A;
}
void EigenPair_invIter(int N, const double *M, double *eigv, double *eigvec) {
	double *A = new double[N*N]; matinv(N, M, A);
	EigenPair_powIter(N, A, eigv, eigvec);
	if (eigv) *eigv = 1. / *eigv;
	delete A;
}




// zero off-diagonal elements of a symmetric matrix using given rotation matrices
// eigvec is an orthogonal matrix of row eigenvectors
// keep result eigenvalues and eigenvectors as diagonalized form (unsorted)
void EigenPairs_Jacobi(int N, const double *M, double *eigv, double *eigvec) {
	double *A = new double[N*N]; matcpy(N, M, A);
	double *C = new double[N*N]; for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) C[i*N + j] = i == j;
	double *tj = new double[N], *ti = new double[N];
	double err = 0.0;
	for (int d = 0; d < 64; d++) {
		err = 0.;
		for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) {
			err += A[i*N + j] * A[i*N + j];
			// calculate the rotation matrix
			double a = A[j*N + j], b = A[i*N + i], d = A[i*N + j];
			auto atan2 = [](double y, double x) { return atan(y / x) + (x > 0 ? 0. : y < 0 ? -PI : PI); };  // idk why this makes it 1.4x faster on my machine
			double t = .5*atan2(2.*d, a - b);  // WARNING: atan2(0,0)
			double c = cos(t), s = sin(t);
			// apply inverse rotation to the left side of A
			for (int k = 0; k < N; k++) {
				tj[k] = c * A[j*N + k] + s * A[i*N + k];
				ti[k] = c * A[i*N + k] - s * A[j*N + k];
			}
			for (int k = 0; k < N; k++) A[j*N + k] = tj[k], A[i*N + k] = ti[k];
			// apply rotation to the right side of A
			for (int k = 0; k < N; k++) {
				tj[k] = c * A[k*N + j] + s * A[k*N + i];
				ti[k] = c * A[k*N + i] - s * A[k*N + j];
			}
			for (int k = 0; k < N; k++) A[k*N + j] = tj[k], A[k*N + i] = ti[k];
			// apply rotation to the right side of C
			for (int k = 0; k < N; k++) {
				tj[k] = c * C[k*N + j] + s * C[k*N + i];
				ti[k] = c * C[k*N + i] - s * C[k*N + j];
			}
			for (int k = 0; k < N; k++) C[k*N + j] = tj[k], C[k*N + i] = ti[k];
		}
		//printf("%lf\n", .5*log10(err));
		if (err < 1e-32) break;
	}
	for (int i = 0; i < N; i++) eigv[i] = A[i*N + i];
	transpose(N, C, eigvec);
	delete A; delete C; delete tj; delete ti;
}




#endif  // __INC_EIGENSYSTEM_H

