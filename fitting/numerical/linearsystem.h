// organized from "Ellipse Fitting.cpp"
// currently all matrixes are 6x6 unpacked


#ifndef __INC_LINEARSYSTEM_H

#define __INC_LINEARSYSTEM_H



// 6x6 matrixes

// copy a matrix
void matcpy(const double src[6][6], double res[6][6]) {
	for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) res[i][j] = src[i][j];
}

// matrix multiplication, C needs to be different from A B
void matmul(const double A[6][6], const double B[6][6], double C[6][6]) {
	for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) {
		C[i][j] = 0;
		for (int k = 0; k < 6; k++) C[i][j] += A[i][k] * B[k][j];
	}
}

// matrix times vector, b needs to be different from x
void matmul(const double A[6][6], const double x[6], double b[6]) {
	for (int i = 0; i < 6; i++) {
		b[i] = 0;
		for (int j = 0; j < 6; j++) b[i] += A[i][j] * x[j];
	}
}

// evaluate uᵀAv
double quamul(const double u[6], const double A[6][6], const double v[6]) {
	double s = 0;
	for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) {
		s += u[i] * A[i][j] * v[j];
	}
	return s;
}

// trace of a matrix
double trace(const double M[6][6]) {
	double res = 0;
	for (int i = 0; i < 6; i++) res += M[i][i];
	return res;
}

// determinant of a matrix
double determinant(const double M[6][6]) {
	double A[6][6]; matcpy(M, A);
	double det = 1;
	for (int i = 0; i < 6; i++) {
		for (int j = i + 1; j < 6; j++) {
			double m = -A[j][i] / A[i][i];
			for (int k = i; k < 6; k++) A[j][k] += m * A[i][k];
		}
		det *= A[i][i];
	}
	if (0.0*det != 0.0) return 0.0;
	return det;
}

// matrix inversion
void matinv(const double M[6][6], double I[6][6]) {
	double A[6][6]; matcpy(M, A);
	for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) I[i][j] = double(i == j);
	for (int i = 0; i < 6; i++) {
		double m = 1.0 / A[i][i];
		for (int k = 0; k < 6; k++) {
			A[i][k] *= m, I[i][k] *= m;
		}
		for (int j = 0; j < 6; j++) if (j != i) {
			m = -A[j][i];
			for (int k = 0; k < 6; k++) {
				A[j][k] += m * A[i][k];
				I[j][k] += m * I[i][k];
			}
		}
	}
}

// matrix transpose
void transpose(const double src[6][6], double res[6][6]) {
	for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) {
		res[i][j] = src[j][i];
	}
}
void transpose(double A[6][6]) {
	for (int i = 0; i < 6; i++) for (int j = 0; j < i; j++) {
		double t = A[i][j]; A[i][j] = A[j][i]; A[j][i] = t;
	}
}




#endif // __INC_LINEARSYSTEM_H

