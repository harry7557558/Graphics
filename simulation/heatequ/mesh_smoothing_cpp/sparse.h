#pragma once

#include <vector>
#include <map>


class LilMatrix {
	int n;  // n*n square matrix
	std::vector<std::map<int, float>> mat;  // list of dict?!
public:
	friend class CsrMatrix;

	LilMatrix(int n) {
		this->n = n;
		mat.resize(n);
	}

	void addValue(int row, int col, float val) {
		mat[row][col] += val;
	}

};

class CsrMatrix {
	int n;  // n*n square matrix
	std::vector<int> rows;  // see Wikipedia
	std::vector<int> cols;
	std::vector<float> vals;

	// square of norm of a vector
	float vecnorm2(const float *r) const {
		float ans = 0.0f;
		for (int i = 0; i < n; i++) ans += r[i]*r[i];
		return ans;
	}

	// dot product between two vectors
	float vecdot(const float *u, const float *v) const {
		float ans = 0.0f;
		for (int i = 0; i < n; i++) ans += u[i]*v[i];
		return ans;
	}

public:

	CsrMatrix() : n(0) {}
	CsrMatrix(const LilMatrix &lil) {
		this->n = lil.n;
		rows.push_back(0);
		for (int i = 0; i < n; i++) {
			for (std::pair<int, float> indice : lil.mat[i]) {
				if (indice.second != 0.0f) {
					cols.push_back(indice.first);
					vals.push_back(indice.second);
				}
			}
			rows.push_back((int)cols.size());
		}
	}

	// multiply by scalar
	void operator*=(float k) {
		for (int i = 0, l = rows.back(); i < l; i++)
			vals[i] *= k;
	}

	// res = mat * src
	void matvecmul(const float *src, float *res) const {
		for (int i = 0; i < n; i++) {
			res[i] = 0.0f;
			for (int ji = rows[i]; ji < rows[i+1]; ji++) {
				res[i] += vals[ji] * src[cols[ji]];
			}
		}
	}

	// return u^T * mat * v
	float vecmatvecmul(const float *u, const float *v) const {
		float res = 0.0f;
		for (int i = 0; i < n; i++) {
			for (int ji = rows[i]; ji < rows[i+1]; ji++) {
				res += u[i] * vals[ji] * v[cols[ji]];
			}
		}
		return res;
	}

	// solve mat*x=b using conjugate gradient, returns the number of iterations
	// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
	int cg(const float *b, float *x, int maxiter, float atol) const {
		float tol = float(n)*atol*atol;
		// r = b - Ax
		float *r = new float[n];
		this->matvecmul(x, r);
		for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
		float r20 = vecnorm2(r);
		//if (r20 < tol) { delete r; return 0; }
		// p = r
		float *p = new float[n];
		std::memcpy(p, r, n*sizeof(float));
		// repeat
		float *Ap = new float[n];
		int k; for (k = 0; k < maxiter; k++) {
			//printf("%f,", sqrt(r20/n));
			// α = rᵀr / pᵀAp
			this->matvecmul(p, Ap);
			float alpha = r20 / vecdot(p, Ap);
			// x = x + αp
			for (int i = 0; i < n; i++) x[i] += alpha * p[i];
			// r = r - αAp
			for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];
			float r21 = vecnorm2(r);
			if (r21 < tol) { k++; break; }
			// β = r₁ᵀr₁ / r₀ᵀr₀
			float beta = r21 / r20;
			r20 = r21;
			// p = r + βp
			for (int i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
		}
		delete r; delete p; delete Ap;
		return k;
	}

};
