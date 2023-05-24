#pragma once

#include <vector>
#include <map>


class LilMatrix {
	int n;  // n*n square matrix
	std::vector<std::map<int, float>> mat;  // list of dict?!

public:
	friend class CsrMatrix;
	friend class State;

	LilMatrix(int n) {
		this->n = n;
		mat.resize(n);
	}
	int getN() const { return n; }
	void addValue(int row, int col, float val) {
		mat[row][col] += val;
	}

	// return true iff the matrix is strictly symmetric
	bool isSymmetric() const {
		for (int i = 0; i < n; i++) {
			for (std::pair<int, float> jw : mat[i]) {
				int j = jw.first;
				if (j >= i) break;
				auto p = mat[j].find(i);
				if (p == mat[j].end() || p->second != jw.second)
					return false;
			}
		}
		return true;
	}

	// res = mat * src
	template<typename vec>
	void matvecmul(const vec *src, vec *res) const {
		for (int i = 0; i < n; i++) {
			res[i] = vec(0.0f);
			for (std::pair<int, float> jw : mat[i]) {
				res[i] += jw.second * src[jw.first];
			}
		}
	}

	// transpose
	LilMatrix transpose() const {
		LilMatrix res(n);
		for (int i = 0; i < n; i++) {
			for (std::pair<int, float> jw : mat[i]) {
				res.addValue(jw.first, i, jw.second);
			}
		}
		return res;
	}

};


class CsrMatrix {
	int n;  // n*n square matrix
	std::vector<int> rows;  // see Wikipedia
	std::vector<int> cols;
	std::vector<float> vals;

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
	int getN() const { return n; }

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

	// res = mat^T * src
	void vecmatmul(const float *src, float *res) const {
		for (int j = 0; j < n; j++)
			res[j] = 0.0f;
		for (int i = 0; i < n; i++) {
			for (int ji = rows[i]; ji < rows[i+1]; ji++) {
				res[cols[ji]] += vals[ji] * src[i];
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

	// evaluate (mat*x-b)^2
	float linequError2(const float *x, const float *b) const {
		float toterr = 0.0f;
		for (int i = 0; i < n; i++) {
			float err = -b[i];
			for (int ji = rows[i]; ji < rows[i+1]; ji++) {
				err += vals[ji] * x[cols[ji]];
			}
			toterr += err * err;
		}
		return toterr;
	}

	// iterative linear system solvers
	int CsrMatrix::cg(const float *b, float *x, int maxiter, float tol) const;
	int CsrMatrix::bicg(const float *b, float *x, int maxiter, float tol) const;
	int CsrMatrix::bicgstab(const float *b, float *x, int maxiter, float tol) const;
};

// Solve mat*x=b using conjugate gradient, returns the number of iterations.
// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
// Requires the matrix to be symmetric to converge.
int CsrMatrix::cg(const float *b, float *x, int maxiter, float tol) const {
	tol = float(n)*tol*tol;
	// r = b - Ax
	float *r = new float[n];
	this->matvecmul(x, r);
	for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
	float r20 = vecnorm2(r);
	// p = r
	float *p = new float[n];
	std::memcpy(p, r, n*sizeof(float));
	// loop
	float *Ap = new float[n];
	int k; for (k = 0; k < maxiter; k++) {
#ifdef _DEBUG
		printf("%f,", sqrt(linequError2(x, b)/n));
#endif
		// α = rᵀr / pᵀAp
		this->matvecmul(p, Ap);
		float alpha = r20 / vecdot(p, Ap);
		// x = x + αp
		for (int i = 0; i < n; i++) x[i] += alpha * p[i];
		// r = r - αAp
		for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];
		// β = r₁ᵀr₁ / r₀ᵀr₀
		float r21 = vecnorm2(r);
		if (r21 < tol) { k++; break; }
		float beta = r21 / r20;
		r20 = r21;
		// p = r + βp
		for (int i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
	}
	delete r; delete p; delete Ap;
	return k;
}

// Solve mat*x=b using biconjugate gradient, returns the number of iterations.
// https://en.wikipedia.org/wiki/Biconjugate_gradient_method#Unpreconditioned_version_of_the_algorithm
// Convergence may be unstable. Identical to `cg` when the matrix is symmetric.
int CsrMatrix::bicg(const float *b, float *x, int maxiter, float tol) const {
	tol = float(n)*tol*tol;
	// x*
	float *xt = new float[n];
	// r = b - A x
	float *r = new float[n];
	this->matvecmul(x, r);
	for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
	// r* = b - x* A
	float *rt = new float[n];
	this->vecmatmul(x, rt);
	for (int i = 0; i < n; i++) rt[i] = b[i] - rt[i];
	float r20 = vecdot(r, rt);
	// p = r, p* = r*
	float *p = new float[n];
	std::memcpy(p, r, sizeof(float)*n);
	float *pt = new float[n];
	std::memcpy(pt, rt, sizeof(float)*n);
	// loop
	float *Ap = new float[n], *pA = new float[n];
	int k; for (k = 0; k < maxiter; k++) {
#ifdef _DEBUG
		printf("%f,", sqrt(linequError2(x, b)/n));
#endif
		this->matvecmul(p, Ap);
		this->vecmatmul(p, pA);
		// α = r* r / p* A p
		float alpha = r20 / vecdot(p, Ap);
		// x = x + α p
		for (int i = 0; i < n; i++)
			x[i] += alpha * p[i],
			xt[i] += alpha * pt[i];
		// r = r - α Ap, r* = r* - α pA
		for (int i = 0; i < n; i++)
			r[i] -= alpha * Ap[i],
			rt[i] -= alpha * pA[i];
		// β = r₁* r₁ / r₀* r₀
		float r21 = vecdot(r, rt);
		if (r21 < tol) { k++; break; }
		float beta = r21 / r20;
		r20 = r21;
		// p = r + β p
		for (int i = 0; i < n; i++)
			p[i] = r[i] + beta * p[i],
			pt[i] = rt[i] + beta * pt[i];
	}
	delete xt; delete r; delete rt; delete p; delete pt;
	delete Ap; delete pA;
	return k;
}

// Solve mat*x=b using stabilized biconjugate gradient, returns the number of iterations.
// https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method#Unpreconditioned_BiCGSTAB
// Experiments show that this function isn't slower than `cg` and `bicg`.
int CsrMatrix::bicgstab(const float *b, float *x, int maxiter, float tol) const {
	tol = float(n)*tol*tol;
	// r = b - A x
	float *r = new float[n];
	this->matvecmul(x, r);
	for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
	// r* = r
	float *r0 = new float[n];
	std::memcpy(r0, r, sizeof(float)*n);
	float *rt = r0;
	// rho0 = alpha = omega0 = 1
	float rho0 = 1.0f,
		alpha = 1.0f,
		omega = 1.0f;
	// v0 = p0 = 0
	float *v = new float[n],
		*p = new float[n],
		*h = new float[n],
		*s = new float[n],
		*t = new float[n];
	for (int i = 0; i < n; i++) v[i] = p[i] = 0.0f;
	// loop
	float *temp = new float[n];
	int k; for (k = 0; k < maxiter; k++) {
#ifdef _DEBUG
		printf("%f,", sqrt(linequError2(x, b)/n));
#endif
		// rho1 = (r0, r)
		float rho1 = vecdot(r0, r);
		// beta = (rho1/rho0)*(alpha/omega)
		float beta = (rho1/rho0)*(alpha/omega);
		rho0 = rho1;
		// p = r + beta*(p-omega*v)
		for (int i = 0; i < n; i++)
			p[i] = r[i] + beta * (p[i] - omega*v[i]);
		// v = A p
		this->matvecmul(p, v);
		// alpha = rho / (rt, v)
		alpha = rho1 / vecdot(rt, v);
		// h = x + alpha * p
		for (int i = 0; i < n; i++)
			h[i] = x[i] + alpha * p[i];
		// s = r - alpha * v
		for (int i = 0; i < n; i++)
			s[i] = r[i] - alpha * v[i];
		// t = A s
		this->matvecmul(s, t);
		// omega = (t, s) / (t, t)
		omega = vecdot(t, s) / vecdot(t, t);
		// x = h + omega * s
		for (int i = 0; i < n; i++)
			x[i] = h[i] + omega * s[i];
		// accurate enough?
		if (linequError2(x, b) <= tol) break;
		// r = s - omega * t
		for (int i = 0; i < n; i++)
			r[i] = s[i] - omega * t[i];
	}
	delete r; delete r0; delete temp;
	delete v; delete p; delete h; delete s; delete t;
	return k;
}
