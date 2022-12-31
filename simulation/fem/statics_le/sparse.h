#include <vector>
#include <map>
#include <functional>
#include <cstring>  // memcpy
#include <cstdio>

#include <assert.h>


class LilMatrix {
    int n;  // n*n square matrix
    std::vector<std::map<int, double>> mat;  // list of dict?!

public:
    friend class CsrMatrix;

    LilMatrix(int n) {
        this->n = n;
        mat.resize(n);
    }
    int getN() const { return n; }
    void addValue(int row, int col, double val) {
        mat[row][col] += val;
    }
    void setValue(int row, int col, double val) {
        mat[row][col] = val;
    }
    double at(int i, int j, double default_ = 0.0) const {
        auto p = mat[i].find(j);
        if (p == mat[i].end()) return default_;
        return p->second;
    }

    // return true iff the matrix is strictly symmetric
    bool isSymmetric() const {
        for (int i = 0; i < n; i++) {
            for (std::pair<int, double> jw : mat[i]) {
                int j = jw.first;
                if (j >= i) break;
                auto p = mat[j].find(i);
                if (p == mat[j].end() || p->second != jw.second)
                    return false;
            }
        }
        return true;
    }

    // visualize
    void print(FILE* fp = stdout) const {
        char* c = new char[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) c[j] = ' ';
            for (std::pair<int, double> jw : mat[i]) {
                if (jw.second > 0.0) c[jw.first] = '+';
                if (jw.second < 0.0) c[jw.first] = '-';
                if (std::isnan(jw.second)) c[jw.first] = '#';
                //printf("%lf ", jw.second);
            }
            fwrite(c, 1, n, fp);
            fputc('\n', fp);
        }
        delete c;
    }
    void printOctave(FILE* fp = stdout) const {
        fprintf(fp, "[");
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                fprintf(fp, "%.12lg%c", this->at(i, j), j + 1 == n ? ';' : ',');
            }
        fprintf(fp, "];");
    }

    // res = mat * src
    template<typename vec>
    void matvecmul(const vec* src, vec* res) const {
        for (int i = 0; i < n; i++) {
            res[i] = vec(0.0);
            for (std::pair<int, double> jw : mat[i]) {
                res[i] += jw.second * src[jw.first];
            }
        }
    }

    // transpose
    LilMatrix transpose() const {
        LilMatrix res(n);
        for (int i = 0; i < n; i++) {
            for (std::pair<int, double> jw : mat[i]) {
                res.addValue(jw.first, i, jw.second);
            }
        }
        return res;
    }

    // Incomplete Cholesky factorization
    // result in an lower triangular matrix
    LilMatrix incompleteCholesky1() const {
        LilMatrix res(n);
        for (int i = 0; i < n; i++) {
            double diag = sqrt(abs(this->at(i, i)));
            res.addValue(i, i, diag);
            for (std::pair<int, double> kw : mat[i]) {
                int k = kw.first;
                if (k > i) res.addValue(k, i, kw.second / diag);
            }
        }
        return res;
    }
    // 3x3 block
    LilMatrix incompleteCholesky3() const {
        assert(n % 3 == 0);
        LilMatrix res(n);
        for (int i = 0; i < n; i += 3) {
            // get the diagonal
            mat3 diag;
            for (int u = 0; u < 3; u++)
                for (int v = 0; v < 3; v++)
                    diag[u][v] = this->at(i + u, i + v);
            for (int u = 0; u < 3; u++)
                for (int v = 0; v <= u; v++) {
                    double s = 0.0;
                    for (int w = 0; w < v; w++)
                        s += diag[u][w] * diag[v][w];
                    diag[u][v] = (u == v) ? sqrt(diag[u][u] - s)
                        : (diag[v][u] - s) / diag[v][v];
                    res.addValue(i + u, i + v, diag[u][v]);
                }
            diag[0][1] = diag[0][2] = diag[1][2] = 0.0;
            // this should work because all 3x3 blocks (including zeros)
            // were added in stiffness matrix construction
            mat3 invDiag = inverse(diag.transpose());
            for (std::pair<int, double> kw : mat[i]) {
                int k = kw.first;
                if (k <= i || k % 3 != 0) continue;
                mat3 vik;  // transposed
                for (int u = 0; u < 3; u++)
                    for (int v = 0; v < 3; v++)
                        vik[v][u] = this->at(i + u, k + v);
                mat3 m = vik * invDiag;
                for (int u = 0; u < 3; u++)
                    for (int v = 0; v < 3; v++)
                        res.addValue(k + u, i + v, m[u][v]);
            }
        }
        return res;
    }

};


class CsrMatrix {
    int n;  // n*n square matrix
    std::vector<int> rows;  // see Wikipedia
    std::vector<int> cols;
    std::vector<double> vals;

public:
    // for convertion from LiLMatrix
    enum FromLilMode {
        FROM_LIL_NONZERO,  // only nonzero elements
        FROM_LIL_FORCE_DIAG,  // must have diagonal, including zeros
        FROM_LIL_LOWER,  // lower triangular with diagonal
        FROM_LIL_UPPER,  // upper triangular with diagonal
    };

    CsrMatrix(): n(0) {}
    CsrMatrix(const LilMatrix& lil, FromLilMode mode = FROM_LIL_FORCE_DIAG) {
        this->n = lil.n;
        rows.push_back(0);
        for (int i = 0; i < n; i++) {
            for (std::pair<int, double> indice : lil.mat[i]) {
                if (
                    (indice.second != 0.0 || (mode != FROM_LIL_NONZERO && indice.first == i))
                    && !(mode == FROM_LIL_LOWER && indice.first > i)
                    && !(mode == FROM_LIL_UPPER && indice.first < i)
                 ) {
                    cols.push_back(indice.first);
                    vals.push_back(indice.second);
                }
            }
            rows.push_back((int)cols.size());
        }
    }
    int getN() const { return n; }
    int getNonzeros() const { return (int)vals.size(); }

    // square of norm of a vector
    double vecnorm2(const double* r) const {
        double ans = 0.0;
        for (int i = 0; i < n; i++) ans += r[i] * r[i];
        return ans;
    }

    // dot product between two vectors
    double vecdot(const double* u, const double* v) const {
        double ans = 0.0;
        for (int i = 0; i < n; i++) ans += u[i] * v[i];
        return ans;
    }

    // multiply by scalar
    void operator*=(double k) {
        for (int i = 0, l = rows.back(); i < l; i++)
            vals[i] *= k;
    }

    // res = mat * src
    void matvecmul(const double* src, double* res) const {
        for (int i = 0; i < n; i++) {
            res[i] = 0.0;
            for (int ji = rows[i]; ji < rows[i + 1]; ji++) {
                res[i] += vals[ji] * src[cols[ji]];
            }
        }
    }

    // res = mat^T * src
    void vecmatmul(const double* src, double* res) const {
        for (int j = 0; j < n; j++)
            res[j] = 0.0;
        for (int i = 0; i < n; i++) {
            for (int ji = rows[i]; ji < rows[i + 1]; ji++) {
                res[cols[ji]] += vals[ji] * src[i];
            }
        }
    }

    // return u^T * mat * v
    double vecmatvecmul(const double* u, const double* v) const {
        double res = 0.0;
        for (int i = 0; i < n; i++) {
            for (int ji = rows[i]; ji < rows[i + 1]; ji++) {
                res += u[i] * vals[ji] * v[cols[ji]];
            }
        }
        return res;
    }

    // solve a lower triangular matrix equation
    void lowerSolve(double* x) const {
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int ji = rows[i]; ji < rows[i + 1] - 1; ji++)
                s += vals[ji] * x[cols[ji]];
            int ji = rows[i + 1] - 1;
            assert(i == cols[ji]);
            x[i] = (x[i] - s) / vals[ji];
        }
    }
    // solve an upper triangular matrix equation
    void upperSolve(double* x) const {
        for (int i = n - 1; i >= 0; i--) {
            double s = 0.0;
            for (int ji = rows[i + 1] - 1; ji > rows[i]; ji--)
                s += vals[ji] * x[cols[ji]];
            int ji = rows[i];
            assert(i == cols[ji]);
            x[i] = (x[i] - s) / vals[ji];
        }
    }

    // evaluate (mat*x-b)^2
    double linequError2(const double* x, const double* b) const {
        double toterr = 0.0;
        for (int i = 0; i < n; i++) {
            double err = -b[i];
            for (int ji = rows[i]; ji < rows[i + 1]; ji++) {
                err += vals[ji] * x[cols[ji]];
            }
            toterr += err * err;
        }
        return toterr;
    }
};



double vecnorm2(int n, const double* r) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += r[i] * r[i];
    return s;
}
double vecdot(int n, const double* u, const double* v) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += u[i] * v[i];
    return s;
}

// conjugate gradient from symmetric linear operator
// x must be initialized
// returns the number of iterations
int conjugateGradient(
    int n,
    std::function<void(const double* src, double* res)> A,
    const double* b, double* x, int maxiter, double tol
) {
    // r = b - Ax
    double* r = new double[n];
    A(x, r);
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
    double r20 = vecnorm2(n, r);
    // p = r
    double* p = new double[n];
    std::memcpy(p, r, n * sizeof(double));
    // loop
    double* Ap = new double[n];
    int k; for (k = 0; k < maxiter; k++) {
        // α = rᵀr / pᵀAp
        A(p, Ap);
        double alpha = r20 / vecdot(n, p, Ap);
        // x = x + αp
        for (int i = 0; i < n; i++) x[i] += alpha * p[i];
        // r = r - αAp
        for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];
        // β = r₁ᵀr₁ / r₀ᵀr₀
        double r21 = vecnorm2(n, r);
        if (r21 < tol * tol || std::isnan(r21)) { k++; break; }
        double beta = r21 / r20;
        r20 = r21;
        // p = r + βp
        for (int i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
        // verbose
        if ((k + 1) % 100 == 0) {
            double maxdif = 0.0;
            for (int i = 0; i < n; i += 3)
                maxdif = max(maxdif, length(*((vec3*)&r[i])));
            printf("%d %lg\n", k + 1, maxdif);
        }
    }
    delete[] r; delete[] p; delete[] Ap;
    return k;
}

// preconditioned conjugate gradient
int conjugateGradientPreconditioned(
    int n,
    std::function<void(const double* src, double* res)> A,
    std::function<void(const double* src, double* res)> M,
    const double* b, double* x, int maxiter, double tol
) {
    // r = b - Ax
    double* r = new double[n];
    A(x, r);
    for (int i = 0; i < n; i++) r[i] = b[i] - r[i];
    // z = M⁻¹ r
    double* z = new double[n];
    M(r, z);
    double rz0 = vecdot(n, r, z);
    // p = r
    double* p = new double[n];
    std::memcpy(p, z, n * sizeof(double));
    // loop
    double* Ap = new double[n];
    int k; for (k = 0; k < maxiter; k++) {
        // α = rᵀz / pᵀAp
        A(p, Ap);
        double alpha = rz0 / vecdot(n, p, Ap);
        // x = x + αp
        for (int i = 0; i < n; i++) x[i] += alpha * p[i];
        // r = r - αAp
        for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];
        double r2 = vecnorm2(n, r);
        if (r2 < tol * tol || std::isnan(r2)) { k++; break; }
        // z₁ = M⁻¹ r₁
        M(r, z);
        // β = r₁ᵀz₁ / r₀ᵀz₀
        double rz1 = vecdot(n, r, z);
        double beta = rz1 / rz0;
        rz0 = rz1;
        // p = z + βp
        for (int i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        // verbose
        if ((k + 1) % 100 == 0) {
            double maxdif = 0.0;
            for (int i = 0; i < n; i += 3)
                maxdif = max(maxdif, length(*((vec3*)&r[i])));
            printf("%d %lg\n", k + 1, maxdif);
        }
    }
    delete[] r; delete[] z; delete[] p; delete[] Ap;
    return k;
}
