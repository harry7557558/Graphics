// solve the FEM problem

#pragma once

#include <vector>
#include <unordered_map>
#include <set>
#include <unordered_set>

#include "elements.h"
#include "sparse.h"



// PRNG
#ifndef PI
#define PI 3.1415926535897932384626
#endif
unsigned int _IDUM = 1;
unsigned randu() { return _IDUM = _IDUM * 1664525u + 1013904223u; }
float randf() { return randu() * (1. / 4294967296.); }
float randn() { return sqrt(-2.0 * log(1.0 - randf())) * cos(2.0 * PI * randf()); }




template<typename Tf, typename Tu>
struct DiscretizedModel {
    int N;  // number of vertices
    int M;  // number of elements
    std::vector<vec2> X;  // vertices, N
    std::vector<const AreaElement*> SE;  // elements, M
    std::vector<int> boundary;  // indices of boundary vertices
    std::vector<bool> isBoundary;  // boundary vertex? N
    std::vector<Tf> F;  // inputs, N, valid for non boundary
    std::vector<Tu> U;  // solutions, N
    std::vector<vec2> gradU;  // gradient of solutions, valid for float Tu

    void startSolver() {
        N = (int)X.size();
        assert((int)F.size() == N);
        assert((int)U.size() == N);
        M = (int)SE.size();
        printf("%d vertices, %d elements.\n", N, M);
        isBoundary.resize(N, false);
        for (int i : boundary) {
            assert(i >= 0 && i < (int)N);
            isBoundary[i] = true;
        }
    }

    void constructMatrix(LilMatrix &lil, const int* Imap);
    void constructCotangentMatrix(LilMatrix &lil, std::vector<float> &masses, const int* Imap);

    void constructLaplacianMatrix(LilMatrix &lil, const int *Imap);
    void constructLaplacianMatrixLarge(LilMatrix &lil, const int *Imap);

    // solvers
    void solveLaplacian();

    // need to do this manually
    void destroyElements() {
        for (const AreaElement* se : SE)
            delete se;
        SE.clear();
    }

};

template<>
void DiscretizedModel<float, float>::constructMatrix(
    LilMatrix &lil, const int* Imap
) {
    float K[MAX_AREA_ELEMENT_N * MAX_AREA_ELEMENT_N];
    int maxPossibleRank = 0;
    for (const AreaElement* c : SE) {
        int n = c->getN();
        maxPossibleRank += n;
        const int* vi = c->getVi();
        c->evalK(&X[0], K);
        for (int i0 = 0; i0 < n; i0++) {
            int i = Imap[vi[i0]]; if (i == -1) continue;
            for (int j0 = 0; j0 < n; j0++) {
                int j = Imap[vi[j0]]; if (j == -1) continue;
                float k = K[i0 * n + j0];
                lil.addValue(i, j, k);
            }
        }
    }
}



template<>
void DiscretizedModel<float, float>::constructCotangentMatrix(
    LilMatrix &lil, std::vector<float> &masses, const int* Imap
) {
    int N = lil.getN();
    masses = std::vector<float>(N, 0.0);
    for (const AreaElement *c : SE) {
        assert(c->getN() == 3);
        const int *v0 = c->getVi();
        float dA = determinant(mat2(
            X[v0[1]] - X[v0[0]], X[v0[2]] - X[v0[0]]
        )) / 6.0f;
        for (int i = 0; i < 3; i++)
            if (Imap[v0[i]] != -1)
                masses[Imap[v0[i]]] += dA;
    }
    for (const AreaElement *c : SE) {
        const int *v0 = c->getVi();
        for (int i = 0; i < 3; i++) {
            const int v[3] = {
                v0[i], v0[(i + 1) % 3], v0[(i + 2) % 3]
            };
            vec2 a = X[v[1]] - X[v[0]];
            vec2 b = X[v[2]] - X[v[0]];
            vec2 c = X[v[2]] - X[v[1]];
            float cos = dot(a, b) / sqrt(dot(a, a) * dot(b, b));
            float w = 0.5f * cos / sqrt(1.0f - cos * cos);
            lil.addValue(Imap[v[1]], Imap[v[1]], w);
            lil.addValue(Imap[v[1]], Imap[v[2]], -w);
            lil.addValue(Imap[v[2]], Imap[v[1]], -w);
            lil.addValue(Imap[v[2]], Imap[v[2]], w);
        }
    }
}



template<>
void DiscretizedModel<float, float>::constructLaplacianMatrix(
    LilMatrix &lil, const int* Imap
) {
    // get edges
    std::unordered_map<uint64_t, int> edges;
    for (const AreaElement *c : SE) {
        assert(c->getN() == 3);
        const int *v = c->getVi();
        for (int _ = 0; _ < 3; _++) {
            int i = v[_];
            int j = v[(_ + 1) % 3];
            if (i > j) std::swap(i, j);
            uint64_t idx = ((uint64_t)i << 32) | j;
            if (edges.find(idx) == edges.end())
                edges[idx] = edges.size();
        }
    }

    // get equations
    int N = (int)X.size();
    int M = (int)edges.size();
    std::vector<std::vector<int>> equj; equj.resize(3 * N);
    std::vector<std::vector<float>> equv; equv.resize(3 * N);
    for (const AreaElement *c : SE) {
        const int *v = c->getVi();
        for (int _ = 0; _ < 3; _++) {
            int i = v[_];
            int j = v[(_ + 1) % 3];
            if (i > j && !(Imap[i] == -1 && Imap[j] == -1))
                continue;
            if (i > j) std::swap(i, j);
            uint64_t idx = ((uint64_t)i << 32) | j;
            assert(edges.find(idx) != edges.end());
            int ei = edges[idx];
            vec2 dx = X[j] - X[i];
            float dx2 = dot(dx, dx);
            for (int r = 0; r < 3; r++) {
                equj[r * N + i].push_back(ei);
                equj[r * N + j].push_back(ei);
            }
            equv[0 * N + i].push_back(dx.x);
            equv[0 * N + j].push_back(-dx.x);
            equv[1 * N + i].push_back(dx.y);
            equv[1 * N + j].push_back(-dx.y);
            equv[2 * N + i].push_back(dx2);
            equv[2 * N + j].push_back(dx2);
        }
    }

    printf("Edge weights: %d equs, %d vars\n", 3*N, M);

    // normalize
    std::vector<float> ss(3 * N);
    for (int k = 0; k < 3 * N; k++) {
        float s = 0.0f;
        for (float v : equv[k]) s += v * v;
        s = 1.0f / sqrt(s);
        if (k < 2 * N) s *= 0.2f;
        for (size_t i = 0; i < equv[k].size(); i++)
            equv[k][i] *= s;
        ss[k] = s;
    }
    std::vector<float> equb(M, 0.0f);
    for (int k = 2 * N; k < 3 * N; k++) {
        for (size_t i = 0; i < equv[k].size(); i++) {
            equb[equj[k][i]] += 4.0f * ss[k] * equv[k][i];
        }
    }

    // solve for weights
    LilMatrix lilw(M);
    for (int k = 0; k < 3 * N; k++) {
        size_t l = equj[k].size();
        for (int i = 0; i < l; i++)
            for (int j = 0; j < l; j++) {
                float ks = (i == j) ? (k < 2 * N ? 1.3f : 1.1f) : 1.0f;
                lilw.addValue(equj[k][i], equj[k][j], ks * equv[k][i] * equv[k][j]);
            }
    }
    CsrMatrix csr(lilw);
    auto linopr = [&](const float* src, float* res) {
        csr.matvecmul(src, res);
    };
    std::vector<float> invDiag(M, 0.0f);
    for (int i = 0; i < M; i++)
        invDiag[i] = 1.0 / lilw.at(i, i, 1.0f);
    auto precond = [&](const float* src, float* res) {
        for (int i = 0; i < M; i++)
            res[i] = invDiag[i] * src[i];
    };
    std::vector<float> x(M, 0.0f);
    float tol = 0.0;
    for (int i = 0; i < M; i++)
        tol += dot(equb[i], equb[i]);
    tol = 1e-6 * sqrt(tol);
    int niters = conjugateGradientPreconditioned(
        M, linopr, precond, &equb[0], &x[0], 100000, tol);
    printf("%d iterations\n", niters);

    // construct matrix
    for (std::pair<uint64_t, int> ei : edges) {
        int i = Imap[(int)(ei.first >> 32)];
        int j = Imap[(int)ei.first];
        float w = x[ei.second];
        lil.addValue(i, i, w);
        lil.addValue(i, j, -w);
        lil.addValue(j, i, -w);
        lil.addValue(j, j, w);
    }
}



template<>
void DiscretizedModel<float, float>::constructLaplacianMatrixLarge(
    LilMatrix &lil, const int* Imap
) {
    // get edges
    std::vector<std::vector<int>> neighbors(N);
    for (const AreaElement *c : SE) {
        assert(c->getN() == 3);
        const int *v = c->getVi();
        for (int _ = 0; _ < 3; _++) {
            int i = v[_];
            int j = v[(_ + 1) % 3];
            neighbors[i].push_back(j);
            neighbors[j].push_back(i);
        }
    }

    // expand edges
    std::vector<std::unordered_set<int>> neighbors1(N);
    for (int i = 0; i < (int)neighbors.size(); i++) {
        for (int j : neighbors[i]) {
            neighbors1[i].insert(j);
            for (int k : neighbors[j]) {
                // if (Imap[i] == -1 || Imap[j] == -1 || Imap[k] == -1)
                if (Imap[i] == -1 || Imap[j] == -1)
                // if (Imap[i] == -1)
                neighbors1[i].insert(k);
            }
        }
    }
    std::vector<ivec2> edges;
    for (int i = 0; i < (int)neighbors1.size(); i++)
        for (int j : neighbors1[i])
            if (i < j) edges.push_back(ivec2(i, j));

    // get equations
    int N = (int)X.size();
    int M = (int)edges.size();
    std::vector<std::vector<int>> equj; equj.resize(M);
    std::vector<std::vector<float>> equv; equv.resize(M);
    for (int ei = 0; ei < M; ei++) {
        int i = edges[ei].x, j = edges[ei].y;
        vec2 dx = X[j] - X[i];
        float dx2 = dot(dx, dx);
        for (int r = 0; r < 3; r++) {
            equj[ei].push_back(r * N + i);
            equj[ei].push_back(r * N + j);
        }
        equv[ei].push_back(dx.x);
        equv[ei].push_back(-dx.x);
        equv[ei].push_back(dx.y);
        equv[ei].push_back(-dx.y);
        equv[ei].push_back(dx2);
        equv[ei].push_back(dx2);
    }

    printf("Edge weights: %d equs, %d vars\n", 3*N, M);

    // normalize
    std::vector<float> ss(3 * N, 0.0f);
    for (int k = 0; k < M; k++) {
        for (size_t i = 0; i < equv[k].size(); i++)
            ss[equj[k][i]] += equv[k][i] * equv[k][i];
    }
    for (int i = 0; i < 3 * N; i++) {
        ss[i] = 1.0f / sqrt(ss[i]);
        if (i < 2 * N) ss[i] *= 1.0f;
    }
    std::vector<float> equb(3 * N, 0.0f);
    for (int k = 0; k < M; k++) {
        for (size_t i = 0; i < equv[k].size(); i++)
            equv[k][i] *= ss[equj[k][i]];
    }
    for (int k = 2 * N; k < 3 * N; k++)
        equb[k] = 4.0f * ss[k];

    // solve for weights
    LilMatrix lilw(3 * N);
    for (int k = 0; k < M; k++) {
        int l = equj[k].size();
        for (int i = 0; i < l; i++)
            for (int j = 0; j < l; j++)
                lilw.addValue(equj[k][i], equj[k][j], equv[k][i] * equv[k][j]);
    }
    const float ks = 0.3f;  // stability?
    for (int i = 0; i < 2 * N; i++)
        lilw.addValue(i, i, ks * lilw.at(i, i, 1.0f));
    CsrMatrix csr(lilw);
    auto linopr = [&](const float* src, float* res) {
        csr.matvecmul(src, res);
    };
    std::vector<float> invDiag(3 * N, 0.0f);
    for (int i = 0; i < 3 * N; i++)
        invDiag[i] = 1.0 / lilw.at(i, i, 1.0f);
    auto precond = [&](const float* src, float* res) {
        for (int i = 0; i < 3 * N; i++)
            res[i] = invDiag[i] * src[i];
    };
    std::vector<float> x(3 * N, 0.0f);
    float tol = 0.0;
    for (int i = 0; i < M; i++)
        tol += dot(equb[i], equb[i]);
    tol = 1e-6 * sqrt(tol);
    int niters = conjugateGradientPreconditioned(
        3 * N, linopr, precond, &equb[0], &x[0], 10000, tol);
    printf("%d iterations\n", niters);

    // construct matrix
    std::vector<float> ws(M, 0.0f);
    for (int k = 0; k < M; k++) {
        for (size_t i = 0; i < equv[k].size(); i++)
            ws[k] += equv[k][i] * x[equj[k][i]];
    }
    for (int ei = 0; ei < M; ei++) {
        int i = Imap[edges[ei].x];
        int j = Imap[edges[ei].y];
        float w = ws[ei];
        lil.addValue(i, i, w);
        lil.addValue(i, j, -w);
        lil.addValue(j, i, -w);
        lil.addValue(j, j, w);
    }
}


template<>
void DiscretizedModel<float, float>::solveLaplacian() {
    float time0 = getTimePast();
    startSolver();
    // return;

    // map vertex indices to indices in the linear system
    // (don't consider fixed vertices)
    int* Imap = new int[N];
    for (int i = 0; i < N; i++) Imap[i] = 0;
    for (int i : boundary) Imap[i] = -1;
    int Ns = 0;
    for (int i = 0; i < N; i++) {
        if (Imap[i] != -1) Imap[i] = Ns++;
    }

    // construct the matrix
    LilMatrix lil(Ns);
    std::vector<float> masses(Ns, 1.0f);
    // constructMatrix(lil, Imap);
    constructCotangentMatrix(lil, masses, Imap);
    // constructLaplacianMatrix(lil, Imap);
    // constructLaplacianMatrixLarge(lil, Imap);

    // construct the vectors
    float* f = new float[Ns];
    for (int i = 0; i < N; i++)
        if (Imap[i] != -1)
            f[Imap[i]] = masses[Imap[i]] * F[i];
    float* u = new float[Ns];
    for (int i = 0; i < Ns; i++)
        u[i] = 1e-4f * randn();

    // solve the linear system
    CsrMatrix csr(lil);
    auto linopr = [&](const float* src, float* res) {
        csr.matvecmul(src, res);
    };
    float time1 = getTimePast();
    printf("Linear system constructed in %.2g secs. (%dx%d, %d nonzeros)\n",
        time1 - time0, Ns, Ns, csr.getNonzeros());
    // tolerance
    float tol = 0.0;
    for (int i = 0; i < Ns; i++)
        tol += dot(f[i], f[i]);
    tol = 1e-10 * sqrt(tol);
#define PRECOND 1  // 1: diag; 2: cholesky; 3: ssor
#if !PRECOND
    float time2 = time1;
    int niters = conjugateGradient(
        Ns, linopr, (float*)f, (float*)u, 10000, tol);
#else  // !PRECOND
#if PRECOND == 1
    // block diagonal preconditioning
    std::vector<float> invDiag(Ns, 0.0);
    for (int i = 0; i < Ns; i++)
        invDiag[i] = 1.0 / lil.at(i, i, 1.0);
    auto precond = [&](const float* src, float* res) {
        for (int i = 0; i < Ns; i++)
            res[i] = invDiag[i] * src[i];
    };
#elif PRECOND == 2
    // incomplete Cholesky decomposition
    LilMatrix iclil = lil.incompleteCholesky1();
    CsrMatrix precondL(iclil), precondU(iclil.transpose());
    auto precond = [&](const float* src, float* res) {
        memcpy(res, src, sizeof(float) * Ns);
        precondL.lowerSolve(res);
        precondU.upperSolve(res);
    };
#elif PRECOND == 3
    // SSoR preconditioning
    std::vector<float> diag(Ns, 0.0);
    for (int i = 0; i < Ns; i++)
        diag[i] = lil.at(i, i, 1.0);
    CsrMatrix precondL(lil, CsrMatrix::FROM_LIL_LOWER);
    CsrMatrix precondU(lil, CsrMatrix::FROM_LIL_UPPER);
    auto precond = [&](const float* src, float* res) {
        memcpy(res, src, sizeof(float) * Ns);
        precondL.lowerSolve(res);
        for (int i = 0; i < Ns; i++) res[i] *= diag[i];
        precondU.upperSolve(res);
    };
#endif  // preconditioner
    float time2 = getTimePast();
    printf("Linear system preconditioned in %.2g secs.\n", time2 - time1);
    int niters = conjugateGradientPreconditioned(
        Ns, linopr, precond, (float*)f, (float*)u, 10000, tol);
#endif  // !PRECOND
    printf("%d iterations.\n", niters);
    float time3 = getTimePast();
    printf("Linear system solved in %.2g secs. (includes preconditioning)\n", time3 - time1);

    // get the result
    for (int i = 0; i < N; i++)
        U[i] = Imap[i] == -1 ? 0.0 : u[Imap[i]];
    delete[] Imap; delete[] f; delete[] u;
}


DiscretizedModel<float, float> solveLaplacianLinearTrig(
    std::vector<vec2> X_,  // N
    std::vector<float> L_,  // N
    std::vector<ivec3> E_  // M
) {
    DiscretizedModel<float, float> structure;
    structure.X = X_;
    structure.F = L_;
    structure.U = std::vector<float>(X_.size(), 0.0f);
    structure.boundary.clear();

    std::map<uint64_t, int> edges;
    for (ivec3 t : E_) {
        for (int _ = 0; _ < 3; _++) {
            uint64_t i = t[_], j = t[(_+1)%3];
            if (i > j) std::swap(j, i);
            edges[(i<<32)|j] += 1;
        }
    }
    std::vector<bool> isBoundary = std::vector<bool>(X_.size(), false);
    for (std::pair<uint64_t, int> ec : edges) {
        if (ec.second != 1) continue;
        uint64_t e = ec.first;
        int i = (int)(e >> 32), j = (int)e;
        isBoundary[i] = isBoundary[j] = true;
    }
    for (int i = 0; i < (int)isBoundary.size(); i++)
        if (isBoundary[i]) structure.boundary.push_back(i);

    for (ivec3 e : E_)
        structure.SE.push_back(new LinearTrigElement(
            (int*)&e, &structure.X[0]));
    structure.solveLaplacian();

    return structure;
}

DiscretizedModel<float, float> solveLaplacianQuadraticTrig(
    std::vector<vec2> X_,  // N
    std::vector<float> L_,  // N
    std::vector<ivec3> E_  // M
) {
    DiscretizedModel<float, float> structure;
    structure.X = X_;
    structure.F = L_;
    structure.U = std::vector<float>(X_.size(), 0.0f);
    structure.boundary.clear();

    std::map<uint64_t, int> edges;
    for (ivec3 t : E_) {
        for (int _ = 0; _ < 3; _++) {
            uint64_t i = t[_], j = t[(_+1)%3];
            if (i > j) std::swap(j, i);
            edges[(i<<32)|j] += 1;
        }
    }
    std::vector<bool> isBoundary = std::vector<bool>(X_.size(), false);
    for (std::pair<uint64_t, int> ec : edges) {
        if (ec.second != 1) continue;
        uint64_t e = ec.first;
        int i = (int)(e >> 32), j = (int)e;
        isBoundary[i] = isBoundary[j] = true;
    }

    for (std::pair<uint64_t, int> ec : edges) {
        isBoundary.push_back(ec.second == 1);
        edges[ec.first] = (int)structure.X.size();
        int i = (int)ec.first, j = (int)(ec.first >> 32);
        structure.X.push_back(0.5f*(X_[i]+X_[j]));
        structure.F.push_back(0.5f*(L_[i]+L_[j]));
        structure.U.push_back(0.0f);
    }
    for (int i = 0; i < (int)isBoundary.size(); i++)
        if (isBoundary[i]) structure.boundary.push_back(i);

    for (ivec3 t : E_) {
        int e[6] = { t[0], t[1], t[2] };
        for (int _ = 0; _ < 3; _++) {
            uint64_t i = t[_], j = t[(_+1)%3];
            if (i > j) std::swap(j, i);
            e[_+3] = edges[(i<<32)|j];
        }
        structure.SE.push_back(new QuadraticTrigElement(e, &structure.X[0]));
    }
    structure.solveLaplacian();

    return structure;
}
