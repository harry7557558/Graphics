// solve the FEM problem

#pragma once

#include <vector>
#include <unordered_map>

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

    // solvers
    void solveLaplacian();

    // need to do this manually
    void destroyElements() {
        for (const AreaElement* se : SE)
            delete se;
        SE.clear();
    }

};


// find the deflection
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
    float K[MAX_AREA_ELEMENT_N * MAX_AREA_ELEMENT_N];
    LilMatrix lil(Ns);
    float* invDiag = new float[Ns];
    float* diag = new float[Ns];
    for (int i = 0; i < Ns; i++) invDiag[i] = 0.0;
    for (int i = 0; i < Ns; i++) diag[i] = 0.0;
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
                if (i == j) {
                    diag[i] += k;
                    invDiag[i] += k;
                }
            }
        }
    }
    assert(maxPossibleRank >= Ns);
    for (int i = 0; i < Ns; i++)
        invDiag[i] = 1.0 / (invDiag[i]);

    // construct the vectors
    float* f = new float[Ns];
    for (int i = 0; i < N; i++)
        if (Imap[i] != -1)
            f[Imap[i]] = F[i];
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
    auto precond = [&](const float* src, float* res) {
        for (int i = 0; i < Ns; i++)
            res[i] = invDiag[i] * src[i];
    };
#elif PRECOND == 2
    // incomplete Cholesky decomposition
    LilMatrix iclil = lil.incompleteCholesky3();
    CsrMatrix precondL(iclil), precondU(transpose(iclil));
    auto precond = [&](const float* src, float* res) {
        memcpy(res, src, sizeof(float) * Ns);
        precondL.lowerSolve(res);
        precondU.upperSolve(res);
    };
#elif PRECOND == 3
    // SSoR preconditioning
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
    delete[] invDiag; delete[] diag;
}


DiscretizedModel<float, float> solveLaplacianLinearTrig(
    std::vector<vec2> X_,  // N
    std::vector<float> L_,  // N
    std::vector<ivec3> E_  // M
) {
    DiscretizedModel<float, float> structure;
    structure.X = X_;
    structure.F = L_;
    structure.U = std::vector<float>(X_.size(), 0.0);
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

