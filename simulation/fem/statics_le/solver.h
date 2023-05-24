// solve the FEM problem

#pragma once

#include <vector>
#include <unordered_map>

#include "elements.h"
#include "sparse.h"


// timer
#include <chrono>
auto _TIME_START = std::chrono::high_resolution_clock::now();
double getTimePast() {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - _TIME_START).count();
}



// PRNG
#ifndef PI
#define PI 3.1415926535897932384626
#endif
unsigned int _IDUM = 1;
unsigned randu() { return _IDUM = _IDUM * 1664525u + 1013904223u; }
double randf() { return randu() * (1. / 4294967296.); }
double randn() { return sqrt(-2.0 * log(1.0 - randf())) * cos(2.0 * PI * randf()); }




// calculate the matrix transforming strain to stress for a linear isotropic material
// E: Young's modulus
// nu: Poisson's ratio
// C: a 6x6 matrix
void calculateStressStrainMatrix(double E, double nu, double* C) {
    for (int i = 0; i < 36; i++) C[i] = 0.0;
    double k = E / ((1.0 + nu) * (1.0 - 2.0 * nu));
    C[0] = C[7] = C[14] = k * (1.0 - nu);
    C[1] = C[2] = C[6] = C[8] = C[12] = C[13] = k * nu;
    C[21] = C[28] = C[35] = k * (0.5 - nu);
}


struct DiscretizedStructure {
    int N;  // number of vertices
    int M;  // number of elements
    std::vector<vec3> X;  // vertices, N
    std::vector<const SolidElement*> SE;  // elements, M
    std::vector<vec3> F;  // forces at vertices, N
    std::vector<int> fixed;  // indices of fixed vertices
    std::vector<bool> isFixed;  // fixed vertex? N
    std::vector<vec3> U;  // vertice deflections, N
    std::vector<mat3> Sigma;  // stress tensors, N

    void startSolver() {
        assert(X.size() == F.size());
        N = (int)X.size();
        M = (int)SE.size();
        printf("%d vertices, %d elements.\n", N, M);
        isFixed.resize(N, false);
        for (int i : fixed) {
            assert(i >= 0 && i < N);
            isFixed[i] = true;
        }
        U.resize(N, vec3(0));
        Sigma.resize(N, mat3(0));
    }

    void solveDeflection(const double* C);

    // do this after `solveDeflection()`
    void calcForceStress(const double* C);

    // need to do this manually
    void destroyElements() {
        for (const SolidElement* se : SE)
            delete se;
        SE.clear();
    }

    // scalars that may be visualized
    std::vector<float> getSigmaComp(int i, int j) const {
        std::vector<float> res(N);
        for (int k = 0; k < N; k++) res[k] = (float)Sigma[k][i][j];
        return res;
    }
    std::vector<float> getSigmaX() const { return getSigmaComp(0, 0); }  // stress x
    std::vector<float> getSigmaY() const { return getSigmaComp(1, 1); }  // stress y
    std::vector<float> getSigmaZ() const { return getSigmaComp(2, 2); }  // stress z
    std::vector<float> getTauYZ() const { return getSigmaComp(1, 2); }  // shear yz
    std::vector<float> getTauXZ() const { return getSigmaComp(0, 2); }  // shear xz
    std::vector<float> getTauXY() const { return getSigmaComp(0, 1); }  // shear xy
    std::vector<float> getSigmaVonMises() const {  // von mises stress
        std::vector<float> res(N);
        for (int k = 0; k < N; k++) {
            mat3 s = Sigma[k];
            vec3 p(s[1][1] - s[2][2], s[0][0] - s[2][2], s[0][0] - s[1][1]);
            vec3 q(s[1][2], s[0][2], s[0][1]);
            res[k] = (float)sqrt(0.5 * dot(p, p) + 3.0 * dot(q, q));
        }
        return res;
    }
    std::vector<float> getSigmaT() const {  // tensile stress
        std::vector<float> res(N);
        for (int k = 0; k < N; k++) {
            vec3 eigs = eigvalsh(Sigma[k]);
            res[k] = (float)max(max(eigs.x, eigs.y), max(eigs.z, 0.0));
        }
        return res;
    }
    std::vector<float> getSigmaC() const {  // compressive stress
        std::vector<float> res(N);
        for (int k = 0; k < N; k++) {
            vec3 eigs = eigvalsh(Sigma[k]);
            res[k] = (float)min(min(eigs.x, eigs.y), min(eigs.z, 0.0));
        }
        return res;
    }
    std::vector<float> getVecComp(const std::vector<vec3>& v, int i) const {
        std::vector<float> res(N);
        for (int k = 0; k < N; k++)
            res[k] = float(((double*)&v[k])[i]);
        return res;
    }
    std::vector<float> getUX() const { return getVecComp(U, 0); }  // deflection x
    std::vector<float> getUY() const { return getVecComp(U, 1); }  // deflection y
    std::vector<float> getUZ() const { return getVecComp(U, 2); }  // deflection z
    std::vector<float> getUM() const {  // deflection magnitude
        std::vector<float> res(N);
        for (int k = 0; k < N; k++) res[k] = length(U[k]);
        return res;
    }
};


// find the deflection
void DiscretizedStructure::solveDeflection(const double* C) {
    double time0 = getTimePast();
    startSolver();
    return;

    // map vertex indices to indices in the linear system
    // (don't consider fixed vertices)
    int* Imap = new int[N];
    for (int i = 0; i < N; i++) Imap[i] = 0;
    for (int i : fixed) Imap[i] = -1;
    int Ns = 0;
    for (int i = 0; i < N; i++) {
        if (Imap[i] != -1) Imap[i] = Ns++;
    }

    // construct the matrix
    double K[9 * MAX_SOLID_ELEMENT_N * MAX_SOLID_ELEMENT_N];
    LilMatrix lil(3 * Ns);
    mat3* invDiag = new mat3[Ns];
    double* diag = new double[3 * Ns];
    for (int i = 0; i < Ns; i++) invDiag[i] = mat3(0);
    for (int i = 0; i < 3 * Ns; i++) diag[i] = 0.0;
    int maxPossibleRank = 0;
    for (const SolidElement* c : SE) {
        int n = c->getN();
        maxPossibleRank += n;
        const int* vi = c->getVi();
        c->evalK(&X[0], C, K);
        for (int i0 = 0; i0 < n; i0++) {
            int i = Imap[vi[i0]]; if (i == -1) continue;
            for (int j0 = 0; j0 < n; j0++) {
                int j = Imap[vi[j0]]; if (j == -1) continue;
                mat3 m;
                for (int u = 0; u < 3; u++) for (int v = 0; v < 3; v++) {
                    double k = K[(3 * i0 + u) * (3 * n) + (3 * j0 + v)];
                    lil.addValue(3 * i + u, 3 * j + v, k);
                    m[u][v] = k;
                }
                if (i == j) {
                    invDiag[i] += m;
                    for (int t = 0; t < 3; t++)
                        diag[3 * i + t] += m[t][t];
                }
            }
        }
    }
    assert(maxPossibleRank >= Ns);
    for (int i = 0; i < Ns; i++)
        invDiag[i] = inverse(invDiag[i]);

    // construct the vectors
    vec3* f = new vec3[Ns];
    for (int i = 0; i < N; i++)
        if (Imap[i] != -1)
            f[Imap[i]] = F[i];
    vec3* u = new vec3[Ns];
    for (int i = 0; i < Ns; i++)
        u[i] = 1e-100 * vec3(randn(), randn(), randn());

    // solve the linear system
    CsrMatrix csr(lil);
    auto linopr = [&](const double* src, double* res) {
        csr.matvecmul(src, res);
    };
    double time1 = getTimePast();
    printf("Linear system constructed in %.2lf secs. (%dx%d, %d nonzeros)\n",
        time1 - time0, 3 * Ns, 3 * Ns, csr.getNonzeros());
    // tolerance
    double tol = 0.0;
    for (int i = 0; i < Ns; i++)
        tol += dot(f[i], f[i]);
    tol = 1e-10 * sqrt(tol);
#define PRECOND 1  // 1: diag; 2: cholesky; 3: ssor
#if !PRECOND
    double time2 = time1;
    int niters = conjugateGradient(
        3 * Ns, linopr, (double*)f, (double*)u, 10000, tol);
#else  // !PRECOND
#if PRECOND == 1
    // block diagonal preconditioning
    auto precond = [&](const double* src, double* res) {
        for (int i = 0; i < Ns; i++)
            ((vec3*)res)[i] = invDiag[i] * ((vec3*)src)[i];
    };
#elif PRECOND == 2
    // incomplete Cholesky decomposition
    LilMatrix iclil = lil.incompleteCholesky3();
    CsrMatrix precondL(iclil), precondU(iclil.transpose());
    auto precond = [&](const double* src, double* res) {
        memcpy(res, src, sizeof(double) * 3 * Ns);
        precondL.lowerSolve(res);
        precondU.upperSolve(res);
    };
#elif PRECOND == 3
    // SSoR preconditioning
    CsrMatrix precondL(lil, CsrMatrix::FROM_LIL_LOWER);
    CsrMatrix precondU(lil, CsrMatrix::FROM_LIL_UPPER);
    auto precond = [&](const double* src, double* res) {
        memcpy(res, src, sizeof(double) * 3 * Ns);
        precondL.lowerSolve(res);
        for (int i = 0; i < 3 * Ns; i++) res[i] *= diag[i];
        precondU.upperSolve(res);
    };
#endif  // preconditioner
    double time2 = getTimePast();
    printf("Linear system preconditioned in %.2lf secs.\n", time2 - time1);
    int niters = conjugateGradientPreconditioned(
        3 * Ns, linopr, precond, (double*)f, (double*)u, 10000, tol);
#endif  // !PRECOND
    printf("%d iterations.\n", niters);
    double time3 = getTimePast();
    printf("Linear system solved in %.2lf secs. (includes preconditioning)\n", time3 - time1);

    // get the result
    for (int i = 0; i < N; i++)
        U[i] = Imap[i] == -1 ? vec3(0) : u[Imap[i]];
    delete[] Imap; delete[] f; delete[] u;
    delete[] invDiag; delete diag;
}


// solve a structure discretized into tetrahedra
// X: list of vertex initial positions
// E: list of elements
// Fs: list of surface tractions
// Fv: list of volume tractions
// fixed: list of fixed vertices
// C: strain to stress matrix
// order: the order of elements, must be 1 or 2
// returns: a list of deflections
DiscretizedStructure solveStructureTetrahedral(
    std::vector<vec3> X_,
    std::vector<ivec4> E_,
    std::vector<ElementForce3> Fs,
    std::vector<ElementForce4> Fv,
    std::vector<int> fixed_,
    const double* C,
    int order
) {
    assert(order == 1 || order == 2);
    DiscretizedStructure structure;
    structure.X = X_;
    structure.F.resize(structure.X.size(), vec3(0));
    structure.fixed = fixed_;

    // linear tetrahedral
    if (order == 1) {
        for (ivec4 e : E_)
            structure.SE.push_back(new LinearTetrahedralElement(
                (int*)&e, &structure.X[0]));
        for (ElementForce3 ef : Fs) {
            vec3 f = ef.F / 3.0;
            for (int i = 0; i < 3; i++)
                structure.F[((int*)&ef)[i]] += f;
        }
        for (ElementForce4 ef : Fv) {
            vec3 f = ef.F / 4.0;
            for (int i = 0; i < 4; i++)
                structure.F[((int*)&ef)[i]] += f;
        }
        structure.solveDeflection(C);
        return structure;
    }

    // quadratic tetrahedral

    // edges
    int N0 = (int)structure.X.size();
    printf("%d vertices.\n", N0);
    std::unordered_map<uint64_t, int> edges;  // ivec2 -> index
    auto e2e = [&](int a, int b) -> uint64_t {
        ivec2 e2(max(a, b), min(a, b));
        return *((uint64_t*)&e2);
    };
    std::vector<bool> isFixed(N0, false);
    for (int i : structure.fixed) isFixed[i] = true;
    auto addEdge = [&](int a, int b) {
        assert(a != b && a < N0&& b < N0);
        uint64_t e = e2e(a, b);
        auto i = edges.find(e);
        if (i == edges.end()) {
            edges[e] = structure.X.size();
            if (isFixed[a] && isFixed[b])
                structure.fixed.push_back(structure.X.size());
            structure.X.push_back(0.5 * (structure.X[a] + structure.X[b]));
        }
    };
    for (ivec4 e : E_) {
        addEdge(e.x, e.y); addEdge(e.x, e.z); addEdge(e.x, e.w);
        addEdge(e.y, e.z); addEdge(e.y, e.w); addEdge(e.z, e.w);
    }
    printf("%d edge nodes for quadratic elements.\n", edges.size());
    // elements
    for (ivec4 e : E_) {
        structure.SE.push_back(new QuadraticTetrahedralElement({
            e.x, e.y, e.z, e.w,
            edges[e2e(e.x, e.y)], edges[e2e(e.x, e.z)], edges[e2e(e.x, e.w)],
            edges[e2e(e.y, e.z)], edges[e2e(e.y, e.w)], edges[e2e(e.z, e.w)]
            }, &structure.X[0]));
    }
    // forces
    structure.F.resize(structure.X.size(), vec3(0));
    auto addEForce = [&](int a, int b, vec3 f) {
        int i = edges[e2e(a, b)];
        assert(i >= N0 && i < (int)structure.F.size());
        structure.F[i] += f;
    };
    for (ElementForce3 ef : Fs) {
        vec3 fv = ef.F / 12.0, fe = ef.F / 4.0;
        for (int i = 0; i < 3; i++)
            structure.F[((int*)&ef)[i]] += fv;
        addEForce(ef.x, ef.y, fe), addEForce(ef.x, ef.z, fe), addEForce(ef.y, ef.z, fe);
    }
    for (ElementForce4 ef : Fv) {
        vec3 fv = ef.F / 32.0, fe = ef.F * (7.0 / 48.0);
        int* e = (int*)&ef;
        for (int i = 0; i < 4; i++) {
            structure.F[e[i]] += fv;
            for (int j = 0; j < i; j++)
                addEForce(e[i], e[j], fe);
        }
    }
    // solve
    structure.solveDeflection(C);
    return structure;
}


// solve a structure discretized into bricks
// X: list of vertex initial positions
// E: list of elements
// Fs: list of surface tractions
// Fv: list of volume tractions
// fixed: list of fixed vertices
// C: strain to stress matrix
// order: the order of elements, must be 1 or 2
// returns: a list of deflections
DiscretizedStructure solveStructureBrick(
    std::vector<vec3> X,
    std::vector<ivec8> E,
    std::vector<ElementForce4> Fs,
    std::vector<ElementForce8> Fv,
    std::vector<int> fixed,
    const double* C,
    int order
) {
    assert(order == 1 || order == 2);
    DiscretizedStructure structure;
    structure.X = X;
    structure.F.resize(X.size(), vec3(0));
    structure.fixed = fixed;

    // linear brick
    if (order == 1) {
        for (ivec8 e : E)
            structure.SE.push_back(new LinearBrickElement((int*)&e, &X[0]));
        for (ElementForce4 ef : Fs) {
            // not mathematically analyzed but hope this works
            vec3 x[4] = { X[ef.x], X[ef.y], X[ef.z], X[ef.w] };
            vec3 c = 0.25 * (x[0] + x[1] + x[2] + x[3]);
            const static double WU[4][4] = { {-0.75,0.75,0.25,-0.25}, {-0.75,0.75,0.25,-0.25}, {-0.25,0.25,0.75,-0.75}, {-0.25,0.25,0.75,-0.75} };
            const static double WV[4][4] = { {-0.75,-0.25,0.25,0.75}, {-0.25,-0.75,0.75,0.25}, {-0.25,-0.75,0.75,0.25}, {-0.75,-0.25,0.25,0.75} };
            double dA[4], totA = 0.0;
            for (int i = 0; i < 4; i++) {
                vec3 dxdu(0), dxdv(0);
                for (int j = 0; j < 4; j++)
                    dxdu += WU[i][j] * x[j], dxdv += WV[i][j] * x[j];
                dA[i] = length(cross(dxdu, dxdv));
                totA += dA[i];
            }
            for (int i = 0; i < 4; i++)
                structure.F[((int*)&ef)[i]] += ef.F * dA[i] / totA;
        }
        for (ElementForce8 ef : Fv) {
            vec3 x[8] = { X[ef.a], X[ef.b], X[ef.c], X[ef.d], X[ef.e], X[ef.f], X[ef.g], X[ef.h] };
            vec3 c = 0.125 * (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7]);
            const static double WU[8][8] = { {-0.5625,0.5625,0.1875,-0.1875,-0.1875,0.1875,0.0625,-0.0625}, {-0.5625,0.5625,0.1875,-0.1875,-0.1875,0.1875,0.0625,-0.0625}, {-0.1875,0.1875,0.5625,-0.5625,-0.0625,0.0625,0.1875,-0.1875}, {-0.1875,0.1875,0.5625,-0.5625,-0.0625,0.0625,0.1875,-0.1875}, {-0.1875,0.1875,0.0625,-0.0625,-0.5625,0.5625,0.1875,-0.1875}, {-0.1875,0.1875,0.0625,-0.0625,-0.5625,0.5625,0.1875,-0.1875}, {-0.0625,0.0625,0.1875,-0.1875,-0.1875,0.1875,0.5625,-0.5625}, {-0.1875,0.1875,0.5625,-0.5625,-0.0625,0.0625,0.1875,-0.1875} };
            const static double WV[8][8] = { {-0.5625,-0.1875,0.1875,0.5625,-0.1875,-0.0625,0.0625,0.1875}, {-0.1875,-0.5625,0.5625,0.1875,-0.0625,-0.1875,0.1875,0.0625}, {-0.1875,-0.5625,0.5625,0.1875,-0.0625,-0.1875,0.1875,0.0625}, {-0.5625,-0.1875,0.1875,0.5625,-0.1875,-0.0625,0.0625,0.1875}, {-0.1875,-0.0625,0.0625,0.1875,-0.5625,-0.1875,0.1875,0.5625}, {-0.0625,-0.1875,0.1875,0.0625,-0.1875,-0.5625,0.5625,0.1875}, {-0.0625,-0.1875,0.1875,0.0625,-0.1875,-0.5625,0.5625,0.1875}, {-0.5625,-0.1875,0.1875,0.5625,-0.1875,-0.0625,0.0625,0.1875} };
            const static double WW[8][8] = { {-0.5625,-0.1875,-0.0625,-0.1875,0.5625,0.1875,0.0625,0.1875}, {-0.1875,-0.5625,-0.1875,-0.0625,0.1875,0.5625,0.1875,0.0625}, {-0.0625,-0.1875,-0.5625,-0.1875,0.0625,0.1875,0.5625,0.1875}, {-0.1875,-0.0625,-0.1875,-0.5625,0.1875,0.0625,0.1875,0.5625}, {-0.5625,-0.1875,-0.0625,-0.1875,0.5625,0.1875,0.0625,0.1875}, {-0.1875,-0.5625,-0.1875,-0.0625,0.1875,0.5625,0.1875,0.0625}, {-0.0625,-0.1875,-0.5625,-0.1875,0.0625,0.1875,0.5625,0.1875}, {-0.1875,-0.0625,-0.1875,-0.5625,0.1875,0.0625,0.1875,0.5625} };
            double dV[8], totV = 0.0;
            for (int i = 0; i < 8; i++) {
                vec3 dxdu(0), dxdv(0), dxdw(0);
                for (int j = 0; j < 8; j++)
                    dxdu += WU[i][j] * x[j], dxdv += WV[i][j] * x[j], dxdw += WW[i][j] * x[j];
                dV[i] = determinant(mat3(dxdu, dxdv, dxdw));
                assert(dV[i] > 0.0);
                totV += dV[i];
            }
            for (int i = 0; i < 8; i++)
                structure.F[((int*)&ef)[i]] += ef.F * dV[i] / totV;
        }
        structure.solveDeflection(C);
        return structure;
    }

    else {
        assert(false);
    }


    return structure;

}


void DiscretizedStructure::calcForceStress(const double* C) {
    for (int i = 0; i < N; i++) Sigma[i] = mat3(0);
    std::vector<double> SigmaW(N, 0.0);
    for (const SolidElement* e : SE)
        e->addStress(&X[0], C, &U[0], &Sigma[0], &SigmaW[0]);
    for (int i = 0; i < N; i++)
        Sigma[i] *= (1.0 / SigmaW[i]);
}

