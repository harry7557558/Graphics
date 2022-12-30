// solve the FEM problem

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
#define PI 3.1415926535897932384626
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


// solve a structure
// X: a list of vertices
// SE: a list of elements
// F: a list of forces on vertices
// fixed: indices of fixed vertices
// C: 6x6 matrix transforming strain to stress
// returns: a list of deflections
std::vector<vec3> solveStructure(
    std::vector<vec3> X,
    std::vector<const SolidElement*> SE,
    std::vector<vec3> F,
    std::vector<int> fixed,
    const double* C
) {
    assert(X.size() == F.size());
    int N = (int)X.size(), M = (int)SE.size();
    printf("%d vertices, %d elements.\n", N, M);
    double time0 = getTimePast();

    // map vertex indices to indices in the linear system
    // (don't consider fixed vertices)
    int* Imap = new int[N];
    for (int i = 0; i < N; i++) Imap[i] = 0;
    for (int i : fixed) assert(i >= 0 && i < N), Imap[i] = -1;
    int Ns = 0;
    for (int i = 0; i < N; i++) {
        if (Imap[i] != -1) Imap[i] = Ns++;
    }

    // construct the matrix
    double K[9 * MAX_SOLID_ELEMENT_N * MAX_SOLID_ELEMENT_N];
    LilMatrix lil(3 * Ns);
    mat3* invDiag = new mat3[Ns];
    for (int i = 0; i < Ns; i++) invDiag[i] = mat3(0);
    for (const SolidElement* c : SE) {
        int n = c->getN();
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
                if (i == j) invDiag[i] += m;
            }
        }
    }
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
#if 0  // w/o preconditioning
    double time2 = time1;
    int niters = conjugateGradient(
        3 * Ns, linopr, (double*)f, (double*)u, 10000, tol);
#else  // w/o preconditioning
#if 0  // preconditioner
    auto precond = [&](const double* src, double* res) {
        for (int i = 0; i < Ns; i++)
            ((vec3*)res)[i] = invDiag[i] * ((vec3*)src)[i];
    };
#else  // preconditioner
    LilMatrix iclil = lil.incompleteCholesky1();
    CsrMatrix precondL(iclil), precondU(iclil.transpose());
    auto precond = [&](const double* src, double* res) {
        memcpy(res, src, sizeof(double) * 3 * Ns);
        precondL.lowerSolve(res);
        precondU.upperSolve(res);
    };
#endif  // preconditioner
    double time2 = getTimePast();
    printf("Linear system preconditioned in %.2lf secs.\n", time2 - time1);
    int niters = conjugateGradientPreconditioned(
        3 * Ns, linopr, precond, (double*)f, (double*)u, 10000, tol);
#endif  // w/o preconditioning
    printf("%d iterations.\n", niters);
    double time3 = getTimePast();
    printf("Linear system solved in %.2lf secs. (includes preconditioning)\n", time3 - time1);

    // get the result
    std::vector<vec3> U(N, vec3(0));
    for (int i = 0; i < N; i++)
        if (Imap[i] != -1)
            U[i] = u[Imap[i]];
    delete[] Imap; delete[] f; delete[] u;
    delete[] invDiag;
    return U;
}
