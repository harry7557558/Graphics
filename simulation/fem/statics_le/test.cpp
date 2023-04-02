// Windows:
// ccache g++ test.cpp -lopengl32 -lglew32 -lglfw3 -o test ; if ($?) { .\test }

// ECF (school computers kinda sus):
// g++ -std=c++17 -I$HOME/.local/include -L$HOME/.local/lib test.cpp -lOpenGL -lGLEW -lglfw3 -ldl -lX11 -lpthread -o ~/temp
// LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH ~/temp

#pragma GCC optimize "O3"

#define SUPPRESS_ASSERT 0

#include <cstdio>
#include <random>

#include "solver.h"
#include "render.h"

#include "meshgen_tet_implicit.h"

// a single tetrahedron
void test_1() {
    // no longer compactible - deprecated
}


// a cantilevel beam for comparing with hand calculation
// @density: weight density in N/mm³
// @load: load at the end of the beam in N
// length: 6000mm
// cross-section: 300mm, I = 675e6 mm^4, Q/Ib = 16.67e-6 mm^-2, E = 200e3 MPa
// (density, load) = (0, 200e3): uz -106.7mm, ux 4mm, flexural 267MPa, shear 3.33MPa
// (density, load) = (100e-6, 0): uz -10.8mm, ux 0.36mm, flexural 36MPa, shear 0.9MPa
// principle of superposition holds
// (density, load) = (100e-6, 36e3): uz 30mm, ux 1.08mm, flexural 84MPa, shear 1.5MPa
DiscretizedStructure test_2(double density, double load) {
    int N = 40, S = 1;
    // int N = 80, S = 2;
    // int N = 120, S = 3;
    // int N = 160, S = 4;
    // int N = 240, S = 6;
    // int N = 480, S = 12;
    /* (density, load) = (0, 200e3), no deformation
    * Linear tets - seems like the error is O(h^2)
    size 150mm, deflection (-72.6 z, 2.72mm x), z diff by 34.1mm
    size 75mm, deflection (-95.2mm z, 3.57mm x), z diff by 11.5mm
    size 50mm, deflection (-101.1mm z, 3.79mm x), z diff by 5.6mm
    size 37.5mm, deflection (-103.3mm z, 3.87mm x), z diff by 3.4mm
    size 25mm, deflection (-105.0mm z, 3.94mm x), z diff by 1.7mm
    * Linear bricks - O(h^2) error with some bias?
    size 150mm, deflection (-92.6mm z, 3.48mm x), z diff by 14.1mm
    size 75mm, deflection (-102.5mm z, 3.84mm x), z diff by 4.2mm
    size 50mm, deflection (-104.6mm z, 3.92mm x), z diff by 2.1mm
    size 37.5mm, deflection (-105.4mm z, 3.95mm x), z diff by 1.3mm
    size 25mm, deflection (-106.0mm z, 3.97mm x), z diff by 0.7mm
    * Quadratic tets - "much better" but doesn't converge to hand calculation?
    size 150mm, deflection (-106.16mm z, 3.983mm x), z diff by 0.51mm
    size 75mm, deflection (-106.35mm z, 3.988mm x), z diff by 0.32mm
    size 50mm, deflection (-106.39mm z, 3.989mm x), z diff by 0.28mm
    */
    /* (density, load) = (100e-6, 36e3), with deformation
    * Linear tets -
    size (80, 2), deflection (26.9, 19.9, -82.2)
    size (120, 3), deflection (33.8, 23.5, -102.0)
    size (160, 4), deflection (39.8, 24.7, -117.0)
    size (240, 6), deflection (50.6, 23.5, -140.3)
    * Quadratic tets -
    size (40, 1), deflection (111.1, -21.6, -230.8)
    size (80, 2), deflection (124.6, -30.3, -250.0)
    * Linear bricks -
    size (40, 1), deflection (20.7, 18.4, -65.8)
    size (80, 2), deflection (29.8, 23.8, -92.2)
    size (120, 3), deflection (37.5, 25.3, -111.5)
    size (160, 4), deflection (45.0, 24.5, -127.5)
    size (240, 6), deflection (58.9, 19.3, -153.5)
    size (480, 12), deflection (89.1, -0.5, -200.9)
    */
    int si = (2 * S + 1) * (2 * S + 1), sj = (2 * S + 1), sk = 1;
    printf("Test 2 (cantilevel beam) started.\n");
    // vertices
    std::vector<vec3> X;
    for (int i = 0; i <= N; i++)
        for (int j = -S; j <= S; j++)
            for (int k = -S; k <= S; k++) {
                vec3 p(
                    (6000. / N) * i,
                    (150. / S) * j, (150. / S) * k
                );
                if (1) {  // deformation
                    float t = 0.5 + 0.5 * tanh(10.0 * (i / (double)N - 0.4));
                    p = vec3(p.x, p.y * (1.0 + 3.0 * t * t), p.z);
                    p += vec3(0, 2400, 2400) * t;
                }
                X.push_back(p);
            }
    // elements
    std::vector<ivec4> SE4;
    std::vector<ivec8> SE8;
    std::vector<ElementForce4> F4v;
    std::vector<ElementForce8> F8v;
    for (int i = 0; i < N; i++)
        for (int j = -S; j < S; j++)
            for (int k = -S; k < S; k++) {
                int s = i * si + (j + S) * sj + (k + S) * sk;
                // tetrahedra
                ivec3 G[5][4] = {
                    {ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(0, 0, 1)},
                    {ivec3(1, 1, 0), ivec3(0, 1, 0), ivec3(1, 0, 0), ivec3(1, 1, 1)},
                    {ivec3(0, 1, 1), ivec3(0, 0, 1), ivec3(0, 1, 0), ivec3(1, 1, 1)},
                    {ivec3(1, 0, 1), ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(1, 0, 0)},
                    {ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(1, 0, 0), ivec3(0, 1, 0)},
                };
                for (int a = 0; a < 5; a++) {
                    int v[4];
                    for (int b = 0; b < 4; b++) {
                        ivec3 g = G[a][b];
                        if (i & 1) g = ivec3(g.y, 1 - g.x, g.z);
                        if (j & 1) g = ivec3(1 - g.y, g.x, g.z);
                        if (k & 1) g = ivec3(g.x, 1 - g.z, g.y);
                        v[b] = s + g.x * si + g.y * sj + g.z * sk;
                    }
                    SE4.push_back(ivec4(v));
                    if (density != 0.0) {
                        double dV = abs(determinant(mat3(
                            X[v[1]] - X[v[0]], X[v[2]] - X[v[0]], X[v[3]] - X[v[0]]))) / 6.0;
                        F4v.push_back(ElementForce4{ v, vec3(0, 0, -density * dV) });
                    }
                }
                // bricks
                ivec3 H[8] = {
                    ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(1, 1, 0), ivec3(0, 1, 0),
                    ivec3(0, 0, 1), ivec3(1, 0, 1), ivec3(1, 1, 1), ivec3(0, 1, 1)
                };
                int v[8];
                for (int b = 0; b < 8; b++)
                    v[b] = s + H[b].x * si + H[b].y * sj + H[b].z * sk;
                SE8.push_back(ivec8(v));
                if (density != 0.0) {
                    double dV = abs(determinant(mat3(
                        X[v[1]] - X[v[0]], X[v[3]] - X[v[0]], X[v[4]] - X[v[0]])));
                    F8v.push_back(ElementForce8{ v, vec3(0, 0, -density * dV) });
                }
            }
    // surface tractions
    std::vector<ElementForce3> F4s;
    std::vector<ElementForce4> F8s;
    for (int j = 0; j < 2 * S; j++)
        for (int k = 0; k < 2 * S; k++) {
            vec3 perFace = vec3(0, 0, -load) / (4 * S * S);
            if ((N + j + k) & 1) {
                F4s.push_back(ElementForce3({ {
                        N * si + j * sj + k * sk,
                        N * si + (j + 1) * sj + (k + 1) * sk,
                        N * si + j * sj + (k + 1) * sk,
                    }, perFace / 2.0 }));
                F4s.push_back(ElementForce3({ {
                        N * si + j * sj + k * sk,
                        N * si + (j + 1) * sj + k * sk,
                        N * si + (j + 1) * sj + (k + 1) * sk,
                    }, perFace / 2.0 }));
            }
            else {
                F4s.push_back(ElementForce3({ {
                        N * si + j * sj + k * sk,
                        N * si + (j + 1) * sj + k * sk,
                        N * si + j * sj + (k + 1) * sk,
                    }, perFace / 2.0 }));
                F4s.push_back(ElementForce3({ {
                        N * si + (j + 1) * sj + k * sk,
                        N * si + (j + 1) * sj + (k + 1) * sk,
                        N * si + j * sj + (k + 1) * sk,
                    }, perFace / 2.0 }));
            }
            F8s.push_back(ElementForce4({ {
                    N * si + j * sj + k * sk,
                    N * si + (j + 1) * sj + k * sk,
                    N * si + (j + 1) * sj + (k + 1) * sk,
                    N * si + j * sj + (k + 1) * sk,
                }, perFace }));
        }
    // fixed
    std::vector<int> fixed;
    for (int i = 0; i < si; i++) fixed.push_back(i);
    // solve
    double C[36]; calculateStressStrainMatrix(200e3, 0.33, C);
    DiscretizedStructure structure = solveStructureTetrahedral(X, SE4, F4s, F4v, fixed, C, 1);
    // DiscretizedStructure structure = solveStructureBrick(X, SE8, F8s, F8v, fixed, C, 1);
    structure.calcForceStress(C);
    if (true) {  // check
        for (int i = N * si; i < X.size(); i++) {
            vec3 u = structure.U[i];
            printf("%lg %lg %lg\n", u.x, u.y, u.z);
        }
        double tensile = 0.0, compressive = 0.0, shear = 0.0;
        for (mat3 sigma : structure.Sigma) {
            tensile = max(tensile, sigma[0][0]);
            compressive = min(compressive, sigma[0][0]);
            shear = max(shear, sigma[0][2]);
        }
        printf("%lf %lf %lf\n", tensile, compressive, shear);
    }
    return structure;
}


DiscretizedStructure test_3(double density) {
    MeshgenTetImplicit::ScalarFieldF F = [](double x, double y, double z) {
        vec3 p(x, y, z);
        // return dot(p, p) - 1.0;
        // return dot(p, p) - 6.0;
        // return hypot(x, sqrt(y * y + z * z + 1.99 * sin(y * z)) - 1.) - 0.5;
        // return 2. * dot(p * p, p * p) - 3. * dot(p, p) + 2.;
        // return x * x + y * y - (1. - z) * z * z - 0.1;
        // return max(p.x * p.x + p.y * p.y - 1.0, abs(p.z) - 0.5);
        // return pow(p.x * p.x + 2. * p.y * p.y + p.z * p.z - 1., 3.) - (p.x * p.x + .1 * p.y * p.y) * pow(p.z, 3.);
        return sin(2.0*p.x) + sin(2.0*p.y) + sin(2.0*p.z);
        // return cos(10.0 * (sin(x) * sin(y) + sin(z)));
        // return  sin(12345.67 * sin(12.34 * sin(x) + 56.78 * sin(y) + 90.12 * sin(z) + 34.56) + 89.0);
    };
    MeshgenTetImplicit::ScalarFieldFBatch Fs = [&](int n, const vec3 *p, double *v) {
        for (int i = 0; i < n; i++)
            v[i] = F(p[i].x, p[i].y, p[i].z);
    };
    vec3 bc = vec3(0), br = vec3(2);
    auto constraint = [=](vec3 p) {
        p -= bc;
        return -vec3(
            sign(p.x) * max(abs(p.x) - br.x, 0.0),
            sign(p.y) * max(abs(p.y) - br.y, 0.0),
            sign(p.z) * max(abs(p.z) - br.z, 0.0)
        );
    };

#if 0
    std::vector<MeshgenTetImplicit::MeshVertex> vertsM;
    std::vector<ivec4> tetsM;
    F = MeshgenTetImplicit::generateInitialTetrahedraInBox(
        F, bc, br, 0.25, vertsM, tetsM);
    std::vector<vec3> vs;
    std::vector<ivec4> tets;
    if (1) {
        std::vector<vec3> vertsN;
        std::vector<ivec4> tetsN;
        MeshgenTetImplicit::cutIsosurface(F, vertsM, tetsM, vertsN, tetsN);
        if (0) MeshgenTetImplicit::mergeCloseSurfaceVertices(vertsN, tetsN, vs, tets);
        else vs = vertsN, tets = tetsN;
    }
    else {
        for (auto mv : vertsM) vs.push_back(mv.x);
        tets = tetsM;
    }
    MeshgenTetImplicit::assertVolumeEqual(vs, tets);
    MeshgenTetImplicit::smoothMesh(vs, tets, 50, Fs);
    MeshgenTetImplicit::assertVolumeEqual(vs, tets);
#else
    std::vector<vec3> vs;
    std::vector<ivec4> tets;
    std::vector<int> constraintI;
    std::vector<vec3> constraintN;
    MeshgenTetImplicit::generateTetrahedraBCC(
        F, bc-br, bc+br, ivec3(12),
        vs, tets,
        constraintI, constraintN
    );
    MeshgenTetImplicit::assertVolumeEqual(vs, tets);
    MeshgenTetImplicit::smoothMesh(
        vs, tets, 8, Fs,
        constraint, constraintI, constraintN);
    MeshgenTetImplicit::assertVolumeEqual(vs, tets);
#endif

    auto vec3Cmp = [](vec3 a, vec3 b) {
        return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
    };

    std::vector<ElementForce4> Fv;
    for (ivec4 t : tets) {
        double dV = abs(determinant(mat3(
            vs[t.y] - vs[t.x], vs[t.z] - vs[t.x], vs[t.w] - vs[t.x]))) / 6.0;
        Fv.push_back(ElementForce4{ t, vec3(0, 0, -density * dV) });
    }

    double C[36]; calculateStressStrainMatrix(200e3, 0.33, C);
    DiscretizedStructure structure = solveStructureTetrahedral(
        vs, tets,
        std::vector<ElementForce3>(), Fv,
        std::vector<int>({ 0, 1, 2, 3 }), C, 1);
    // structure.calcForceStress(C);
    return structure;
}


int main() {
    double t0 = getTimePast();
    // DiscretizedStructure structure = test_2(0.0, 200e3);
    // DiscretizedStructure structure = test_2(100e-6, 0.0);
    // DiscretizedStructure structure = test_2(100e-6, 36e3);
    DiscretizedStructure structure = test_3(1.0);
    double t1 = getTimePast();
    printf("Total %.2lf secs.\n", t1 - t0);
    mainGUI(structure, true);
    return 0;
}
