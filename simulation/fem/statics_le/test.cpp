#pragma GCC optimize "Ofast"

#include <cstdio>
#include <random>

#include "solver.h"
#include "render.h"

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
DiscretizedStructure test_2(double density, double load) {
    // int N = 40, S = 1;
    // int N = 80, S = 2;
    int N = 120, S = 3;
    // int N = 160, S = 4;
    // int N = 240, S = 6;
    // int N = 480, S = 12;
    /* (density, load) = (0, 200e3)
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
                float t = 0.5 + 0.5 * tanh(10.0 * (i / (double)N - 0.4));
                p = vec3(p.x, p.y * (1.0 + 3.0 * t * t), p.z);
                p += vec3(0, 2400, 2400) * t;
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
    // DiscretizedStructure structure = solveStructureTetrahedral(X, SE4, F4s, F4v, fixed, C, 1);
    DiscretizedStructure structure = solveStructureBrick(X, SE8, F8s, F8v, fixed, C, 1);
    if (true) {  // check
        for (int i = N * si; i < X.size(); i++) {
            vec3 u = structure.U[i];
            printf("%lg %lg %lg\n", u.x, u.y, u.z);
        }
        structure.calcForceStress(C);
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


int main() {
    double t0 = getTimePast();
    // DiscretizedStructure structure = test_2(0.0, 200e3);
    // DiscretizedStructure structure = test_2(100e-6, 0.0);
    DiscretizedStructure structure = test_2(100e-6, 36e3);
    double t1 = getTimePast();
    printf("Total %.2lf secs.\n", t1 - t0);
    mainRender(structure);
    return 0;
}
