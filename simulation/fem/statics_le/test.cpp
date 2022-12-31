#pragma GCC optimize "Ofast"

#include <cstdio>
#include <random>

#include "solver.h"


// a single tetrahedron
void test_1() {
    std::vector<vec3> X({
        vec3(-100, 0, 0),
        vec3(100, 0, 0),
        vec3(0, 0, -100),
        vec3(0, 2000, 0)
        });
    std::vector<const SolidElement*> SE({
        new LinearTetrahedralElement({0, 1, 2, 3}, &X[0])
        });
    std::vector<vec3> F({
        vec3(0),
        vec3(0),
        vec3(0),
        vec3(0, 0, -100e3)
        });
    std::vector<int> fixed({ 0, 1, 2 });
    double C[36]; calculateStressStrainMatrix(200e3, 0.33, C);
    std::vector<vec3> U = solveStructure(X, SE, F, fixed, C);
    for (int i = 0; i < 4; i++)
        printf("%lg %lg %lg\n", U[i].x, U[i].y, U[i].z);
}


// a cantilevel beam for comparing with hand calculation
// length: 6000mm
// applied load: 200e3 N at the end
// bending moment: 1200e6 Nmm, "triangular" distribution
// cross-section: 300mm, I = 675e6 mm^4, E = 200e3 MPa, phi[max] = 8.89e-6 mm^-1
// slope: 0.0267; deflection: -106.7mm z, 4.00mm x
void test_2() {
    // int N = 40, S = 1;
    // int N = 80, S = 2;
    int N = 120, S = 3;
    // int N = 160, S = 4;
    // int N = 240, S = 6;
    // int N = 480, S = 12;
    /*
    * Linear tets - seems like the error is O(h^2)
    size 150mm, deflection (-70.4mm z, 2.65mm x), z diff by 36.3mm
    size 75mm, deflection (-96.0mm z, 3.61mm x), z diff by 10.7mm
    size 50mm, deflection (-102.1mm z, 3.83mm x), z diff by 4.6mm
    size 37.5mm, deflection (-104.2mm z, 3.91mm x), z diff by 2.5mm
    size 25mm, deflection (-105.6mm z, 3.96mm x), z diff by 1.1mm
    * Linear bricks - O(h^2) error with some bias?
    size 150mm, deflection (-92.6mm z, 3.48mm x), z diff by 14.1mm
    size 75mm, deflection (-102.5mm z, 3.84mm x), z diff by 4.2mm
    size 50mm, deflection (-104.6mm z, 3.92mm x), z diff by 2.1mm
    size 37.5mm, deflection (-105.4mm z, 3.95mm x), z diff by 1.3mm
    size 25mm, deflection (-106.0mm z, 3.97mm x), z diff by 0.7mm
    * Quadratic tets - "much better" but doesn't converge to hand calculation?
    size 150mm, deflection (-106.1mm z, 3.98mm x), z diff by 0.6mm
    size 75mm, deflection (-106.3mm z, 3.99mm x), z diff by 0.4mm
    size 50mm, deflection (-106.4mm z, 3.99mm x), z diff by 0.3mm
    */
    int si = (2 * S + 1) * (2 * S + 1), sj = (2 * S + 1), sk = 1;
    printf("Test 2 (cantilevel beam) started.\n");
    // vertices
    std::vector<vec3> X;
    for (int i = 0; i <= N; i++)
        for (int j = -S; j <= S; j++)
            for (int k = -S; k <= S; k++)
                X.push_back(vec3(
                    (6000. / N) * i,
                    (150. / S) * j, (150. / S) * k
                ));
    // elements
    std::vector<ivec4> SE4;
    std::vector<ivec8> SE8;
    for (int i = 0; i < N; i++)
        for (int j = -S; j < S; j++)
            for (int k = -S; k < S; k++) {
                int s = i * si + (j + S) * sj + (k + S) * sk;
                // tetrahedra
                ivec3 G[5][4] = {
                    {ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(0, 0, 1)},
                    {ivec3(1, 1, 0), ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(1, 1, 1)},
                    {ivec3(0, 1, 1), ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(0, 1, 0)},
                    {ivec3(1, 0, 1), ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(1, 0, 0)},
                    {ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(1, 0, 0), ivec3(0, 1, 0)},
                };
                for (int a = 0; a < 5; a++) {
                    int v[4];
                    for (int b = 0; b < 4; b++) {
                        if (i & 1) G[a][b].x = 1 - G[a][b].x;
                        if (j & 1) G[a][b].y = 1 - G[a][b].y;
                        if (k & 1) G[a][b].z = 1 - G[a][b].z;
                        v[b] = s + G[a][b].x * si + G[a][b].y * sj + G[a][b].z * sk;
                    }
                    SE4.push_back(ivec4(v));
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
            }
    // if this changes the result for tets, then there is a bug
    auto prng = std::default_random_engine(0);
    for (int i = 0; i < (int)SE4.size(); i++) {
        int* v = (int*)&SE4[i];
        std::shuffle(v, v + 4, prng);
    }
    // forces
    std::vector<ElementForce3> F4s;
    std::vector<ElementForce4> F4v;
    std::vector<ElementForce4> F8s;
    std::vector<ElementForce8> F8v;
    for (int j = 0; j < 2 * S; j++)
        for (int k = 0; k < 2 * S; k++) {
            vec3 perFace = vec3(0, 0, -200e3) / (4 * S * S);
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
    // std::vector<vec3> U = solveStructureTetrahedral(X, SE4, F4s, F4v, fixed, C, 2);
    std::vector<vec3> U = solveStructureBrick(X, SE8, F8s, F8v, fixed, C, 1);
    for (int i = N * si; i < (int)U.size(); i++)
        printf("%lg %lg %lg\n", U[i].x, U[i].y, U[i].z);
}


int main() {
    double t0 = getTimePast();
    test_2();
    double t1 = getTimePast();
    printf("Total %.2lf secs.\n", t1 - t0);
    return 0;
}
