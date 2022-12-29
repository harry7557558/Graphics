#include <cstdio>
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
// slope: 0.0267; deflection: 106.7mm
void test_2() {
    // seems like the error is O(h^2)
    // int N = 40, S = 1;  // size 150mm, deflection -70.5mm, diff by 36.2mm
    int N = 80, S = 2;  // size 75mm, deflection -96.0mm, diff by 10.7mm
    // int N = 120, S = 3;  // size 50mm, deflection -102.0mm, diff by 4.7mm
    // int N = 160, S = 4;  // size 37.5mm, deflection -104.2mm, diff by 2.5mm
    int si = (2 * S + 1) * (2 * S + 1), sj = (2 * S + 1), sk = 1;
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
    std::vector<const SolidElement*> SE;
    for (int i = 0; i < N; i++)
        for (int j = -S; j < S; j++)
            for (int k = -S; k < S; k++) {
                int s = i * si + (j + S) * sj + (k + S) * sk;
                ivec3 G[5][4] = {
                    {ivec3(0, 0, 0), ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(0, 0, 1)},
                    {ivec3(1, 1, 0), ivec3(1, 0, 0), ivec3(0, 1, 0), ivec3(1, 1, 1)},
                    {ivec3(0, 1, 1), ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(0, 1, 0)},
                    {ivec3(1, 0, 1), ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(1, 0, 0)},
                    {ivec3(0, 0, 1), ivec3(1, 1, 1), ivec3(1, 0, 0), ivec3(0, 1, 0)},
                };
                for (int a = 0; a < 5; a++) {
                    int v[4];
                    for (int b = 0; b < 4; b++)
                        v[b] = s + G[a][b].x * si + G[a][b].y * sj + G[a][b].z * sk;
                    SE.push_back(new LinearTetrahedralElement(v, &X[0]));
                }
            }
    // forces
    std::vector<vec3> F(N * si, vec3(0));
    for (int i = 0; i < si; i++) F.push_back(vec3(0, 0, -200e3) / si);
    // fixed
    std::vector<int> fixed;
    for (int i = 0; i < si; i++) fixed.push_back(i);
    // solve
    double C[36]; calculateStressStrainMatrix(200e3, 0.33, C);
    std::vector<vec3> U = solveStructure(X, SE, F, fixed, C);
    for (int i = N * si; i < (int)F.size(); i++)
        printf("%lg %lg %lg\n", U[i].x, U[i].y, U[i].z);
}


int main() {
    test_2();
    return 0;
}
