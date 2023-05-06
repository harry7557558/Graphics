#pragma GCC optimize "O3"

#define SUPPRESS_ASSERT 0

#include <cstdio>
#include <random>

#include "solver.h"
#include "render.h"

#include "meshgen_trig_implicit.h"


DiscretizedModel<float, float> test_3(float density) {

    MeshgenTrigImplicit::ScalarFieldF F = [](float x, float y) {
        vec2 p(x, y);
        // return sin(100.0*x*y+x*x+abs(y)+12.34*x+56.78*y);
        // return x*x+y*y-1.0;
        // return x*x+y*y-abs(x)*y-1.0;
        // return abs(hypot(x,y)-1)-0.5;
        // return sin(2.0*x)*cos(2.0*y)-0.2;
        return 4*x*x+pow(2*y-cos(2*x),2)+cos(12*x)*exp(y-4*x*x)-2;
        // return pow(x,4) + pow(y,4) - 4.0*x*x*y*y;
        // return sin(10.0*(y-sin(x)));
    };
    MeshgenTrigImplicit::ScalarFieldFBatch Fs = [&](int n, const vec2 *p, float *v) {
        for (int i = 0; i < n; i++)
            v[i] = F(p[i].x, p[i].y);
    };
    vec2 bc = vec2(0), br = vec2(2);
    auto constraint = [=](vec2 p) {
        p -= bc;
        return -vec2(
            sign(p.x) * fmax(abs(p.x) - br.x, 0.0),
            sign(p.y) * fmax(abs(p.y) - br.y, 0.0)
        );
    };

    float t0 = getTimePast();
    std::vector<vec2> vs;
    std::vector<ivec3> trigs;
    std::vector<int> constraintI;
    std::vector<vec2> constraintN;
    MeshgenTrigImplicit::generateInitialMesh(
        F, bc-br, bc+br, ivec2(96),
        vs, trigs,
        constraintI, constraintN
    );
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t1 = getTimePast();
    printf("Mesh generated in %.2g secs.\n", t1-t0);
    MeshgenTrigImplicit::smoothMesh(
        vs, trigs, 5, Fs,
        constraint, constraintI, constraintN);
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t2 = getTimePast();
    printf("Mesh optimized in %.2g secs.\n", t2-t1);

    DiscretizedModel<float, float> res = solveLaplacianLinearTrig(
        vs, std::vector<float>(vs.size(), 4.0f), trigs);
    for (int i = 0; i < res.N; i++)
        res.U[i] = sqrt(fmax(res.U[i], 0.0f));
    return res;
}


int main() {
    float t0 = getTimePast();
    DiscretizedModel<float, float> structure = test_3(1.0);
    float t1 = getTimePast();
    printf("Total %.2g secs.\n", t1 - t0);
    mainGUI(structure, true);
    return 0;
}
