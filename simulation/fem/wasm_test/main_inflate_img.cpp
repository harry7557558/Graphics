// generate 3D from images

#pragma GCC optimize "O2"
#define SUPPRESS_ASSERT 1

#include <cstdio>
#include "solver.h"
#include "meshgen_trig_implicit.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "write_model.h"

int main(int argc, char* argv[]) {
    const char filename[] = "/home/harry7557558/inflate-frosh/frosh-front-cropped.png";
    const float threshold = 0.5;
    const float grid0 = 0.01;
    const int depth = 4;

    int X, Y;
    uint8_t *pixels = (uint8_t*)stbi_load(filename, &X, &Y, nullptr, 4);

    vec2 scale = vec2(X, Y) / (float)fmax(X, Y);
    int gx = (int)(scale.x / grid0 + 1.0f);
    int gy = (int)(scale.y / grid0 + 1.0f);

    printf("Image: %d x %d\n", X, Y);
    printf("Initial mesh: %d x %d\n", gx, gy);

    auto getPixel = [&](int x, int y) -> float {
        x = clamp(x, 0, X - 1);
        y = clamp(y, 0, Y - 1);
        int idx = y * X + x;
        return pixels[4 * idx + 3] / 255.0f;
    };

    MeshgenTrigImplicit::ScalarFieldF F = [&](float x, float y) -> float {
        x /= scale.x, y /= scale.y;
        x = fmax(x, 0.0f) * (X - 1);
        y = fmax(1.0f-y, 0.0f) * (Y - 1);
        int xi = (int)x; float xf = x - xi;
        int yi = (int)y; float yf = y - yi;
        float v00 = getPixel(xi + 0, yi + 0);
        float v10 = getPixel(xi + 1, yi + 0);
        float v01 = getPixel(xi + 0, yi + 1);
        float v11 = getPixel(xi + 1, yi + 1);
        // xf = xf*xf*(3.0f-2.0f*xf);
        // yf = yf*yf*(3.0f-2.0f*yf);
        float v = mix(mix(v00, v10, xf), mix(v01, v11, xf), yf);
        return (1.0f-threshold) - v;
    };
    MeshgenTrigImplicit::ScalarFieldFBatch Fs = [&](size_t n, const vec2 *p, float *v) {
        for (int i = 0; i < n; i++)
            v[i] = F(p[i].x, p[i].y);
    };

    vec2 bc = 0.5f * scale, br = 0.5f * scale;
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
        Fs, bc-br, bc+br,
        ivec2(gx, gy), depth,
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
        res.U[i] = 1.0f*sqrt(fmax(res.U[i], 0.0f));
    float maxu = 0.0; for (int i = 0; i < res.N; i++) maxu = fmax(maxu, res.U[i]);
    printf("height: %f\n", maxu);

    float t3 = getTimePast();
    printf("Total: %.2g secs.\n", t3-t0);

    std::vector<vec3> verts;
    int vn = (int)vs.size();
    for (int i = 0; i < vn; i++)
        verts.push_back(vec3(res.X[i], res.U[i]));
    std::vector<int> vmap(vn, -1);
    int nbcount = 0;
    for (int i = 0; i < vn; i++)
        if (!res.isBoundary[i]) {
            vmap[i] = vn + (nbcount++);
            verts.push_back(verts[i]*vec3(1,1,-1));
        }
        else vmap[i] = i;
    int tn = (int)trigs.size();
    for (int i = 0; i < tn; i++) {
        ivec3 t = trigs[i];
        for (int _ = 0; _ < 3; _++)
            t[_] = vmap[t[_]];
        std::swap(t.y, t.z);
        trigs.push_back(t);
    }
    writeGLB("model.glb", verts, trigs);

    free(pixels);
    return 0;
}