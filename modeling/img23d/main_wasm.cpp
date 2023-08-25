#pragma GCC optimize "O3"

#define SUPPRESS_ASSERT 1

#include <emscripten/emscripten.h>

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#error "Please compile this as a C++ file."
#endif


#include <cstdio>
#include <random>

#include "solver.h"
#include "render.h"

#include "meshgen_trig_implicit.h"
#include "test_functions.h"


namespace Image {

uint8_t *data = nullptr;
int w = 0, h = 0;

float getPixel(int x, int y) {
    x = clamp(x, 0, w-1);
    y = clamp(y, 0, h-1);
    int idx = y * w + x;
    return (float)data[4*idx+3] / 255.0f;
}



}

DiscretizedModel<float, float> test_3(float density) {

    MeshgenTrigImplicit::ScalarFieldF F = [](float x, float y) -> float {
        vec2 p(x, y);
        // return sin(100.0*x*y+x*x+abs(y)+12.34*x+56.78*y);
        // return x*x+y*y-1.0;
        // return x*x+y*y-abs(x)*y-1.0;
        // return abs(hypot(x,y)-1)-0.5;
        return sin(2.0*x)*cos(2.0*y)-0.2;
        // return 4*x*x+pow(2*y-cos(2*x),2)+cos(12*x)*exp(y-4*x*x)-2;
        // return pow(x,4) + pow(y,4) - 4.0*x*x*y*y;
        // return sin(10.0*(y-sin(x)));
        // return sin(9*hypot(x,y));
        // return 1.1*(fabs(x)+fabs(y))-fmax(fabs(x),fabs(y))-0.1;
        // return sin(10.0*(x*x+y*y-abs(x)*y));
        // return abs(pow(x*x+y*y-1,3) - x*x*y*y*y)-0.1;
        // return sin(6*atan2(y,x))-4*x*y;
        // return sin(6*x)+sin(6*y)-(sin(12*x)+cos(6*y))*sin(12*y);
        // return fmin(cos(10*x-cos(5*y)),cos(10*y+cos(5*x)))+0.5;
        // return -funMandelbrotSet(x, y);
    };
    MeshgenTrigImplicit::ScalarFieldFBatch Fs = [&](size_t n, const vec2 *p, float *v) {
        for (int i = 0; i < n; i++)
            v[i] = F(2.0f*p[i].x, 2.0f*p[i].y);
    };
    vec2 bc = vec2(0), br = vec2(1);
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
    std::vector<bool> isConstrained[2];
    // MeshgenTrigImplicit::generateInitialMeshOld(
    //     Fs, bc-br, bc+br,
    //     ivec2(67, 63),
    //     // ivec2(33, 31),
    //     // ivec2(19, 17),4f
    //     // ivec2(67, 17),
    //     vs, trigs,
    //     constraintI, constraintN
    // );
    MeshgenTrigImplicit::generateInitialMesh(
        Fs, bc-br, bc+br,
        // ivec2(2, 2), 2,
        // ivec2(9, 7), 2,
        ivec2(19, 17), 5,
        // ivec2(33, 31), 4,
        // ivec2(65, 63), 3,
        vs, trigs, isConstrained
    );
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t1 = getTimePast();
    printf("Mesh generated in %.2g secs.\n", t1-t0);
    int vn0 = (int)vs.size();
    MeshgenTrigImplicit::splitStickyVertices(vs, trigs, isConstrained);
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t2 = getTimePast();
    printf("Mesh cleaned in %.2g secs.\n", t2-t1);
    MeshgenTrigImplicit::smoothMesh(
        vs, trigs, 5, Fs,
        constraint, isConstrained);
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t3 = getTimePast();
    printf("Mesh optimized in %.2g secs.\n", t3-t2);

    DiscretizedModel<float, float> res = solveLaplacianLinearTrig(
        vs, std::vector<float>(vs.size(), 4.0f), trigs);
    for (int i = 0; i < res.N; i++)
        res.U[i] = 1.0f*sqrt(fmax(res.U[i], 0.0f));
        // res.U[i] = 1.0f*res.U[i];
        // res.U[i] = i < vn0 ? 0.0 : 0.5;
    float maxu = 0.0; for (int i = 0; i < res.N; i++) maxu = fmax(maxu, res.U[i]);
    printf("height: %f\n", maxu);
    return res;
}



DiscretizedModel<float, float> imageTo3D() {
    using namespace Image;

    float threshold = 0.5;
    float grid0 = 0.01;
    int depth = 2;

    vec2 scale = vec2(w, h) / (float)fmax(w, h);
    int gx = (int)(scale.x / grid0 + 1.0f);
    int gy = (int)(scale.y / grid0 + 1.0f);

    MeshgenTrigImplicit::ScalarFieldF F = [&](float x, float y) -> float {
        x = 0.5f + 0.5f * x / scale.x;
        y = 0.5f + 0.5f * y / scale.y;
        x = fmax(x, 0.0f) * (w - 1);
        y = fmax(1.0f-y, 0.0f) * (h - 1);
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

    vec2 bc = vec2(0), br = scale;
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
    std::vector<bool> isConstrained[2];
    MeshgenTrigImplicit::generateInitialMesh(
        Fs, bc-br, bc+br,
        ivec2(gx, gy), depth,
        vs, trigs, isConstrained
    );
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t1 = getTimePast();
    printf("Mesh generated in %.2g secs.\n", t1-t0);
    MeshgenTrigImplicit::splitStickyVertices(vs, trigs, isConstrained);
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t2 = getTimePast();
    printf("Mesh cleaned in %.2g secs.\n", t2-t1);

    MeshgenTrigImplicit::smoothMesh(
        vs, trigs, 5, Fs,
        constraint, isConstrained);
    MeshgenTrigImplicit::assertAreaEqual(vs, trigs);
    float t3 = getTimePast();
    printf("Mesh optimized in %.2g secs.\n", t3-t2);

    DiscretizedModel<float, float> res = solveLaplacianLinearTrig(
        vs, std::vector<float>(vs.size(), 4.0f), trigs);
    for (int i = 0; i < res.N; i++)
        res.U[i] = 1.0f*sqrt(fmax(res.U[i], 0.0f));
    float maxu = 0.0; for (int i = 0; i < res.N; i++) maxu = fmax(maxu, res.U[i]);
    printf("height: %f\n", maxu);

    return res;
}



void prepareMesh(DiscretizedModel<float, float> model) {
    renderModel.vertices.resize(model.X.size());
    renderModel.normals = std::vector<vec3>(model.X.size(), vec3(0));
    for (int i = 0; i < model.X.size(); i++)
        renderModel.vertices[i] = vec3(
            model.X[i], model.U[i]);

    renderModel.indicesF.resize(model.SE.size());
    for (int i = 0; i < (int)model.SE.size(); i++) {
        ivec3 t;
        model.SE[i]->getTriangles(&t);
        renderModel.indicesF[i] = t;
        vec3 n = cross(
            renderModel.vertices[t[1]]-renderModel.vertices[t[0]],
            renderModel.vertices[t[2]]-renderModel.vertices[t[0]]);
        n = normalize(n);
        for (int _ = 0; _ < 3; _++)
            renderModel.normals[t[_]] += n;
    }
    for (int i = 0; i < (int)renderModel.normals.size(); i++)
        renderModel.normals[i] = normalize(renderModel.normals[i]);
}


void mainGUICallback() {
    // inside requestAnimationFrame
}

int main() {
    initWindow();
    emscripten_run_script("onReady()");

    float t0 = getTimePast();
    DiscretizedModel<float, float> structure = test_3(1.0);
    float t1 = getTimePast();
    printf("Total %.2g secs.\n", t1 - t0);

    prepareMesh(structure);
    mainGUI(mainGUICallback);
    return 0;
}




EXTERN EMSCRIPTEN_KEEPALIVE
void resizeWindow(int w, int h) {
    RenderParams::iResolution = glm::ivec2(w, h);
    RenderParams::viewport->renderNeeded = true;
}


EXTERN EMSCRIPTEN_KEEPALIVE
void updateImage(int w, int h, uint8_t *data) {
    printf("%d %d\n", w, h);
    Image::w = w;
    Image::h = h;
    if (Image::data)
        delete Image::data;
    Image::data = data;

    prepareMesh(imageTo3D());
    RenderParams::viewport->renderNeeded = true;
}
