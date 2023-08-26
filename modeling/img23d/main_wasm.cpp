// WASM, compile with Emscripten

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

#include "write_model.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


namespace Img23d {

float threshold = 0.5;
float grid0 = 0.01;
int depth = 2;

std::string name;
bool containAlpha = true;
uint8_t *data = nullptr;
int w = 0, h = 0;

bool smoothShading = true;

float getPixel(int x, int y) {
    x = clamp(x, 0, w-1);
    y = clamp(y, 0, h-1);
    int idx = y * w + x;
    return (float)data[4*idx+3] / 255.0f;
}

void removeImageAlpha() {
    if (!containAlpha)
        return;
    uint8_t pth = (uint8_t)(threshold*255+0.5);
    for (int i = 0; i < w*h; i++)
        data[4*i+3] = (data[4*i+3]>pth ? 255 : 0);
    std::vector<ivec2> visited;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            if (data[4*(y*w+x)+3])
                visited.push_back(ivec2(x, y));
    while (!visited.empty()) {
        std::vector<ivec2> visited1;
        for (ivec2 ij : visited) {
            int i0 = ij.x, j0 = ij.y;
            for (int di = -1; di <= 1; di++)
                for (int dj = -1; dj <= 1; dj++) {
                    int i = i0+di, j = j0+dj;
                    if (i < 0 || j < 0 || i >= w || j >= h)
                        continue;
                    if (data[4*(j*w+i)+3])
                        continue;
                    visited1.push_back(ivec2(i, j));
                    ((uint32_t*)data)[j*w+i] = ((uint32_t*)data)[j0*w+i0];
                }
        }
        visited = visited1;
    }
    containAlpha = false;
}

DiscretizedModel<float, float> model;

}


DiscretizedModel<float, float> imageTo3D() {
    using namespace Img23d;

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



void prepareMesh(const DiscretizedModel<float, float> &model) {
    vec2 scale = vec2(Img23d::w, Img23d::h) / (float)fmax(Img23d::w, Img23d::h);

    renderModel.vertices.resize(model.X.size());
    renderModel.normals = std::vector<vec3>(model.X.size(), vec3(0));
    renderModel.texcoords.resize(model.X.size());
    for (int i = 0; i < model.X.size(); i++) {
        renderModel.vertices[i] = vec3(
            model.X[i], model.U[i]);
        renderModel.texcoords[i] = 0.5f + 0.5f * vec2(1,-1) * model.X[i] / scale;
    }

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

    Img23d::removeImageAlpha();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
        Img23d::w, Img23d::h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
        Img23d::data);
}


void mainGUICallback() {
    // inside requestAnimationFrame
}

int main() {
    initWindow();
    emscripten_run_script("onReady()");

    mainGUI(mainGUICallback);
    return 0;
}




EXTERN EMSCRIPTEN_KEEPALIVE
void resizeWindow(int w, int h) {
    RenderParams::iResolution = glm::ivec2(w, h);
    RenderParams::viewport->renderNeeded = true;
}


EXTERN EMSCRIPTEN_KEEPALIVE
void updateImage(std::string name, int w, int h, uint8_t *data) {
    printf("Image size: %d x %d\n", w, h);
    Img23d::name = name;
    Img23d::w = w;
    Img23d::h = h;
    if (Img23d::data)
        delete Img23d::data;
    Img23d::data = data;

    Img23d::containAlpha = false;
    for (int i = 0; i < w*h; i++)
        if (Img23d::data[4*i+3] != 255) {
            Img23d::containAlpha = true;
            break;
        }
    if (!Img23d::containAlpha)
        emscripten_run_script("onError('warning: image has no alpha')");

    Img23d::model = imageTo3D();
    prepareMesh(Img23d::model);
    RenderParams::viewport->renderNeeded = true;
}


std::vector<uint8_t> fileBuffer;

EXTERN EMSCRIPTEN_KEEPALIVE
bool isModelEmpty() {
    return renderModel.vertices.empty()
        || renderModel.indicesF.empty();
}

EXTERN EMSCRIPTEN_KEEPALIVE
size_t getFileSize() {
    return fileBuffer.size();
}

EXTERN EMSCRIPTEN_KEEPALIVE
uint8_t* generateSTL() {
    fileBuffer = writeSTL_(
        renderModel.vertices,
        renderModel.indicesF
    );
    return fileBuffer.data();
}

EXTERN EMSCRIPTEN_KEEPALIVE
uint8_t* generatePLY() {
    if (Img23d::smoothShading) {
        fileBuffer = writePLY_(
            renderModel.vertices,
            renderModel.indicesF,
            renderModel.normals
        );
    }
    else {
        Img23d::smoothShading = true;
        prepareMesh(Img23d::model);
        fileBuffer = writePLY_(
            renderModel.vertices,
            renderModel.indicesF
        );
        Img23d::smoothShading = false;
        prepareMesh(Img23d::model);
    }
    return fileBuffer.data();
}

EXTERN EMSCRIPTEN_KEEPALIVE
uint8_t* generateOBJ() {
    if (Img23d::smoothShading) {
        fileBuffer = writeOBJ_(
            Img23d::name.data(),
            renderModel.vertices,
            renderModel.indicesF,
            renderModel.normals
        );
    }
    else {
        Img23d::smoothShading = true;
        prepareMesh(Img23d::model);
        fileBuffer = writeOBJ_(
            Img23d::name.data(),
            renderModel.vertices,
            renderModel.indicesF
        );
        Img23d::smoothShading = false;
        prepareMesh(Img23d::model);
    }
    return fileBuffer.data();
}

EXTERN EMSCRIPTEN_KEEPALIVE
uint8_t* generateGLB() {
    int nbytes;
    uint8_t *imgbytes_raw = stbi_write_png_to_mem(
        Img23d::data, 4*Img23d::w, Img23d::w, Img23d::h, 4, &nbytes);
    std::vector<uint8_t> imgbytes(imgbytes_raw, imgbytes_raw+nbytes);
    free(imgbytes_raw);

    if (Img23d::smoothShading) {
        fileBuffer = writeGLB_(
            Img23d::name.data(),
            renderModel.vertices,
            renderModel.indicesF,
            renderModel.normals,
            renderModel.texcoords,
            imgbytes
        );
    }
    else {
        Img23d::smoothShading = true;
        prepareMesh(Img23d::model);
        fileBuffer = writeGLB_(
            Img23d::name.data(),
            renderModel.vertices,
            renderModel.indicesF,
            std::vector<vec3>(),
            renderModel.texcoords,
            imgbytes
        );
        Img23d::smoothShading = false;
        prepareMesh(Img23d::model);
    }
    return fileBuffer.data();
}
