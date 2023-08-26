// CLI, compile with GCC

#pragma GCC optimize "O2"
#define SUPPRESS_ASSERT 1

#include <cstdio>
#include "solver.h"
#include "meshgen_trig_implicit.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "write_model.h"

int main(int argc, char* argv[]) {
    std::string filename = "";
    std::string fileout = "";
    float threshold = 0.5;
    float grid0 = 0.01;
    int depth = 4;
    bool hasNormal = true;
    bool hasColor = false;

    enum ArgvNext {
        vNone, vFilename, vOut, vThreshold, vGrid0, vDepth, vNormal, vColor, vEnd
    } argvNext = vFilename;
    for (int i = 1; i <= argc; i++) {
        if (i < argc && argv[i][0] == '-') {
            if (argvNext == vNormal)
                hasNormal = true;
            if (argvNext == vColor)
                hasColor = true;
            const char *p = &argv[i][0];
            while (*p == '-') p++;
            argvNext = vNone;
            if (*p == 'i')
                argvNext = vFilename;
            if (*p == 'o')
                argvNext = vOut;
            if (*p == 't')
                argvNext = vThreshold;
            if (*p == 'g')
                argvNext = vGrid0;
            if (*p == 'd')
                argvNext = vDepth;
            if (*p == 'n')
                argvNext = vNormal;
            if (*p == 'c')
                argvNext = vColor;
            if (argvNext == vNone) {
                printf("Unknown parameter name %s\n", argv[i]);
                return 1;
            }
            continue;
        }
        if (argvNext == vFilename) {
            if (i < argc)
                filename = argv[i];
        }
        else if (argvNext == vOut) {
            if (i < argc)
                fileout = argv[i];
        }
        else if (argvNext == vThreshold) {
            try {
                if (i < argc)
                    threshold = std::stof(argv[i]);
            } catch (...) { threshold = -1.0; }
            if (!(threshold > 0.0 && threshold < 1.0)) {
                printf("Threshold (%s) must be *between* 0 and 1.\n", argv[i]);
                return 1;
            }
        }
        else if (argvNext == vGrid0) {
            try {
                if (i < argc)
                    grid0 = std::stof(argv[i]);
            } catch (...) { grid0 = -1.0; }
            if (!(grid0 > 0.0 && grid0 < 1.0)) {
                printf("Initial grid (%s) must be *between* 0 and 1.\n", argv[i]);
                return 1;
            }
        }
        else if (argvNext == vDepth) {
            try {
                if (i < argc)
                    depth = std::stoi(argv[i]);
            } catch (...) { depth = -1; }
            if (!(depth >= 0)) {
                printf("Depth (%s) must be a non-negative integer.\n", argv[i]);
                return 1;
            }
        }
        else if (argvNext == vNormal) {
            hasNormal = i < argc ? (argv[i][0] != '0') : true;
        }
        else if (argvNext == vColor) {
            hasColor = i < argc ? (argv[i][0] != '0') : true;
        }
        if (argvNext != vEnd)
            argvNext = (ArgvNext)((int)argvNext + 1);
        else argvNext = vNone;
    }

    if (filename == "")
        return 0 * printf("No input file specified.\n");
    if (fileout == "")
        return 0 * printf("No output file specificed.\n");

    enum FileType {
        fUnknown, STL, PLY, OBJ, GLB
    } fileType = fUnknown;
    int extstart = fileout.rfind('.')+1;
    std::string ext = fileout.substr(extstart, (int)fileout.size()-extstart);
    if (ext == "stl") fileType = STL;
    else if (ext == "ply") fileType = PLY;
    else if (ext == "obj") fileType = OBJ;
    else if (ext == "glb") fileType = GLB;
    else printf("Warning: Unknow file type %s\n", &ext[0]);

    int X, Y;
    uint8_t *pixels = (uint8_t*)stbi_load(&filename[0], &X, &Y, nullptr, 4);

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
        x = 0.5f + 0.5f * x / scale.x;
        y = 0.5f + 0.5f * y / scale.y;
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

    float t4 = getTimePast();
    printf("Total: %.2g secs.\n", t4-t0);

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

    // STL
    if (fileType == STL) {
        writeSTL(&fileout[0], verts, trigs);
        free(pixels);
        return 0;
    }

    // compute normal
    std::vector<vec3> normals;
    if (hasNormal) {
        normals = std::vector<vec3>(verts.size(), vec3(0));
        for (ivec3 t : trigs) {
            vec3 n = cross(verts[t[1]]-verts[t[0]], verts[t[2]]-verts[t[0]]);
            n = normalize(n);
            for (int _ = 0; _ < 3; _++)
                normals[t[_]] += n;
        }
        for (size_t i = 0; i < normals.size(); i++)
            normals[i] = normalize(normals[i]);
    }

    // PLY
    if (fileType == PLY) {
        writePLY(&fileout[0], verts, trigs,
            hasNormal ? normals : std::vector<vec3>());
        free(pixels);
        return 0;
    }

    // compute texture coords
    std::vector<vec2> texcoords;
    if (hasColor) {
        texcoords = std::vector<vec2>(verts.size(), vec3(0));
        for (int i = 0; i < (int)verts.size(); i++) {
            texcoords[i] = 0.5f + 0.5f * vec2(verts[i].x, -verts[i].y) / scale;
        }
    }

    // compute texture, BFS remove aliasing
    std::vector<uint8_t> imgbytes;
    if (hasColor) {
        uint8_t pth = (uint8_t)(threshold*255+0.5);
        for (int i = 0; i < X * Y; i++)
            pixels[4*i+3] = (pixels[4*i+3]>pth ? 255 : 0);
        std::vector<ivec2> visited;
        for (int y = 0; y < Y; y++)
            for (int x = 0; x < X; x++)
                if (pixels[4*(y*X+x)+3])
                    visited.push_back(ivec2(x, y));
        while (!visited.empty()) {
            std::vector<ivec2> visited1;
            for (ivec2 ij : visited) {
                int i0 = ij.x, j0 = ij.y;
                for (int di = -1; di <= 1; di++)
                    for (int dj = -1; dj <= 1; dj++) {
                        int i = i0+di, j = j0+dj;
                        if (i < 0 || j < 0 || i >= X || j >= Y)
                            continue;
                        if (pixels[4*(j*X+i)+3])
                            continue;
                        visited1.push_back(ivec2(i, j));
                        ((uint32_t*)pixels)[j*X+i] = ((uint32_t*)pixels)[j0*X+i0];
                    }
            }
            visited = visited1;
        }
        int nbytes;
        uint8_t *imgbytes_raw = stbi_write_png_to_mem(pixels, 4*X, X, Y, 4, &nbytes);
        imgbytes = std::vector<uint8_t>(imgbytes_raw, imgbytes_raw+nbytes);
        free(imgbytes_raw);
    }
    free(pixels);

    // OBJ and GLB
    if (fileType == OBJ) {
        writeOBJ(&fileout[0], verts, trigs, normals, texcoords, imgbytes);
        return 0;
    }
    if (fileType == GLB || true) {
        writeGLB(&fileout[0], verts, trigs, normals, texcoords, imgbytes);
        return 0;
    }

    return 0;
}