#pragma once

#include "elements.h"

#include <vector>
#include <unordered_map>

void marchingSquares(
    int nx, int ny, const uint8_t *img, uint8_t th,
    std::vector<vec2> &verts, std::vector<ivec3> &trigs
) {

    // get edges with sign change
    std::unordered_map<uint64_t, int> edgemap;
    auto getEdgeIdx = [](int i0, int i1) -> uint64_t {
        return ((uint64_t)i0<<32) | (uint64_t)i1;
    };
    verts.clear();
    float thf = (float)th+0.5f;
    for (int y = 0; y < ny-1; y++) {
        for (int x = 0; x < nx; x++) {
            int i0 = y*nx+x;
            int i1 = (y+1)*nx+x;
            if ((img[i0]>th) ^ (img[i1]>th)) {
                float t = -((float)img[i0] - thf) / ((float)img[i1] - (float)img[i0]);
                edgemap[getEdgeIdx(i0,i1)] = (int)verts.size();
                verts.push_back(mix(vec2(x,y), vec2(x,y+1), t));
            }
        }
    }
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx-1; x++) {
            int i0 = y*nx+x;
            int i1 = y*nx+(x+1);
            if ((img[i0]>th) ^ (img[i1]>th)) {
                float t = -((float)img[i0] - thf) / ((float)img[i1] - (float)img[i0]);
                edgemap[getEdgeIdx(i0,i1)] = (int)verts.size();
                verts.push_back(mix(vec2(x,y), vec2(x+1,y), t));
            }
        }
    }

    // get inside vertices
    std::unordered_map<int, int> vertmap;
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            int i = y*nx+x;
            if (img[i] <= th) {
                vertmap[i] = (int)verts.size();
                verts.push_back(vec2(x,y));
            }
        }
    }

    // get trigs
    trigs.clear();
    const int LUT[16][2][12] = {
        { { -1 }, { -1 } },
        { { 0,4,7, -1 }, { -1 } },
        { { 4,1,5, -1 }, { -1 } },
        { { 0,1,7, 7,1,5, -1 }, { 0,1,5, 0,5,7, -1 } },
        { { 5,2,6, -1 }, { -1 } },
        { { 0,4,7, 5,2,6, -1 }, { 0,4,5, 0,5,2, 0,2,6, 0,6,7 } },
        { { 4,1,6, 6,1,2, -1 }, { 4,1,2, 4,2,6, -1 } },
        { { 0,1,7, 7,1,6, 6,1,2, -1 }, { 0,6,7, 0,2,6, 0,1,2, -1 } },
        { { 7,6,3, -1 }, { -1 } },
        { { 0,4,3, 3,4,6, -1 }, { 0,4,6, 0,6,3, -1 } },
        { { 1,5,6, 1,6,3, 1,3,7, 1,7,4 }, { 7,6,3, 4,1,5, -1 } },
        { { 0,1,3, 3,1,6, 6,1,5, -1 }, { 0,1,5, 0,5,6, 0,6,3, -1 } },
        { { 7,5,3, 3,5,2, -1 }, { 7,5,2, 7,2,3, -1 } },
        { { 0,4,3, 3,4,5, 3,5,2, -1 }, { 0,2,3, 0,5,2, 0,4,5, -1 } },
        { { 7,4,3, 3,4,1, 3,1,2, -1 }, { 3,7,2, 2,7,4, 2,4,1, -1 } },
        { { 0,1,3, 3,1,2, -1 }, { 0,1,2, 0,2,3, -1 } },
    };
    auto angleCost = [&](vec2 a, vec2 b) -> float {
        return -log(1.000001f-dot(normalize(a), normalize(b)));
    };
    auto trigCost = [&](ivec3 t) -> float {
        float c = 0.0;
        for (int _ = 0; _ < 3; _++)
            c += angleCost(
                verts[t[(_+1)%3]]-verts[t[_]],
                verts[t[(_+2)%3]]-verts[t[_]]
            );
        return c;
    };
    for (int y = 0; y < ny-1; y++) {
        for (int x = 0; x < nx-1; x++) {
            int vi[4] = {
                y*nx+x, y*nx+(x+1),
                (y+1)*nx+(x+1), (y+1)*nx+x,
            };
            int idx = 0;
            for (int _ = 0; _ < 4; _++)
                idx |= int(img[vi[_]] <= th) << _;
            if (idx == 0)
                continue;
            uint64_t ei[4] = {
                getEdgeIdx(vi[0], vi[1]),
                getEdgeIdx(vi[1], vi[2]),
                getEdgeIdx(vi[3], vi[2]),
                getEdgeIdx(vi[0], vi[3])
            };
            int i[8] = {
                vertmap[vi[0]], vertmap[vi[1]], vertmap[vi[2]], vertmap[vi[3]],
                edgemap[ei[0]], edgemap[ei[1]], edgemap[ei[2]], edgemap[ei[3]],
            };
            // cross add trig
            if (idx == 15) {
                int pi = (x&1)^(y&1);
                const int *lut = &LUT[idx][pi][0];
                if (pi == 1 && *lut == -1)
                    lut = &LUT[idx][0][0];
                for (int _ = 0; _ < 12 && lut[_] != -1; _ += 3)
                    trigs.push_back({ i[lut[_]], i[lut[_+1]], i[lut[_+2]] });
            }
            // choose the combination with lower aspect ratio
            else {
                std::vector<ivec3> newTrigs[2];
                float cost[2] = { 0.0, 0.0 };
                for (int pi = 0; pi < 2; pi++) {
                    const int *lut = &LUT[idx][pi][0];
                    for (int _ = 0; _ < 12 && lut[_] != -1; _ += 3) {
                        ivec3 t(i[lut[_]], i[lut[_+1]], i[lut[_+2]]);
                        newTrigs[pi].push_back(t);
                        cost[pi] = (cost[pi]*(_/3) + trigCost(t)) / (_/3+1);
                    }
                }
                std::vector<ivec3> app = cost[1] == 0.0 ? newTrigs[0]:
                    cost[0] < cost[1] ? newTrigs[0] : newTrigs[1];
                for (ivec3 t : app)
                    trigs.push_back(t);
            }
        }
    }

}

