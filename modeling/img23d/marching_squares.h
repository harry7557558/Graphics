#pragma once

#include "elements.h"

#include <vector>
#include <unordered_map>

void marchingSquares_(
    int nx, int ny, const uint8_t *img, uint8_t th,
    bool boundary_only,
    std::vector<vec2> &verts,
    std::vector<ivec2> &boundaryEdges,
    std::vector<ivec3> &trigs
) {
    float time0 = getTimePast();

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
                if (boundary_only && !(y == 0 || y == ny-1 || x == 0 || x == nx-1))
                    continue;
                vertmap[i] = (int)verts.size()+1;
                verts.push_back(vec2(x,y));
            }
        }
    }

    float time1 = getTimePast();

    // get trigs
    trigs.clear();
    std::unordered_set<uint64_t> bedges;
    const int LUTF[16][2][12] = {
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
    const int LUTE[16][2][12] = {
        { { -1 }, { -1 } },
        { { 0,4, 4,7, 7,0, -1 }, { -1 } },
        { { 4,1, 1,5, 5,4, -1 }, { -1 } },
        { { 0,1, 1,5, 5,7, 7,0, -1 }, { -1 } },
        { { 5,2, 2,6, 6,5, -1 }, { -1 } },
        { { 0,4, 4,7, 7,0, 5,2, 2,6, 6,5 }, { 0,4, 4,5, 5,2, 2,6, 6,7, 7,0 } },
        { { 4,1, 1,2, 2,6, 6,4, -1 }, { -1 } },
        { { 0,1, 1,2, 2,6, 6,7, 7,0, -1 }, { -1 } },
        { { 6,3, 3,7, 7,6, -1 }, { -1 } },
        { { 0,4, 4,6, 6,3, 3,0, -1 }, { -1 } },
        { { 3,7, 7,4, 4,1, 1,5, 5,6, 6,3 }, { 6,3, 3,7, 7,6, 4,1, 1,5, 5,4 } },
        { { 0,1, 1,5, 5,6, 6,3, 3,0, -1 }, { -1 } },
        { { 7,5, 5,2, 2,3, 3,7, -1 }, { -1 } },
        { { 0,4, 4,5, 5,2, 2,3, 3,0, -1 }, { -1 } },
        { { 4,1, 1,2, 2,3, 3,7, 7,4, -1 }, { -1 } },
        { { 0,1, 1,2, 2,3, 3,0, -1 }, { -1 } },
    };
    auto angleCost = [&](vec2 a, vec2 b) -> float {
        return -log(1.000001f-dot(normalize(a), normalize(b))) + log(2);
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
                vertmap[vi[0]]-1, vertmap[vi[1]]-1, vertmap[vi[2]]-1, vertmap[vi[3]]-1,
                edgemap[ei[0]], edgemap[ei[1]], edgemap[ei[2]], edgemap[ei[3]],
            };
            // cross add trig
            if (idx == 15) {
                int pi = (x&1)^(y&1);
                if (boundary_only) {
                    const int *lut = &LUTE[idx][0][0];
                    for (int _ = 0; _ < 12 && lut[_] != -1; _ += 2) {
                        ivec2 e = { i[lut[_]], i[lut[_+1]] };
                        if (e[0] != -1 && e[1] != -1) {
                            if (bedges.find(getEdgeIdx(e[1], e[0])) != bedges.end())
                                bedges.erase(getEdgeIdx(e[1], e[0]));
                            else bedges.insert(getEdgeIdx(e[0], e[1]));
                        }
                    }
                }
                else {
                    const int *lut = &LUTF[idx][pi][0];
                    if (pi == 1 && *lut == -1)
                        lut = &LUTF[idx][0][0];
                    for (int _ = 0; _ < 12 && lut[_] != -1; _ += 3)
                        trigs.push_back({ i[lut[_]], i[lut[_+1]], i[lut[_+2]] });
                }
            }
            // choose the combination with lower aspect ratio
            else {
                std::vector<ivec3> newTrigs[2];
                float cost[2] = { 0.0, 0.0 };
                for (int pi = 0; pi < 2; pi++) {
                    const int *lut = &LUTF[idx][pi][0];
                    for (int _ = 0; _ < 12 && lut[_] != -1; _ += 3) {
                        ivec3 t(i[lut[_]], i[lut[_+1]], i[lut[_+2]]);
                        newTrigs[pi].push_back(t);
                        cost[pi] = (cost[pi]*(_/3) + trigCost(t)) / (_/3+1);
                    }
                }
                if (boundary_only) {
                    int pi = LUTE[idx][1][0] == -1 || cost[0] < cost[1] ? 0 : 1;
                    const int *lut = &LUTE[idx][pi][0];
                    for (int _ = 0; _ < 12 && lut[_] != -1; _ += 2) {
                        ivec2 e = { i[lut[_]], i[lut[_+1]] };
                        if (e[0] != -1 && e[1] != -1) {
                            if (bedges.find(getEdgeIdx(e[1], e[0])) != bedges.end())
                                bedges.erase(getEdgeIdx(e[1], e[0]));
                            else bedges.insert(getEdgeIdx(e[0], e[1]));
                        }
                    }
                }
                else {
                    std::vector<ivec3> app = cost[1] == 0.0 ? newTrigs[0]:
                        cost[0] < cost[1] ? newTrigs[0] : newTrigs[1];
                    for (ivec3 t : app)
                        trigs.push_back(t);
                }
            }
        }
    }

    float time2 = getTimePast();

    if (boundary_only) {
        boundaryEdges.clear();
        for (uint64_t e : bedges)
            boundaryEdges.push_back(ivec2(int(e>>32), int(e)));
    }

    float time3 = getTimePast();
    printf("marchingSquares_: %.2g + %.2g + %.2g = %.2g secs\n",
        time1-time0, time2-time1, time3-time2, time3-time0);
}


void marchingSquaresTrigs(
    int nx, int ny, const uint8_t *img, uint8_t th,
    std::vector<vec2> &verts, std::vector<ivec3> &trigs
){
    std::vector<ivec2> boundaryEdges;
    marchingSquares_(nx, ny, img, th, false, verts, boundaryEdges, trigs);
}

void marchingSquaresEdges(
    int nx, int ny, const uint8_t *img, uint8_t th,
    std::vector<vec2> &verts, std::vector<std::vector<int>> boundary
    // std::vector<vec2> &verts, std::vector<ivec3> &trigs
) {
    std::vector<ivec2> boundaryEdges;
    std::vector<ivec3> trigs;
    marchingSquares_(nx, ny, img, th, true, verts, boundaryEdges, trigs);

    printf("%d %d\n", (int)verts.size(), (int)boundaryEdges.size());
    assert(verts.size() == boundaryEdges.size());

    std::vector<ivec2> neighbors(verts.size(), ivec2(-1));
    for (ivec2 e : boundaryEdges) {
        assert(neighbors[e.x][1] == -1);
        neighbors[e.x][1] = e.y;
        assert(neighbors[e.y][0] == -1);
        neighbors[e.y][0] = e.x;
    }
    for (ivec2 n : neighbors)
        assert(n.x != -1 && n.y != -1);

    std::unordered_set<int> remainingVerts;
    for (int i = 0; i < (int)verts.size(); i++)
        remainingVerts.insert(i);

    // std::vector<std::vector<int>> boundary;
    while (!remainingVerts.empty()) {
        int p = -1;
        for (int p1 : remainingVerts)
            { p = p1; break; }
        std::vector<int> contour;
        do  {
            contour.push_back(p);
            p = neighbors[p][1];
            remainingVerts.erase(p);
        } while (p != contour[0]);
        // printf("%d\n", (int)contour.size());
        boundary.push_back(contour);
    }
    printf("%d contours\n", (int)boundary.size());

    // for (ivec2 e : boundaryEdges)
    //     trigs.push_back({ e[0], e[0], e[1] });
}
