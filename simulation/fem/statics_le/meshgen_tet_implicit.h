// Generate a tetrahedral mesh from an implicitly defined object
// Defined by: F(x,y,z) < 0

#pragma once

#include <functional>
#include <map>
#include <set>
#include <cstdint>
#include <stdio.h>
#include "elements.h"

#define MESHGEN_TET_IMPLICIT_NS_START namespace MeshgenTetImplicit {
#define MESHGEN_TET_IMPLICIT_NS_END }

MESHGEN_TET_IMPLICIT_NS_START


const int EDGES[6][2] = {
    {0, 1}, {0, 2}, {0, 3},
    {1, 2}, {1, 3}, {2, 3}
};
const int FACES[4][6] = {  // ccw, missing the index
    {1, 2, 3, 7, 9, 8},
    {0, 3, 2, 6, 9, 5},
    {0, 1, 3, 4, 8, 6},
    {0, 2, 1, 5, 7, 4}
};

// Lookup table for surface reconstruction
// 16: vertice signs, little endian, bit 1 = negative
// 6: max 6 possible tetrahedrations to choose from
// 12: max 3 groups of 4 vertices of tetrahedra
const int LUT_MARCH_T[16][6][12] = {
{ {-1} },
{ {0,4,5,6,-1}, {-1} },
{ {1,4,8,7,-1}, {-1} },
{ {0,1,7,8,0,6,8,7,0,5,6,7}, {0,1,7,8,0,5,6,8,0,5,8,7}, {0,1,5,6,1,5,6,7,1,6,8,7}, {0,1,5,6,1,5,8,7,1,5,6,8}, {0,5,6,8,0,1,5,8,1,5,8,7}, {0,5,6,7,0,1,7,6,1,6,8,7} },
{ {2,5,7,9,-1}, {-1} },
{ {0,6,7,9,0,4,7,6,0,2,9,7}, {0,4,9,6,0,2,9,4,2,4,7,9}, {0,2,6,4,2,4,7,9,2,4,9,6}, {0,4,9,6,0,4,7,9,0,2,9,7}, {2,6,7,9,0,4,7,6,0,2,6,7}, {0,2,6,4,2,6,7,9,2,4,7,6} },
{ {1,2,4,8,2,4,9,5,2,4,8,9}, {1,2,4,9,1,4,8,9,2,4,9,5}, {1,4,9,5,1,4,8,9,1,2,5,9}, {2,5,8,9,1,4,8,5,1,2,5,8}, {1,4,8,5,1,5,8,9,1,2,5,9}, {2,5,8,9,2,4,8,5,1,2,4,8} },
{ {0,1,9,6,1,6,8,9,0,1,2,9}, {0,1,2,8,2,6,8,9,0,2,6,8}, {1,2,6,8,2,6,8,9,0,1,2,6}, {0,1,2,9,0,1,9,8,0,6,8,9}, {1,2,6,9,1,6,8,9,0,1,2,6}, {0,2,9,8,0,1,2,8,0,6,8,9} },
{ {3,6,9,8,-1}, {-1} },
{ {3,5,9,8,3,4,5,8,0,3,4,5}, {0,3,8,9,0,5,9,8,0,4,5,8}, {3,4,5,9,3,4,9,8,0,3,4,5}, {0,4,5,8,3,5,9,8,0,3,8,5}, {0,3,8,9,0,4,5,9,0,4,9,8}, {0,4,5,9,0,3,4,9,3,4,9,8} },
{ {3,6,9,7,1,3,7,4,3,4,6,7}, {3,4,6,9,1,4,9,7,1,3,9,4}, {1,4,9,7,1,3,9,6,1,4,6,9}, {1,4,6,7,1,3,9,6,1,6,9,7}, {3,6,9,7,1,4,6,7,1,3,7,6}, {3,4,6,9,1,3,7,4,3,4,9,7} },
{ {3,5,9,7,0,1,7,3,0,3,7,5}, {0,5,9,7,0,1,7,9,0,1,9,3}, {0,1,5,9,1,5,9,7,0,1,9,3}, {0,5,9,7,0,1,7,3,0,3,7,9}, {1,3,9,5,0,1,5,3,1,5,9,7}, {3,5,9,7,0,1,5,3,1,3,7,5} },
{ {2,5,7,8,3,5,8,6,2,3,5,8}, {2,5,7,6,3,6,7,8,2,3,6,7}, {3,5,7,6,3,6,7,8,2,3,5,7}, {2,5,7,6,2,6,7,8,2,3,6,8}, {2,5,7,8,2,5,8,6,2,3,6,8}, {3,5,7,8,3,5,8,6,2,3,5,7} },
{ {3,4,7,8,0,3,4,7,0,2,3,7}, {0,2,3,4,2,4,7,8,2,3,4,8}, {0,2,8,4,2,4,7,8,0,2,3,8}, {0,2,3,4,2,3,4,7,3,4,7,8}, {0,4,7,8,0,2,8,7,0,2,3,8}, {0,4,7,8,0,2,3,7,0,3,8,7} },
{ {1,4,6,5,1,2,5,3,1,3,5,6}, {1,4,6,5,1,2,6,3,1,2,5,6}, {1,2,6,3,2,4,6,5,1,2,4,6}, {3,4,6,5,1,3,5,4,1,2,5,3}, {2,3,5,4,1,2,4,3,3,4,6,5}, {2,4,6,5,2,3,6,4,1,2,4,3} },
{ {0,1,2,3,-1}, {-1} },
};

// Lookup table for surface reconstruction
// Whether there is a point on an edge
const bool LUT_MARCH_E[16][6] = {
{0,0,0,0,0,0}, {1,1,1,0,0,0}, {1,0,0,1,1,0}, {0,1,1,1,1,0},
{0,1,0,1,0,1}, {1,0,1,1,0,1}, {1,1,0,0,1,1}, {0,0,1,0,1,1},
{0,0,1,0,1,1}, {1,1,0,0,1,1}, {1,0,1,1,0,1}, {0,1,0,1,0,1},
{0,1,1,1,1,0}, {1,0,0,1,1,0}, {1,1,1,0,0,0}, {0,0,0,0,0,0},
};

// Lookup table for surface reconstruction
// Face modes; Max 2 faces
const int LUT_MARCH_F[8][2][6] = {
    { {-1} },
    { {0,3,5, -1}, {-1} },
    { {1,4,3, -1}, {-1} },
    { {0,1,5, 5,1,4}, {0,1,4, 0,4,5} },
    { {2,5,4, -1}, {-1} },
    { {2,0,4, 4,0,3}, {2,0,3, 2,3,4} },
    { {1,2,3, 3,2,5}, {1,2,5, 1,5,3} },
    { {0,1,2, -1}, {-1} },
};


typedef std::function<double(double, double, double)> ScalarFieldF;

struct MeshVertex {
    vec3 x;
    double fv;
};


// Expand a single tetrahedron to a tetrahedra mesh filling the volume
// The initial tetrahedron should be a regular tetrahedron with a positive volume
void generateInitialTetrahedra(
    ScalarFieldF F,
    std::vector<MeshVertex>& vertices, std::vector<ivec4>& tets
) {

    // one initial tetrahedron
    assert(vertices.size() == 4);
    assert(tets.size() == 1);
    for (int i = 0; i < 4; i++) {
        vec3 p = vertices[i].x;
        vertices[i].fv = F(p.x, p.y, p.z);
    }

    // store a list of added vertices, use integers to avoid rounding error
    // initially, vertices is (0,0,0), (1,1,0), (0,1,1), (1,0,1)
    auto ivec3Cmp = [&](ivec3 a, ivec3 b) -> bool {
        return a.z != b.z ? a.z < b.z :
            *((uint64_t*)&a.x) < *((uint64_t*)&b.x);
    };
    std::map<ivec3, int, decltype(ivec3Cmp)> vmp(ivec3Cmp);  // map coordinates to index
    std::vector<ivec3> ixs({ ivec3(0, 0, 0),
        ivec3(1, 1, 0), ivec3(0, 1, 1), ivec3(1, 0, 1) });  // map index to coordinates
    for (int i = 0; i < 4; i++) vmp[ixs[i]] = i;
    mat3 i2p = mat3(
        vertices[1].x - vertices[0].x,
        vertices[2].x - vertices[0].x,
        vertices[3].x - vertices[0].x
    ) * inverse(mat3(1, 1, 0, 0, 1, 1, 1, 0, 1));

    // a list of added tets
    auto ivec4Cmp = [&](ivec4 a, ivec4 b) -> bool {
        std::sort(&a.x, &a.x + 4);
        std::sort(&b.x, &b.x + 4);
        return a.z == b.z && a.w == b.w ?
            *((uint64_t*)&a.x) < *((uint64_t*)&b.x) :
            *((uint64_t*)&a.z) < *((uint64_t*)&b.z);
    };
    std::set<ivec4, decltype(ivec4Cmp)> tAdded(ivec4Cmp);
    tAdded.insert(tets[0]);

    // BFS advancing
    // https://en.wikipedia.org/wiki/Tetrahedral-octahedral_honeycomb
    // Divide an octahedron into 4 identical tetrahedra
    int s0 = 0, s1 = (int)tets.size();
    while (s0 != s1) {
        for (int s_ = s0; s_ < s1; s_++) {
            for (int fi = 0; fi < 4; fi++) {
                const int* s = (int*)&tets[s_];
                ivec3 evi[3];
                bool allPositive = true;
                for (int _ = 0; _ < 3; _++) {
                    int vi = s[FACES[fi][_]];
                    evi[_] = ixs[vi];
                    allPositive &= (vertices[vi].fv > 0.0);
                }
                if (allPositive) continue;
                ivec3 evi0 = ixs[s[fi]];
                ivec3 ix1;
                ivec4 t(s[FACES[fi][0]], s[FACES[fi][1]], s[FACES[fi][2]], -1);
                const bool octa4 = true;  // divide an octahedron into 4 tetrahedra instead of 8
                if ((evi[0].x != evi[1].x || evi[0].x != evi[2].x) &&
                    (evi[0].y != evi[1].y || evi[0].y != evi[2].y) &&
                    (evi[0].z != evi[1].z || evi[0].z != evi[2].z)
                    ) {  // middle tetrahedron
                    ix1 = ivec3(
                        evi[0].x == evi[1].x ? evi[0].x : evi[2].x,
                        evi[0].y == evi[1].y ? evi[0].y : evi[2].y,
                        evi[0].z == evi[1].z ? evi[0].z : evi[2].z
                    );
                    if (det(evi[1] - evi[0], evi[2] - evi[0], ix1 - evi[0]) < 0) {  // regular tetrahedron
                        ix1 = evi[0] + evi[1] + evi[2] - 2 * ix1;
                    }
                    else if (octa4) {  // 1/4 of an octahedron
                        ivec3 ix2 = ix1 / 2;
                        int dir = ((ix2.x ^ ix2.y ^ ix2.z) % 3 + 3) % 3;
                        if (dir == 0) ix1.x += std::min({ evi[0].x, evi[1].x, evi[2].x }) == ix1.x ? -1 : 1;
                        if (dir == 1) ix1.y += std::min({ evi[0].y, evi[1].y, evi[2].y }) == ix1.y ? -1 : 1;
                        if (dir == 2) ix1.z += std::min({ evi[0].z, evi[1].z, evi[2].z }) == ix1.z ? -1 : 1;
                    }
                }
                else {  // splitted octahedron
                    ivec3 n = cross(evi[1] - evi[0], evi[2] - evi[0]);
                    assert(dot(n, n) == (octa4 ? 4 : 1));
                    if (octa4) n = n / 2;
                    ix1 = evi0 - 2 * dot(evi0 - evi[0], n) * n;
                }
                if (vmp.find(ix1) == vmp.end()) {  // new point
                    vmp[ix1] = t.w = ixs.size();
                    ixs.push_back(ix1);
                    assert(vmp.size() == ixs.size());
                    vec3 p = vertices[0].x + i2p * vec3(ix1);
                    vertices.push_back(MeshVertex{ p, F(p.x, p.y, p.z) });
                }
                else t.w = vmp[ix1];
                assert(t.w >= 0 && t.w < ixs.size());
                if (tAdded.find(t) == tAdded.end()) {
                    tets.push_back(t);
                    tAdded.insert(t);
                }
            }
        }
        s0 = s1, s1 = (int)tets.size();
    }
}


// Quasi-random
double vanDerCorput(int n, int b) {
    double x = 0.0, e = 1.0 / b;
    while (n) {
        x += (n % b) * e;
        e /= b, n /= b;
    }
    return x;
}


// Generate a tetrahedral mesh from a scalar field
// The domain is constrained to a box
// The object should be connected
ScalarFieldF generateInitialTetrahedraInBox(
    ScalarFieldF F0, vec3 bc, vec3 br, double tetSize,
    std::vector<MeshVertex>& vertices, std::vector<ivec4>& tets
) {
    assert(vertices.empty() && tets.empty());
    ScalarFieldF F = [=](double x, double y, double z) {
        double f = F0(x, y, z);
        double b = std::max({
            abs(x - bc.x) - br.x,
            abs(y - bc.y) - br.y,
            abs(z - bc.z) - br.z });
        return max(f, b);
    };
    // find a point inside the object
    // introduce some randomness to prevent degenerate cases
    vec3 c; int i;
    for (i = 0x1000000; i < 0x1010000; i++) {
        vec3 s(vanDerCorput(i, 2), vanDerCorput(i, 3), vanDerCorput(i, 5));
        c = bc + br * (2.0 * s - 1.0);
        if (F0(c.x, c.y, c.z) < 0.) break;
    }
    // random rotation matrix
    i++;
    double u = 2.0 * PI * vanDerCorput(i, 2);
    double v = 2.0 * vanDerCorput(i, 3) - 1.0;
    mat3 m = axis_angle(vec3(
        sqrt(1.0 - v * v) * cos(u), sqrt(1.0 - v * v) * sin(u), v),
        2.0 * PI * vanDerCorput(i, 5));
    // generate a tetrahedron
    m = tetSize * m;
    vertices.assign({
        MeshVertex{ c },
        MeshVertex{ c + m * vec3(1, 1, 0) },
        MeshVertex{ c + m * vec3(0, 1, 1) },
        MeshVertex{ c + m * vec3(1, 0, 1) }
        });
    tets.push_back(ivec4(0, 1, 2, 3));
    generateInitialTetrahedra(F, vertices, tets);
    return F;
}


void cutIsosurface(
    ScalarFieldF F0,
    const std::vector<MeshVertex>& verts, const std::vector<ivec4>& tets,
    std::vector<vec3>& resVerts, std::vector<ivec4>& resTets
) {
    // map index in `verts` to index in `resVerts`
    std::vector<int> vMap(verts.size(), -1);

    // map edge vertex to index in `resVerts`
    auto ivec2Cmp = [](ivec2 a, ivec2 b) {
        assert(a.x < a.y && b.x < b.y);
        return a.x == b.x ? a.y < b.y : a.x < b.x;
    };
    std::map<ivec2, int, decltype(ivec2Cmp)> eMap(ivec2Cmp);

    // generate a list of points
    auto calcIndex = [&](ivec4 tet) -> int {
        int idx = 0;
        for (int _ = 0; _ < 4; _++)
            idx |= int(verts[tet[_]].fv < 0.) << _;
        return idx;
    };
    for (ivec4 tet : tets) {
        // add vertices to list
        for (int _ = 0; _ < 4; _++) {
            int i = tet[_];
            if (vMap[i] == -1 && verts[i].fv < 0.) {
                vMap[i] = (int)resVerts.size();
                resVerts.push_back(verts[i].x);
            }
        }
        // add edges to list
        int idx = calcIndex(tet);
        auto LutE = LUT_MARCH_E[idx];
        for (int _ = 0; _ < 6; _++) {
            if (LutE[_]) {
                int i = tet[EDGES[_][0]], j = tet[EDGES[_][1]];
                if (i > j) std::swap(i, j);
                if (eMap.find(ivec2(i, j)) == eMap.end()) {
                    eMap[ivec2(i, j)] = (int)resVerts.size();
                    double v0 = verts[i].fv, v1 = verts[j].fv;
                    vec3 x0 = verts[i].x, x1 = verts[j].x;
                    assert(v0 * v1 <= 0.0);
                    double t = -verts[i].fv / (v1 - v0);
                    vec3 pe = mix(x0, x1, t);
                    if (true) {  // quadratic interpolation
                        double vc = F0(pe.x, pe.y, pe.z);
                        double a = -vc; // (1.0 - t) * v0 + t * v1 - vc
                        double b = (t * t - 1) * v0 + (-t * t) * v1 + vc;
                        double c = (t - t * t) * v0;
                        double d = sqrt(b * b - 4.0 * a * c);
                        double t1 = (-b + d) / (2.0 * a), t2 = (-b - d) / (2.0 * a);
                        t = a == 0.0 ? -c / b : abs(t1 - 0.5) < abs(t2 - 0.5) ? t1 : t2;
                        assert(t >= 0.0 && t <= 1.0);
                        pe = mix(x0, x1, t);
                    }
                    resVerts.push_back(pe);
                }
            }
        }
    }

    // generate a list of faces
    auto rotateIvec3 = [](ivec3 v) {
        assert(v.x != v.y && v.x != v.z && v.y != v.z);
        int i = v.x < v.y && v.x < v.z ? 0 :
            v.y < v.x && v.y < v.z ? 1 : 2;
        return ivec3(v[i], v[(i + 1) % 3], v[(i + 2) % 3]);
    };
    auto ivec3Cmp = [&](ivec3 a, ivec3 b) -> bool {
        return a.z != b.z ? a.z < b.z :
            *((uint64_t*)&a.x) < *((uint64_t*)&b.x);
    };
    std::map<ivec3, const int*, decltype(ivec3Cmp)> fMap(ivec3Cmp);  // face, type in LUT
    for (ivec4 tet : tets) {
        int idxt = calcIndex(tet);
        if (idxt == 0) continue;
        // if it's a prism, all quad face indices must not be the same for it to have a solution
        bool isPrismIndex = (idxt == 0b0111 || idxt == 0b1011 || idxt == 0b1101 || idxt == 0b1110);
        // go through each face
        int idx[4]; bool isQuadIndex[4]; ivec3 f[4];
        int qti[4] = { -1,-1,-1,-1 };  // triangulation choice index
        for (int fi = 0; fi < 4; fi++) {
            // get face
            for (int _ = 0; _ < 3; _++)
                f[fi][_] = tet[FACES[fi][_]];
            // calculate face index
            idx[fi] = 0;
            for (int _ = 0; _ < 3; _++)
                idx[fi] |= int(verts[f[fi][_]].fv < 0.) << _;
            isQuadIndex[fi] = (idx[fi] == 3 || idx[fi] == 5 || idx[fi] == 6);
            f[fi] = rotateIvec3(f[fi]);
            // already exists
            ivec3 fo = rotateIvec3(ivec3(f[fi].x, f[fi].z, f[fi].y));
            if (isQuadIndex[fi] && fMap.find(fo) != fMap.end()) {
                const int* p = fMap[fo];
                int idxo = ((p - LUT_MARCH_F[0][0]) / 6) % 2;
                qti[fi] = 1 - idxo;
            }
        }
        // fill qti, 2 ways to divide
        if (isPrismIndex) {
            int has1 = 0, has0 = 0;
            for (int fi = 0; fi < 4; fi++)
                has1 += (qti[fi] == 1), has0 += (qti[fi] == 0);
            int fill = has1 ? 0 : 1;  // default filling
            int fill0 = (has1 | has0) || (3 - (has1 + has0) <= 1)
                ? fill : 1 - fill;  // first one, unique
            for (int fi = 0; fi < 4; fi++)
                if (qti[fi] == -1) {
                    if (idx[fi] == 0b111) qti[fi] = 0;  // the base
                    else if (fill0 != fill) qti[fi] = fill0, fill0 = fill;  // unique
                    else qti[fi] = fill;  // default
                }
        }
        // only one way to divide
        else {
            for (int fi = 0; fi < 4; fi++)
                qti[fi] = max(qti[fi], 0);
        }
        // put face
        for (int fi = 0; fi < 4; fi++) {
            auto LutF = LUT_MARCH_F[idx[fi]];
            fMap[f[fi]] = LutF[0][0] == -1 ?
                nullptr : LutF[qti[fi]];
        }
    }

    // add tets
    for (ivec4 tet : tets) {
        // lookup edges
        int es[10] = {
            vMap[tet.x], vMap[tet.y], vMap[tet.z], vMap[tet.w],
            -1, -1, -1, -1, -1, -1
        };
        int idx = calcIndex(tet);
        for (int _ = 0; _ < 6; _++) {
            if (LUT_MARCH_E[idx][_]) {
                int i = tet[EDGES[_][0]], j = tet[EDGES[_][1]];
                if (i > j) std::swap(i, j);
                es[_ + 4] = eMap[ivec2(i, j)];
            }
        }
        // get a list of must-have faces
        ivec3 reqFaces[8];
        int facesN = 0;
        for (int fi = 0; fi < 4; fi++) {
            ivec3 f;
            for (int _ = 0; _ < 3; _++)
                f[_] = tet[FACES[fi][_]];
            f = rotateIvec3(f);
            const int* fs = fMap[f];
            if (!fs) continue;
            for (int vi = 0; vi < 6; vi += 3) {
                if (fs[vi] == -1) break;
                for (int _ = 0; _ < 3; _++)
                    f[_] = FACES[fi][fs[vi + _]];
                reqFaces[facesN] = rotateIvec3(f);
                facesN += 1;
            }
        }
        // find a tet combination that meets all faces
        const int* LutTBest = nullptr;
        bool found = false;
        for (int tsi = 0; tsi < 6; tsi++) {
            auto LutT = LUT_MARCH_T[idx][tsi];
            if (LutT[0] == -1) break;
            bool meetFaces[8] = { 0,0,0,0,0,0,0,0 };
            int meetCount = 0;
            for (int ti = 0; ti < 12; ti += 4) {
                const int* t = &LutT[ti];
                if (*t == -1) break;
                for (int fi = 0; fi < 4; fi++) {
                    ivec3 f;
                    for (int _ = 0; _ < 3; _++)
                        f[_] = t[FACES[fi][_]];
                    f = rotateIvec3(f);
                    for (int _ = 0; _ < facesN; _++)
                        if (f == reqFaces[_]) {
                            assert(!meetFaces[_]);
                            meetFaces[_] = true;
                            meetCount += 1;
                        }
                }
            }
            if (meetCount == facesN) {
                LutTBest = LutT;
                found = true;
                break;
            }
        }
        // add tets
        if (LutTBest) {
            for (int i = 0; i < 12; i += 4) {
                if (LutTBest[i] == -1) break;
                ivec4 tet1;
                for (int _ = 0; _ < 4; _++) {
                    tet1[_] = es[LutTBest[i + _]];
                    assert(tet1[_] != -1);
                }
                resTets.push_back(tet1);
            }
        }
        // not found - create a new vertex
        else if (!found) {
            assert(facesN == 7);
            // add a vertex in the middle
            es[idx == 0b0111 ? 3 : idx == 0b1011 ? 2 :
                idx == 0b1101 ? 1 : idx == 0b1110 ? 0 : -1] = -1;
            vec3 mid(0); int count = 0;
            for (int i = 0; i < 10; i++)
                if (es[i] != -1)
                    mid += resVerts[es[i]], count++;
            assert(count == 6);
            mid = mid * (1.0 / count);
            int vn = (int)resVerts.size();
            resVerts.push_back(mid);
            // construct tets
            for (int i = 0; i < facesN; i++) {
                ivec3 t = reqFaces[i];
                resTets.push_back(ivec4(vn,
                    es[t[0]], es[t[1]], es[t[2]]));
            }
            ivec3 t = idx == 0b1110 ? ivec3(4, 6, 5)
                : idx == 0b1101 ? ivec3(4, 7, 8)
                : idx == 0b1011 ? ivec3(7, 5, 9)
                : idx == 0b0111 ? ivec3(6, 8, 9) : ivec3(-1);
            resTets.push_back(ivec4(vn,
                es[t[0]], es[t[1]], es[t[2]]));
        }
    }
}


MESHGEN_TET_IMPLICIT_NS_END
