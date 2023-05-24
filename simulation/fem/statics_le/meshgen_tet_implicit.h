// Generate a tetrahedral mesh from an implicitly defined object
// Defined by: F(x,y,z) < 0

#pragma once

#include <stdio.h>
#include "meshgen_misc.h"

#if SUPPRESS_ASSERT
#undef assert
#define assert(x) 0
#endif

#define MESHGEN_TET_IMPLICIT_NS_START namespace MeshgenTetImplicit {
#define MESHGEN_TET_IMPLICIT_NS_END }

MESHGEN_TET_IMPLICIT_NS_START

using namespace MeshgenMisc;

namespace MeshgenTetLoss {
#include "meshgen_tet_loss.h"
}



/* Mesh Generator 1 */

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
const int EDGE_FACES[6][2] = {  // no particular order
    {2, 3}, {1, 3}, {1, 2},
    {0, 3}, {0, 2}, {0, 1}
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


// Cut the list of tetrahedra generated by marching to construct the isosurface
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


// merge vertices that are too close produced by isosurface cutting
// WARNING: can result in tets with negative volumes; Fix it by compression
void mergeCloseSurfaceVertices(
    const std::vector<vec3>& verts, const std::vector<ivec4>& tets,
    std::vector<vec3>& resVerts, std::vector<ivec4>& resTets,
    double kf = 0.1
) {
    assert(resVerts.empty() && resTets.empty());
    DisjointSet dsj((int)verts.size());

    // get faces
    std::set<ivec3, decltype(ivec3Cmp)> faces_s(ivec3Cmp);
    for (ivec4 tet : tets) {
        for (int fi = 0; fi < 4; fi++) {
            ivec3 f;
            for (int _ = 0; _ < 3; _++)
                f[_] = tet[FACES[fi][_]];
            f = rotateIvec3(f);
            assert(faces_s.find(f) == faces_s.end());
            ivec3 fo = ivec3(f.x, f.z, f.y);
            if (faces_s.find(fo) != faces_s.end())
                faces_s.erase(fo);
            else faces_s.insert(f);
        }
    }
    std::vector<ivec3> faces(faces_s.begin(), faces_s.end());

    // get surface vertices
    std::vector<bool> isSurface(verts.size(), false);
    for (ivec3 f : faces) {
        for (int i = 0; i < 3; i++)
            isSurface[f[i]] = true;
    }

    // merge faces
    for (ivec3 f : faces) {
        std::pair<int, double> el[3];  // index, length
        for (int i = 0; i < 3; i++)
            el[i] = std::pair<int, double>(i,
                length(verts[f[(i + 1) % 3]] - verts[f[i]]));
        std::sort(el, el + 3,
            [](std::pair<int, double> a, std::pair<int, double> b) {
                return a.second < b.second;
            });
        if (el[0].second < kf * el[1].second)
            dsj.unionSet(f[el[0].first], f[(el[0].first + 1) % 3]);
    }

    // get merged surface
    std::vector<int> imap;
    int resN = dsj.getCompressedMap(imap);
    resVerts = std::vector<vec3>(resN, vec3(0));
    std::vector<int> resCount(resN, 0);
    for (int i = 0; i < (int)verts.size(); i++)
        resVerts[imap[i]] += verts[i], resCount[imap[i]]++;
    for (int i = 0; i < resN; i++)
        resVerts[i] *= (1.0 / resCount[i]);

    // merge "exposed" tetrahedra
    auto isUnique = [](ivec4 t) {
        return t[0] != t[1] && t[0] != t[2] && t[0] != t[3] &&
            t[1] != t[2] && t[1] != t[3] && t[2] != t[3];
    };
    for (ivec4 tet : tets) {
        int t[4];
        for (int i = 0; i < 4; i++)
            t[i] = imap[tet[i]];
        if (!isUnique(t)) continue;
        if (!(det(resVerts[t[1]] - resVerts[t[0]],
            resVerts[t[2]] - resVerts[t[0]],
            resVerts[t[3]] - resVerts[t[0]]) > 0.0)) {
            for (int i = 1; i < 4; i++)
                dsj.unionSet(tet[0], tet[i]);
        }
    }

    // get result
    resN = dsj.getCompressedMap(imap);
    resVerts = std::vector<vec3>(resN, vec3(0));
    resCount = std::vector<int>(resN, 0);
    for (int i = 0; i < (int)verts.size(); i++)
        if (isSurface[i])
            resVerts[imap[i]] += verts[i], resCount[imap[i]]++;
    for (int i = 0; i < (int)verts.size(); i++)
        if (resCount[imap[i]] == 0)
            resVerts[imap[i]] += verts[i], resCount[imap[i]]++;
    for (int i = 0; i < resN; i++)
        resVerts[i] *= (1.0 / resCount[i]);
    for (ivec4 tet : tets) {
        for (int i = 0; i < 4; i++)
            tet[i] = imap[tet[i]];
        if (isUnique(tet)) {
            bool positiveVolume = det(resVerts[tet[1]] - resVerts[tet[0]],
                resVerts[tet[2]] - resVerts[tet[0]],
                resVerts[tet[3]] - resVerts[tet[0]]) > 0.0;
            // if (1) assert(positiveVolume);
            // else if (!positiveVolume) continue;
            resTets.push_back(tet);
        }
    }
}




/* Mesh Generator 2 */


// March through a BCC crystalline lattice
void generateTetrahedraBCC(
    ScalarFieldF F, vec3 b0, vec3 b1, ivec3 bn,
    std::vector<vec3> &vertices, std::vector<ivec4> &tets,
    std::vector<int> &constraintI, std::vector<vec3> &constraintN
) {
    assert(bn.x >= 1 && bn.y >= 1 && bn.z >= 1);
    assert(bn.x <= 1000 && bn.y <= 1000 && bn.z <= 1000); // prevent overflow

    // verts
    auto vci = [=](int i, int j, int k) {  // cube vertex index
        return (k * (bn[1] + 1) + j) * (bn[0] + 1) + i;
    };
    const int vcn = (bn[0] + 1) * (bn[1] + 1) * (bn[2] + 1);
    auto vbi = [=](int i, int j, int k) {  // cube body index
        return vcn + (k * bn[1] + j) * bn[0] + i;
    };
    int vn = vcn + bn[0] * bn[1] * bn[2];
    std::vector<vec3> verts(vn);
    constraintI.clear(), constraintN.clear();
    for (int k = 0; k <= bn[2]; k++)
        for (int j = 0; j <= bn[1]; j++)
            for (int i = 0; i <= bn[0]; i++)
                verts[vci(i, j, k)] = mix(b0, b1, vec3(i, j, k) / vec3(bn));
    for (int k = 0; k < bn[2]; k++)
        for (int j = 0; j < bn[1]; j++)
            for (int i = 0; i < bn[0]; i++)
                verts[vbi(i, j, k)] = mix(b0, b1, (vec3(i, j, k) + vec3(0.5)) / vec3(bn));
    std::vector<double> vals(vn);
    for (int i = 0; i < vn; i++) {
        vec3 p = verts[i];
        vals[i] = F(p.x, p.y, p.z);
    }
    
    // constraints
    std::vector<bool> isOnFaceX(vn, false),
        isOnFaceY(vn, false), isOnFaceZ(vn, false),
        isOnEdge(vn, false);
    std::vector<bool> toConstraintX(vn, false),
        toConstraintY(vn, false), toConstraintZ(vn, false),
        toConstraintEdge(vn, false);
    std::vector<bool> *isOnFace[3] = {
        &isOnFaceX, &isOnFaceY, &isOnFaceZ
    };
    std::vector<bool> *toConstraint[3] = {
        &toConstraintX, &toConstraintY, &toConstraintZ
    };
    for (int k = 0; k <= bn[2]; k++)
        for (int j = 0; j <= bn[1]; j++)
            for (int i = 0; i <= bn[0]; i++) {
                int _ = vci(i, j, k);
                if (i == 0 || i == bn[0])
                    isOnFaceX[_] = true;
                if (j == 0 || j == bn[1])
                    isOnFaceY[_] = true;
                if (k == 0 || k == bn[2])
                    isOnFaceZ[_] = true;
                isOnEdge[_] = int(isOnFaceX[_] + isOnFaceY[_] + isOnFaceZ[_]) >= 2;
            }
    for (int dim = 0; dim < 3; dim++)
        for (int i = 0; i < vn; i++)
            if (vals[i] < 0.0 && isOnFace[dim]->at(i))
                toConstraint[dim]->at(i) = true;

    // volumetric tets
    tets.clear();
    auto calcTet = [&](int t1, int t2, int t3, int t4) {
        ivec4 t(t1, t2, t3, t4);
        int inCount = 0, cCount = 0;
        for (int _ = 0; _ < 4; _++) {
            inCount += int(vals[t[_]] < 0.0);
            cCount += int(toConstraintX[t[_]] || toConstraintY[t[_]] || toConstraintZ[t[_]]);
        }
        return ivec2(inCount, cCount);
    };
    auto testTet = [&](ivec2 p) {
        return p.x - p.y >= 1;
    };
    auto addTet = [&](int t1, int t2, int t3, int t4) {
        ivec2 p = calcTet(t1, t2, t3, t4);
        if (testTet(p))
            tets.push_back(ivec4(t1, t2, t3, t4));
    };
    for (int k = 0; k < bn[2]; k++)
        for (int j = 0; j < bn[1]; j++)
            for (int i = 0; i < bn[0]; i++) {
                if (i+1 < bn[0]) {
                    addTet(vbi(i, j, k), vbi(i+1, j, k), vci(i+1, j+1, k+1), vci(i+1, j, k+1));
                    addTet(vbi(i, j, k), vbi(i+1, j, k), vci(i+1, j, k+1), vci(i+1, j, k));
                    addTet(vbi(i, j, k), vbi(i+1, j, k), vci(i+1, j, k), vci(i+1, j+1, k));
                    addTet(vbi(i, j, k), vbi(i+1, j, k), vci(i+1, j+1, k), vci(i+1, j+1, k+1));
                }
                if (j+1 < bn[1]) {
                    addTet(vbi(i, j, k), vbi(i, j+1, k), vci(i, j+1, k+1), vci(i+1, j+1, k+1));
                    addTet(vbi(i, j, k), vbi(i, j+1, k), vci(i+1, j+1, k+1), vci(i+1, j+1, k));
                    addTet(vbi(i, j, k), vbi(i, j+1, k), vci(i+1, j+1, k), vci(i, j+1, k));
                    addTet(vbi(i, j, k), vbi(i, j+1, k), vci(i, j+1, k), vci(i, j+1, k+1));
                }
                if (k+1 < bn[2]) {
                    addTet(vbi(i, j, k), vbi(i, j, k+1), vci(i+1, j+1, k+1), vci(i, j+1, k+1));
                    addTet(vbi(i, j, k), vbi(i, j, k+1), vci(i, j+1, k+1), vci(i, j, k+1));
                    addTet(vbi(i, j, k), vbi(i, j, k+1), vci(i, j, k+1), vci(i+1, j, k+1));
                    addTet(vbi(i, j, k), vbi(i, j, k+1), vci(i+1, j, k+1), vci(i+1, j+1, k+1));
                }
            }

    // boundary "wedges"
    struct wedge {
        int t0, t1, t2, t3, t4;
        union {
            int _;
            struct {
                uint8_t preferred;
                uint8_t test1;  // preferred == 1
                uint8_t test2;  // preferred == 0
            };
        };
    };
    // add wedge to the buffer before applying them
    std::vector<wedge> wedgeBuffer;
    auto pushWedge = [&](int t0, int t1, int t2, int t3, int t4) {
        ivec2 t11 = calcTet(t0, t1, t2, t3);
        ivec2 t12 = calcTet(t0, t1, t3, t4);
        ivec2 t21 = calcTet(t0, t1, t2, t4);
        ivec2 t22 = calcTet(t0, t2, t3, t4);
        ivec2 test1 = t11 + t12, test2 = t21 + t22;
        vec3 c = ((verts[t0] - b0) / (b1 - b0) + vec3(0.5)) * vec3(vn);
        int preferred = (int)(test1.y < test2.y ||
            (test1.y == test2.y && ((int)c.x+(int)c.y+(int)c.z)&1)
        );
        wedge w{ t0, t1, t2, t3, t4 };
        w.preferred = (uint8_t)preferred;
        w.test1 = (uint8_t)testTet(t11) + (uint8_t)testTet(t12);
        w.test2 = (uint8_t)testTet(t21) + (uint8_t)testTet(t22);
        if (w.test1 > 0 || w.test2 > 0)
            wedgeBuffer.push_back(w);
    };
    for (int j = 0; j < bn[1]; j++)
        for (int i = 0; i < bn[0]; i++) {
            pushWedge(vbi(i, j, 0),
                vci(i+1, j+1, 0), vci(i+1, j, 0), vci(i, j, 0), vci(i, j+1, 0));
            pushWedge(vbi(i, j, bn[2]-1),
                vci(i+1, j+1, bn[2]), vci(i, j+1, bn[2]), vci(i, j, bn[2]), vci(i+1, j, bn[2]));
        }
    for (int k = 0; k < bn[2]; k++)
        for (int i = 0; i < bn[0]; i++) {
            pushWedge(vbi(i, 0, k),
                vci(i, 0, k), vci(i+1, 0, k), vci(i+1, 0, k+1), vci(i, 0, k+1));
            pushWedge(vbi(i, bn[1]-1, k),
                vci(i, bn[1], k), vci(i, bn[1], k+1), vci(i+1, bn[1], k+1), vci(i+1, bn[1], k));
        }
    for (int k = 0; k < bn[2]; k++)
        for (int j = 0; j < bn[1]; j++) {
            pushWedge(vbi(0, j, k),
                vci(0, j+1, k+1), vci(0, j+1, k), vci(0, j, k), vci(0, j, k+1));
            pushWedge(vbi(bn[0]-1, j, k),
                vci(bn[0], j, k), vci(bn[0], j+1, k), vci(bn[0], j+1, k+1), vci(bn[0], j, k+1));
        }

    // handle wedges
    auto ivec2Cmp = [](ivec2 a, ivec2 b) {
        return a.x == b.x ? a.y < b.y : a.x < b.x;
    };
    std::set<ivec2, decltype(ivec2Cmp)> surfEdges(ivec2Cmp);
    auto addSurfEdge = [&](int a, int b) {
        if (a > b) std::swap(a, b);
        if (surfEdges.find(ivec2(a, b)) != surfEdges.end())
            surfEdges.erase(ivec2(a, b));
        else surfEdges.insert(ivec2(a, b));
    };
    for (wedge w : wedgeBuffer) {
        addSurfEdge(w.t1, w.t2);
        addSurfEdge(w.t2, w.t3);
        addSurfEdge(w.t3, w.t4);
        addSurfEdge(w.t4, w.t1);
    }
    std::vector<bool> isOnSurface(vn, false);
    for (ivec2 p : surfEdges)
        isOnSurface[p.x] = isOnSurface[p.y] = true;
    for (wedge w : wedgeBuffer) {
        int t0 = w.t0, t1 = w.t1, t2 = w.t2, t3 = w.t3, t4 = w.t4;
        ivec4 ttets[2][2] = {
            { ivec4(t0, t1, t2, t4), ivec4(t0, t2, t3, t4) },
            { ivec4(t0, t1, t2, t3), ivec4(t0, t1, t3, t4) }
        };
        auto isFeasible = [&](ivec4* t2) {
            bool feasible = true;
            for (int i = 0; i < 2; i++) {
                if (isOnSurface[t2[i][1]] &&
                    isOnSurface[t2[i][2]] &&
                    isOnSurface[t2[i][3]]) feasible = false;
            }
            return feasible;
        };
        auto addTets = [&](ivec4* t2) {
            for (int i = 0; i < 2; i++) {
                addTet(t2[i][0], t2[i][1], t2[i][2], t2[i][3]);
            }
        };
        // preferred way
        ivec4 *preferred = ttets[w.preferred];
        if (isFeasible(preferred)) {
            addTets(preferred);
            continue;
        }
        // try the other way
        preferred = ttets[1 - w.preferred];
        if (isFeasible(preferred)) {
            addTets(preferred);
            continue;
        }
        addTets(ttets[w.preferred]); continue;
        // if there is only one constrained edge, free the other vertices
        bool noEdge[4] = { true, true, true, true };
        int t[4] = { t1, t2, t3, t4 };
        for (int i = 0; i < 4; i++) {
            int a = t[i], b = t[(i+1)%4];
            if (a > b) std::swap(a, b);
            if (surfEdges.find(ivec2(a, b)) == surfEdges.end())
                noEdge[i] = noEdge[(i+1)%4] = false;
        }
        int noEdgeCount = 0;
        for (int i = 0; i < 4; i++)
            noEdgeCount += (int)noEdge[i];
        if (noEdgeCount % 2 == 0) {
            for (int i = 0; i < 4; i++) if (noEdge[i]) {
                for (int dim = 0; dim < 3; dim++)
                    isOnFace[dim]->at(t[i]) = false;
            }
            addTets(ttets[w.preferred]);
        }
    }

    // face constraints
    for (ivec4 t : tets) {
        for (int dim = 0; dim < 3; dim++) {
            int niof = 0, nin = 0, vi[4];
            for (int _ = 0; _ < 4; _++) {
                if (isOnFace[dim]->at(t[_])) {
                    vi[niof++] = t[_];
                    nin += int(vals[t[_]] < 0.0);
                }
            }
            if (niof != 3) continue;
            if (nin > 0)
                for (int _ = 0; _ < niof; _++)
                    toConstraint[dim]->at(vi[_]) = true;
        }
    }
    // edge constraints
    for (ivec4 t : tets) {
        int nioe = 0, nin = 0, vi[4];
        for (int _ = 0; _ < 4; _++) {
            if (isOnEdge[t[_]]) {
                vi[nioe++] = t[_];
                nin += int(vals[t[_]] < 0.0);
            }
        }
        if (nioe == 2 && nin > 0)
            for (int _ = 0; _ < nioe; _++)
                toConstraintEdge[vi[_]] = true;
    }
    for (int i = 0; i < vn; i++)
        if (isOnEdge[i] && !toConstraintEdge[i])
            for (int dim = 0; dim < 3; dim++)
                toConstraint[dim]->at(i) = false;
    // vertex constraints
    for (int i = 0; i < vn; i++)
        if (isOnFaceX[i] && isOnFaceY[i] && isOnFaceZ[i] && vals[i] < 0.0)
            for (int dim = 0; dim < 3; dim++)
                toConstraint[dim]->at(i) = true;
    // apply
    for (int k = 0; k <= bn[2]; k++)
        for (int j = 0; j <= bn[1]; j++)
            for (int i = 0; i <= bn[0]; i++) {
                int _ = vci(i, j, k);
                if (toConstraintX[_])
                    constraintI.push_back(_), constraintN.push_back(vec3(1, 0, 0));
                if (toConstraintY[_])
                    constraintI.push_back(_), constraintN.push_back(vec3(0, 1, 0));
                if (toConstraintZ[_])
                    constraintI.push_back(_), constraintN.push_back(vec3(0, 0, 1));
            }

    // remove unused vertices
    std::vector<int> vmap(vn, 0);
    for (ivec4 t : tets)
        for (int _ = 0; _ < 4; _++)
            vmap[t[_]] = 1;
    vertices.clear();
    for (int i = 0; i < vn; i++) {
        if (vmap[i]) {
            vmap[i] = (int)vertices.size();
            vertices.push_back(verts[i]);
        }
        else vmap[i] = -1;
    }
    for (int i = 0; i < (int)tets.size(); i++) {
        for (int _ = 0; _ < 4; _++) {
            tets[i][_] = vmap[tets[i][_]];
            assert(tets[i][_] >= 0);
        }
    }
    int j = 0;
    for (int i = 0; i < (int)constraintI.size(); i++) {
        constraintI[j] = vmap[constraintI[i]];
        if (constraintI[j] != -1) j++;
    }
    constraintI.resize(j);
    constraintN.resize(j);
}





/* Mesh Optimizer */


// Lookup table for smaller tetrahedra after edge split
const int LUT_EDGE_SPLIT[64][8][32] = {
    { {0,1,2,3,-1}, {-1} },
    { {0,2,3,4,1,2,4,3,-1}, {-1} },
    { {0,1,5,3,1,2,5,3,-1}, {-1} },
    { {0,3,4,5,1,2,4,3,2,3,5,4,-1}, {0,3,4,5,1,2,5,3,1,3,5,4,-1}, {-1} },
    { {0,1,2,6,1,2,6,3,-1}, {-1} },
    { {0,2,6,4,1,2,4,3,2,3,6,4,-1}, {0,2,6,4,1,2,6,3,1,2,4,6,-1}, {-1} },
    { {0,1,5,6,1,2,5,3,1,3,5,6,-1}, {0,1,5,6,1,2,6,3,1,2,5,6,-1}, {-1} },
    { {0,4,5,6,1,4,6,5,1,2,6,3,1,2,5,6,-1}, {0,4,5,6,2,4,6,5,1,2,6,3,1,2,4,6,-1}, {0,4,5,6,1,4,6,5,1,2,5,3,1,3,5,6,-1}, {0,4,5,6,1,2,5,3,3,4,6,5,1,3,5,4,-1}, {0,4,5,6,1,2,4,3,2,3,6,4,2,4,6,5,-1}, {0,4,5,6,1,2,4,3,3,4,6,5,2,3,5,4,-1}, {-1} },
    { {0,1,7,3,0,2,3,7,-1}, {-1} },
    { {0,2,3,4,1,3,7,4,2,3,4,7,-1}, {0,2,3,7,0,3,4,7,1,3,7,4,-1}, {-1} },
    { {0,1,5,3,1,3,7,5,2,3,5,7,-1}, {0,1,7,3,0,3,7,5,2,3,5,7,-1}, {-1} },
    { {0,3,4,5,2,3,5,7,1,3,7,4,3,4,5,7,-1}, {-1} },
    { {0,1,7,6,0,2,6,7,1,3,7,6,2,3,6,7,-1}, {-1} },
    { {0,2,6,4,1,3,7,4,2,3,6,4,2,3,4,7,-1}, {-1} },
    { {0,1,5,6,1,3,5,6,1,3,7,5,2,3,5,7,-1}, {-1} },
    { {0,4,5,6,2,5,7,6,4,5,6,7,1,3,7,6,1,4,6,7,2,3,6,7,-1}, {0,4,7,6,0,5,6,7,2,5,7,6,1,3,7,6,1,4,6,7,2,3,6,7,-1}, {-1} },
    { {0,1,2,8,0,2,3,8,-1}, {-1} },
    { {0,2,3,4,1,2,4,8,2,3,4,8,-1}, {0,2,3,8,0,2,8,4,1,2,4,8,-1}, {-1} },
    { {0,1,5,8,0,3,8,5,1,2,5,8,2,3,5,8,-1}, {-1} },
    { {0,3,4,5,1,2,4,8,2,3,5,4,2,3,4,8,-1}, {-1} },
    { {0,1,2,6,1,2,6,8,2,3,6,8,-1}, {0,1,2,8,0,2,6,8,2,3,6,8,-1}, {-1} },
    { {0,2,6,4,2,3,6,8,1,2,4,8,2,4,8,6,-1}, {-1} },
    { {0,1,5,6,2,3,6,8,1,2,5,6,1,2,6,8,-1}, {-1} },
    { {0,4,5,6,4,5,6,8,1,2,5,8,3,5,8,6,1,4,8,5,2,3,5,8,-1}, {0,4,5,8,0,5,6,8,1,2,5,8,3,5,8,6,1,4,8,5,2,3,5,8,-1}, {-1} },
    { {0,1,7,8,0,2,3,7,0,3,8,7,-1}, {0,1,7,8,0,2,3,8,0,2,8,7,-1}, {-1} },
    { {1,4,8,7,0,2,3,8,0,2,8,7,0,4,7,8,-1}, {1,4,8,7,0,2,3,7,0,3,8,7,0,4,7,8,-1}, {1,4,8,7,0,2,3,7,3,4,7,8,0,3,4,7,-1}, {0,2,3,4,1,4,8,7,2,3,4,8,2,4,7,8,-1}, {0,2,3,4,2,3,4,7,3,4,7,8,1,4,8,7,-1}, {2,4,7,8,1,4,8,7,0,2,3,8,0,2,8,4,-1}, {-1} },
    { {0,1,7,8,0,3,7,5,0,3,8,7,2,3,5,7,-1}, {-1} },
    { {2,5,7,8,0,3,8,5,0,4,5,8,1,4,8,5,1,5,8,7,2,3,5,8,-1}, {2,5,7,8,0,3,8,5,0,4,5,8,4,5,8,7,1,4,8,7,2,3,5,8,-1}, {-1} },
    { {2,3,6,8,0,1,7,8,0,2,6,8,0,2,8,7,-1}, {-1} },
    { {0,2,6,7,0,4,7,6,3,6,7,8,4,6,8,7,1,4,8,7,2,3,6,7,-1}, {0,2,6,7,0,4,7,6,3,6,7,8,1,4,6,7,1,6,8,7,2,3,6,7,-1}, {-1} },
    { {2,5,7,6,0,1,7,6,0,5,6,7,3,6,7,8,1,6,8,7,2,3,6,7,-1}, {0,1,5,8,2,5,7,8,0,5,6,8,3,5,8,6,1,5,8,7,2,3,5,8,-1}, {-1} },
    { {0,4,7,6,2,5,7,6,0,5,6,7,2,3,6,7,3,6,7,8,1,4,8,7,4,6,8,7,-1}, {0,4,5,8,0,5,6,8,2,3,5,8,3,5,8,6,1,4,8,7,4,5,8,7,2,5,7,8,-1}, {0,4,7,6,0,5,6,7,2,3,6,7,3,6,7,8,1,4,6,7,2,5,7,6,1,6,8,7,-1}, {0,4,5,6,2,5,7,6,2,3,6,7,3,6,7,8,4,5,6,7,1,4,8,7,4,6,8,7,-1}, {0,4,5,6,2,3,5,8,3,5,8,6,4,5,6,8,1,4,8,7,4,5,8,7,2,5,7,8,-1}, {0,4,5,8,0,5,6,8,2,3,5,8,3,5,8,6,1,4,8,5,1,5,8,7,2,5,7,8,-1}, {0,4,5,6,1,4,6,7,2,3,6,7,3,6,7,8,4,5,6,7,2,5,7,6,1,6,8,7,-1}, {0,4,5,6,2,3,5,8,3,5,8,6,1,4,8,5,4,5,6,8,1,5,8,7,2,5,7,8,-1} },
    { {0,1,2,9,0,1,9,3,-1}, {-1} },
    { {0,2,9,4,0,3,4,9,1,2,4,9,1,3,9,4,-1}, {-1} },
    { {0,1,5,3,1,2,5,9,1,3,9,5,-1}, {0,1,9,3,0,1,5,9,1,2,5,9,-1}, {-1} },
    { {0,3,4,5,1,2,5,9,1,3,5,4,1,3,9,5,-1}, {-1} },
    { {0,1,2,6,1,2,6,9,1,3,9,6,-1}, {0,1,2,9,0,1,9,6,1,3,9,6,-1}, {-1} },
    { {0,2,6,4,1,2,4,6,1,2,6,9,1,3,9,6,-1}, {-1} },
    { {0,1,5,6,1,3,9,6,1,2,5,9,1,5,6,9,-1}, {-1} },
    { {0,4,5,6,2,4,9,5,1,2,4,9,3,4,6,9,4,5,6,9,1,3,9,4,-1}, {0,4,5,9,0,4,9,6,2,4,9,5,1,2,4,9,3,4,6,9,1,3,9,4,-1}, {-1} },
    { {0,1,7,3,0,2,9,7,0,3,7,9,-1}, {0,1,9,3,0,1,7,9,0,2,9,7,-1}, {-1} },
    { {0,2,9,7,0,3,4,7,0,3,7,9,1,3,7,4,-1}, {-1} },
    { {0,1,7,3,2,5,7,9,3,5,9,7,0,3,7,5,-1}, {0,1,5,3,1,3,9,5,1,5,9,7,2,5,7,9,-1}, {0,1,9,3,2,5,7,9,0,1,7,9,0,5,9,7,-1}, {0,1,5,3,1,3,7,5,2,5,7,9,3,5,9,7,-1}, {1,5,9,7,2,5,7,9,0,1,9,3,0,1,5,9,-1}, {0,1,7,3,2,5,7,9,0,3,7,9,0,5,9,7,-1}, {-1} },
    { {2,4,9,5,2,4,7,9,0,3,4,9,0,4,5,9,1,3,9,4,1,4,9,7,-1}, {0,3,4,9,2,5,7,9,0,4,5,9,4,5,9,7,1,3,9,4,1,4,9,7,-1}, {-1} },
    { {0,1,9,6,0,1,7,9,0,2,9,7,1,3,9,6,-1}, {-1} },
    { {0,2,6,7,2,6,7,9,0,4,7,6,3,6,9,7,1,3,7,6,1,4,6,7,-1}, {0,2,9,4,2,4,7,9,0,4,9,6,3,4,6,9,1,3,9,4,1,4,9,7,-1}, {-1} },
    { {0,1,7,6,2,5,7,9,0,5,6,7,3,6,9,7,5,6,7,9,1,3,7,6,-1}, {2,5,7,6,0,1,7,6,2,6,7,9,0,5,6,7,3,6,9,7,1,3,7,6,-1}, {-1} },
    { {0,4,7,6,1,4,6,7,0,5,6,7,3,6,9,7,5,6,7,9,1,3,7,6,2,5,7,9,-1}, {0,4,5,6,1,4,9,7,3,4,6,9,1,3,9,4,4,5,6,9,4,5,9,7,2,5,7,9,-1}, {0,4,5,6,2,4,9,5,1,4,9,7,3,4,6,9,1,3,9,4,4,5,6,9,2,4,7,9,-1}, {2,6,7,9,0,4,7,6,1,4,6,7,0,5,6,7,2,5,7,6,3,6,9,7,1,3,7,6,-1}, {0,4,5,6,2,6,7,9,1,4,6,7,2,5,7,6,3,6,9,7,4,5,6,7,1,3,7,6,-1}, {0,4,5,6,1,4,6,7,3,6,9,7,4,5,6,7,5,6,7,9,1,3,7,6,2,5,7,9,-1}, {0,4,5,9,0,4,9,6,1,4,9,7,3,4,6,9,2,4,9,5,1,3,9,4,2,4,7,9,-1}, {0,4,5,9,0,4,9,6,1,4,9,7,3,4,6,9,1,3,9,4,4,5,9,7,2,5,7,9,-1} },
    { {0,1,2,8,0,2,9,8,0,3,8,9,-1}, {0,1,2,9,0,1,9,8,0,3,8,9,-1}, {-1} },
    { {0,2,8,4,0,2,9,8,0,3,8,9,1,2,4,8,-1}, {-1} },
    { {0,1,5,9,0,1,9,8,0,3,8,9,1,2,5,9,-1}, {-1} },
    { {0,3,8,5,2,5,8,9,0,4,5,8,3,5,9,8,1,2,5,8,1,4,8,5,-1}, {0,3,4,9,2,4,9,5,0,4,5,9,3,4,9,8,1,2,4,9,1,4,8,9,-1}, {-1} },
    { {0,1,2,8,2,6,8,9,3,6,9,8,0,2,6,8,-1}, {0,1,2,9,0,1,9,8,3,6,9,8,0,6,8,9,-1}, {1,6,8,9,0,1,2,9,0,1,9,6,3,6,9,8,-1}, {0,1,2,8,3,6,9,8,0,2,9,8,0,6,8,9,-1}, {0,1,2,6,1,2,6,8,2,6,8,9,3,6,9,8,-1}, {0,1,2,6,1,2,6,9,1,6,8,9,3,6,9,8,-1}, {-1} },
    { {0,2,9,4,3,4,6,9,0,4,9,6,3,4,9,8,1,2,4,9,1,4,8,9,-1}, {0,2,9,4,0,4,9,6,3,6,9,8,4,6,8,9,1,2,4,9,1,4,8,9,-1}, {-1} },
    { {0,1,5,8,2,5,8,9,0,5,6,8,3,6,9,8,5,6,8,9,1,2,5,8,-1}, {0,1,5,8,2,5,8,9,3,5,8,6,0,5,6,8,3,5,9,8,1,2,5,8,-1}, {-1} },
    { {0,4,5,9,0,4,9,6,1,4,8,9,3,6,9,8,1,2,4,9,2,4,9,5,4,6,8,9,-1}, {2,5,8,9,0,4,5,8,1,4,8,5,0,5,6,8,3,6,9,8,1,2,5,8,5,6,8,9,-1}, {0,4,5,6,2,5,8,9,1,4,8,5,3,5,8,6,3,5,9,8,4,5,6,8,1,2,5,8,-1}, {0,4,5,6,1,4,8,9,3,4,6,9,3,4,9,8,2,4,9,5,1,2,4,9,4,5,6,9,-1}, {0,4,5,9,0,4,9,6,1,4,8,9,3,4,6,9,3,4,9,8,1,2,4,9,2,4,9,5,-1}, {0,4,5,6,2,5,8,9,1,4,8,5,3,6,9,8,4,5,6,8,1,2,5,8,5,6,8,9,-1}, {2,5,8,9,0,4,5,8,1,4,8,5,0,5,6,8,3,5,8,6,3,5,9,8,1,2,5,8,-1}, {0,4,5,6,1,4,8,9,2,4,9,5,3,6,9,8,1,2,4,9,4,5,6,9,4,6,8,9,-1} },
    { {0,1,7,8,0,3,8,9,0,2,9,7,0,7,9,8,-1}, {-1} },
    { {0,2,9,4,2,4,7,9,4,7,9,8,0,3,4,9,3,4,9,8,1,4,8,7,-1}, {1,4,8,9,0,2,9,4,2,4,7,9,0,3,4,9,3,4,9,8,1,4,9,7,-1}, {-1} },
    { {0,1,5,8,2,5,7,8,2,5,8,9,0,3,8,5,3,5,9,8,1,5,8,7,-1}, {0,1,5,8,2,5,7,9,5,7,9,8,0,3,8,5,3,5,9,8,1,5,8,7,-1}, {-1} },
    { {0,3,4,9,1,4,8,7,3,4,9,8,0,4,5,9,4,5,9,7,4,7,9,8,2,5,7,9,-1}, {0,3,4,9,1,4,8,7,3,4,9,8,0,4,5,9,2,4,9,5,4,7,9,8,2,4,7,9,-1}, {2,5,8,9,1,4,8,7,0,3,8,5,0,4,5,8,3,5,9,8,4,5,8,7,2,5,7,8,-1}, {2,5,8,9,1,4,8,5,0,3,8,5,1,5,8,7,0,4,5,8,3,5,9,8,2,5,7,8,-1}, {1,4,8,7,0,3,8,5,0,4,5,8,3,5,9,8,4,5,8,7,5,7,9,8,2,5,7,9,-1}, {1,4,8,5,0,3,8,5,1,5,8,7,0,4,5,8,3,5,9,8,5,7,9,8,2,5,7,9,-1}, {0,3,4,9,1,4,9,7,1,4,8,9,3,4,9,8,0,4,5,9,4,5,9,7,2,5,7,9,-1}, {0,3,4,9,1,4,9,7,1,4,8,9,3,4,9,8,0,4,5,9,2,4,9,5,2,4,7,9,-1} },
    { {0,1,7,6,2,6,7,9,0,2,6,7,3,6,9,8,6,7,9,8,1,6,8,7,-1}, {0,1,7,6,2,6,7,9,3,6,7,8,0,2,6,7,3,6,9,7,1,6,8,7,-1}, {-1} },
    { {2,6,7,9,0,2,6,7,4,6,8,7,1,4,8,7,3,6,7,8,0,4,7,6,3,6,9,7,-1}, {0,2,9,4,3,4,6,9,1,4,9,7,1,4,8,9,3,4,9,8,0,4,9,6,2,4,7,9,-1}, {0,2,9,4,3,4,6,9,1,4,8,7,3,4,9,8,0,4,9,6,4,7,9,8,2,4,7,9,-1}, {0,2,9,4,1,4,9,7,1,4,8,9,0,4,9,6,4,6,8,9,2,4,7,9,3,6,9,8,-1}, {2,6,7,9,1,4,6,7,0,2,6,7,1,6,8,7,0,4,7,6,3,6,9,8,6,7,9,8,-1}, {0,2,9,4,1,4,8,7,0,4,9,6,4,6,8,9,4,7,9,8,2,4,7,9,3,6,9,8,-1}, {2,6,7,9,1,4,6,7,0,2,6,7,1,6,8,7,3,6,7,8,0,4,7,6,3,6,9,7,-1}, {2,6,7,9,0,2,6,7,1,4,8,7,0,4,7,6,3,6,9,8,4,6,8,7,6,7,9,8,-1} },
    { {0,1,5,8,3,5,8,6,1,5,8,7,3,5,9,8,0,5,6,8,2,5,7,8,2,5,8,9,-1}, {0,1,5,8,1,5,8,7,0,5,6,8,5,6,8,9,2,5,7,8,2,5,8,9,3,6,9,8,-1}, {2,6,7,9,0,1,7,6,1,6,8,7,0,5,6,7,3,6,9,8,2,5,7,6,6,7,9,8,-1}, {0,1,7,6,5,6,7,9,1,6,8,7,3,6,7,8,0,5,6,7,3,6,9,7,2,5,7,9,-1}, {0,1,5,8,3,5,8,6,1,5,8,7,3,5,9,8,0,5,6,8,5,7,9,8,2,5,7,9,-1}, {2,5,7,9,0,1,7,6,1,6,8,7,0,5,6,7,3,6,9,8,5,6,7,9,6,7,9,8,-1}, {2,6,7,9,0,1,7,6,1,6,8,7,2,5,7,6,3,6,7,8,0,5,6,7,3,6,9,7,-1}, {0,1,5,8,1,5,8,7,0,5,6,8,5,6,8,9,5,7,9,8,2,5,7,9,3,6,9,8,-1} },
    { {0,4,5,6,3,6,9,8,4,5,6,9,4,5,9,7,4,6,8,9,4,7,9,8,2,5,7,9,1,4,8,7}, {0,4,5,6,3,6,9,8,4,5,6,7,4,6,8,7,5,6,7,9,2,5,7,9,6,7,9,8,1,4,8,7}, {0,4,5,6,3,6,9,8,4,5,6,8,4,5,8,7,5,6,8,9,5,7,9,8,2,5,7,9,1,4,8,7}, {-1} }
};

// Lookup table for the corresponding face index for small tetrahedra after edge split
const int LUT_EDGE_SPLIT_OF[64][8][32] = {
{ {0,1,2,3} },
{ {-1,2,3,1,-1,2,0,3} },
{ {-1,1,2,3,1,-1,0,3} },
{ {-1,3,1,2,-1,2,0,3,-1,3,-1,1}, {-1,3,1,2,1,-1,0,3,-1,3,2,-1} },
{ {-1,1,2,3,1,2,0,-1} },
{ {-1,2,3,1,-1,2,0,3,2,-1,-1,1}, {-1,2,3,1,1,2,0,-1,-1,2,-1,3} },
{ {-1,1,2,3,1,-1,0,3,1,-1,2,-1}, {-1,1,2,3,1,2,0,-1,1,-1,-1,3} },
{ {-1,1,2,3,-1,-1,3,2,1,2,0,-1,1,-1,-1,3}, {-1,1,2,3,-1,1,3,-1,1,2,0,-1,-1,2,-1,3}, {-1,1,2,3,-1,-1,3,2,1,-1,0,3,1,-1,2,-1}, {-1,1,2,3,1,-1,0,3,-1,1,-1,2,-1,3,2,-1}, {-1,1,2,3,-1,2,0,3,2,-1,-1,1,-1,1,3,-1}, {-1,1,2,3,-1,2,0,3,-1,1,-1,2,-1,3,-1,1} },
{ {0,-1,2,3,0,-1,3,1} },
{ {-1,2,3,1,-1,3,2,0,-1,3,0,-1}, {0,-1,3,1,-1,3,-1,2,-1,3,2,0} },
{ {-1,1,2,3,-1,3,-1,0,-1,3,0,1}, {0,-1,2,3,-1,3,1,-1,-1,3,0,1} },
{ {-1,3,1,2,-1,3,0,1,-1,3,2,0,3,-1,-1,-1} },
{ {-1,-1,2,3,-1,-1,3,1,-1,-1,2,0,-1,-1,0,1} },
{ {-1,2,3,1,-1,3,2,0,2,-1,-1,1,-1,3,0,-1} },
{ {-1,1,2,3,1,-1,2,-1,-1,3,-1,0,-1,3,0,1} },
{ {-1,1,2,3,-1,-1,1,3,-1,-1,3,-1,-1,-1,2,0,-1,-1,3,2,-1,-1,0,1}, {-1,-1,2,3,-1,-1,3,1,-1,-1,1,3,-1,-1,2,0,-1,-1,3,2,-1,-1,0,1} },
{ {0,-1,2,3,0,2,-1,1} },
{ {-1,2,3,1,-1,2,0,3,2,-1,0,-1}, {0,2,-1,1,-1,2,3,-1,-1,2,0,3} },
{ {-1,-1,2,3,-1,-1,1,2,-1,-1,0,3,-1,-1,0,1} },
{ {-1,3,1,2,-1,2,0,3,-1,3,-1,1,2,-1,0,-1} },
{ {-1,1,2,3,-1,2,0,-1,2,-1,0,1}, {0,-1,2,3,-1,2,-1,1,2,-1,0,1} },
{ {-1,2,3,1,2,-1,0,1,-1,2,0,3,2,-1,-1,-1} },
{ {-1,1,2,3,2,-1,0,1,1,-1,-1,3,-1,2,0,-1} },
{ {-1,1,2,3,-1,2,-1,-1,-1,-1,0,3,-1,2,1,-1,-1,-1,3,2,-1,-1,0,1}, {-1,-1,2,3,-1,2,-1,1,-1,-1,0,3,-1,2,1,-1,-1,-1,3,2,-1,-1,0,1} },
{ {0,-1,2,3,0,-1,3,1,0,-1,-1,2}, {0,-1,2,3,0,2,-1,1,0,-1,3,-1} },
{ {-1,0,3,2,0,2,-1,1,0,-1,3,-1,-1,-1,2,3}, {-1,0,3,2,0,-1,3,1,0,-1,-1,2,-1,-1,2,3}, {-1,0,3,2,0,-1,3,1,-1,0,2,-1,-1,3,-1,2}, {-1,2,3,1,-1,0,3,2,2,-1,0,-1,-1,0,-1,3}, {-1,2,3,1,-1,3,0,-1,-1,0,2,-1,-1,0,3,2}, {-1,0,-1,3,-1,0,3,2,0,2,-1,1,-1,2,3,-1} },
{ {0,-1,2,3,-1,3,1,-1,0,-1,-1,2,-1,3,0,1} },
{ {-1,0,-1,3,-1,-1,1,2,-1,-1,2,3,-1,-1,3,2,-1,0,3,-1,-1,-1,0,1}, {-1,0,-1,3,-1,-1,1,2,-1,-1,2,3,-1,-1,3,-1,-1,0,3,2,-1,-1,0,1} },
{ {2,-1,0,1,0,-1,2,3,-1,2,-1,1,0,-1,3,-1} },
{ {-1,-1,3,1,-1,-1,2,3,-1,0,2,-1,-1,-1,-1,2,-1,0,3,2,-1,-1,0,1}, {-1,-1,3,1,-1,-1,2,3,-1,0,2,-1,-1,-1,3,2,-1,0,-1,2,-1,-1,0,1} },
{ {-1,-1,1,3,-1,-1,2,3,-1,-1,3,1,-1,0,2,-1,-1,0,-1,2,-1,-1,0,1}, {-1,-1,2,3,-1,0,-1,3,-1,2,-1,1,-1,2,1,-1,-1,0,3,-1,-1,-1,0,1} },
{ {-1,-1,2,3,-1,-1,1,3,-1,-1,3,1,-1,-1,0,1,-1,0,2,-1,-1,0,3,2,-1,-1,-1,2}, {-1,-1,2,3,-1,2,-1,1,-1,-1,0,1,-1,2,1,-1,-1,0,3,2,-1,-1,3,-1,-1,0,-1,3}, {-1,-1,2,3,-1,-1,3,1,-1,-1,0,1,-1,0,2,-1,-1,-1,3,2,-1,-1,1,3,-1,0,-1,2}, {-1,1,2,3,-1,-1,1,3,-1,-1,0,1,-1,0,2,-1,-1,-1,3,-1,-1,0,3,2,-1,-1,-1,2}, {-1,1,2,3,-1,-1,0,1,-1,2,1,-1,-1,2,-1,-1,-1,0,3,2,-1,-1,3,-1,-1,0,-1,3}, {-1,-1,2,3,-1,2,-1,1,-1,-1,0,1,-1,2,1,-1,-1,-1,3,2,-1,0,3,-1,-1,0,-1,3}, {-1,1,2,3,-1,-1,3,2,-1,-1,0,1,-1,0,2,-1,-1,-1,3,-1,-1,-1,1,3,-1,0,-1,2}, {-1,1,2,3,-1,-1,0,1,-1,2,1,-1,-1,-1,3,2,-1,2,-1,-1,-1,0,3,-1,-1,0,-1,3} },
{ {0,1,-1,3,0,1,2,-1} },
{ {-1,-1,3,1,-1,-1,1,2,-1,-1,0,3,-1,-1,2,0} },
{ {-1,1,2,3,1,-1,0,3,1,-1,-1,0}, {0,1,2,-1,-1,1,-1,3,1,-1,0,3} },
{ {-1,3,1,2,1,-1,0,3,-1,3,2,-1,1,-1,-1,0} },
{ {-1,1,2,3,1,-1,0,-1,1,-1,2,0}, {0,1,-1,3,-1,1,2,-1,1,-1,2,0} },
{ {-1,2,3,1,-1,2,-1,3,1,-1,0,-1,1,-1,2,0} },
{ {-1,1,2,3,1,-1,2,0,1,-1,0,3,1,-1,-1,-1} },
{ {-1,1,2,3,-1,1,3,-1,-1,-1,0,3,-1,1,-1,2,1,-1,-1,-1,-1,-1,2,0}, {-1,1,-1,3,-1,1,2,-1,-1,1,3,-1,-1,-1,0,3,-1,1,-1,2,-1,-1,2,0} },
{ {0,-1,2,3,0,-1,3,1,0,-1,1,-1}, {0,1,2,-1,0,-1,-1,3,0,-1,3,1} },
{ {0,-1,3,1,-1,3,-1,2,0,-1,1,-1,-1,3,2,0} },
{ {0,-1,2,3,-1,0,1,3,-1,0,-1,1,-1,3,1,-1}, {-1,1,2,3,1,-1,-1,0,-1,0,3,-1,-1,0,1,3}, {0,1,2,-1,-1,0,1,3,0,-1,-1,3,-1,-1,3,1}, {-1,1,2,3,-1,3,-1,0,-1,0,1,3,-1,0,-1,1}, {-1,0,3,-1,-1,0,1,3,0,1,2,-1,-1,1,-1,3}, {0,-1,2,3,-1,0,1,3,0,-1,1,-1,-1,-1,3,1} },
{ {-1,1,3,-1,-1,0,-1,3,-1,-1,1,2,-1,1,-1,3,-1,-1,2,0,-1,0,3,-1}, {-1,-1,1,2,-1,0,1,3,-1,1,-1,3,-1,-1,3,-1,-1,-1,2,0,-1,0,3,-1} },
{ {-1,1,2,-1,0,-1,-1,3,0,-1,3,1,1,-1,2,0} },
{ {-1,-1,3,1,-1,0,1,-1,-1,-1,2,3,-1,0,-1,1,-1,-1,2,0,-1,-1,3,2}, {-1,-1,3,1,-1,0,-1,3,-1,1,2,-1,-1,1,-1,2,-1,-1,2,0,-1,0,3,-1} },
{ {-1,-1,2,3,-1,0,1,3,-1,-1,3,1,-1,0,-1,1,-1,-1,1,-1,-1,-1,2,0}, {-1,-1,1,3,-1,-1,2,3,-1,0,1,-1,-1,-1,3,1,-1,0,-1,1,-1,-1,2,0} },
{ {-1,-1,2,3,-1,-1,3,2,-1,-1,3,1,-1,0,-1,1,-1,-1,1,-1,-1,-1,2,0,-1,0,1,3}, {-1,1,2,3,-1,0,3,-1,-1,1,-1,2,-1,-1,2,0,1,-1,-1,-1,-1,-1,3,-1,-1,0,1,3}, {-1,1,2,3,-1,1,3,-1,-1,0,3,-1,-1,1,-1,2,-1,-1,2,0,1,-1,-1,-1,-1,0,-1,3}, {-1,0,1,-1,-1,-1,2,3,-1,-1,3,2,-1,-1,3,1,-1,-1,1,3,-1,0,-1,1,-1,-1,2,0}, {-1,1,2,3,-1,0,1,-1,-1,-1,3,2,-1,-1,1,3,-1,0,-1,1,-1,-1,3,-1,-1,-1,2,0}, {-1,1,2,3,-1,-1,3,2,-1,0,-1,1,-1,-1,3,-1,-1,-1,1,-1,-1,-1,2,0,-1,0,1,3}, {-1,1,-1,3,-1,1,2,-1,-1,0,3,-1,-1,1,-1,2,-1,1,3,-1,-1,-1,2,0,-1,0,-1,3}, {-1,1,-1,3,-1,1,2,-1,-1,0,3,-1,-1,1,-1,2,-1,-1,2,0,-1,-1,3,-1,-1,0,1,3} },
{ {0,-1,2,3,0,-1,-1,1,0,-1,1,2}, {0,1,-1,3,0,-1,2,-1,0,-1,1,2} },
{ {-1,2,3,-1,0,-1,-1,1,0,-1,1,2,-1,2,0,3} },
{ {-1,1,-1,3,0,-1,2,-1,0,-1,1,2,1,-1,0,3} },
{ {-1,-1,1,2,-1,0,1,-1,-1,-1,2,3,-1,0,-1,1,-1,-1,0,3,-1,-1,3,2}, {-1,-1,1,2,-1,1,3,-1,-1,1,-1,3,-1,0,2,-1,-1,-1,0,3,-1,0,-1,2} },
{ {0,-1,2,3,-1,0,1,-1,-1,0,2,1,-1,2,-1,1}, {0,1,-1,3,0,-1,2,-1,-1,0,2,1,-1,-1,1,2}, {-1,0,-1,2,0,1,-1,3,-1,1,2,-1,-1,0,2,1}, {0,-1,2,3,-1,0,2,1,0,-1,-1,1,-1,-1,1,2}, {-1,1,2,3,-1,2,0,-1,-1,0,1,-1,-1,0,2,1}, {-1,1,2,3,1,-1,0,-1,-1,0,-1,2,-1,0,2,1} },
{ {-1,-1,3,1,-1,1,-1,2,-1,1,2,-1,-1,0,2,-1,-1,-1,0,3,-1,0,-1,2}, {-1,-1,3,1,-1,1,2,-1,-1,0,2,1,-1,-1,-1,2,-1,-1,0,3,-1,0,-1,2} },
{ {-1,-1,2,3,-1,0,1,-1,-1,2,-1,1,-1,0,2,1,-1,-1,1,-1,-1,-1,0,3}, {-1,-1,2,3,-1,0,1,-1,-1,2,1,-1,-1,2,-1,1,-1,0,-1,1,-1,-1,0,3} },
{ {-1,1,-1,3,-1,1,2,-1,-1,0,-1,2,-1,0,2,1,-1,-1,0,3,-1,1,3,-1,-1,-1,-1,2}, {-1,0,1,-1,-1,-1,2,3,-1,-1,3,2,-1,2,-1,1,-1,0,2,1,-1,-1,0,3,-1,-1,1,-1}, {-1,1,2,3,-1,0,1,-1,-1,-1,3,2,-1,2,1,-1,-1,0,-1,1,-1,2,-1,-1,-1,-1,0,3}, {-1,1,2,3,-1,0,-1,2,-1,1,-1,2,-1,0,2,-1,-1,1,3,-1,-1,-1,0,3,1,-1,-1,-1}, {-1,1,-1,3,-1,1,2,-1,-1,0,-1,2,-1,1,-1,2,-1,0,2,-1,-1,-1,0,3,-1,1,3,-1}, {-1,1,2,3,-1,0,1,-1,-1,-1,3,2,-1,0,2,1,-1,2,-1,-1,-1,-1,0,3,-1,-1,1,-1}, {-1,0,1,-1,-1,-1,2,3,-1,-1,3,2,-1,2,-1,1,-1,2,1,-1,-1,0,-1,1,-1,-1,0,3}, {-1,1,2,3,-1,0,-1,2,-1,1,3,-1,-1,0,2,1,-1,-1,0,3,1,-1,-1,-1,-1,-1,-1,2} },
{ {0,-1,2,3,0,-1,1,2,0,-1,3,1,0,-1,-1,-1} },
{ {-1,-1,3,1,-1,0,-1,3,0,-1,-1,-1,-1,-1,1,2,-1,0,2,-1,-1,0,3,2}, {-1,0,-1,2,-1,-1,3,1,-1,0,-1,3,-1,-1,1,2,-1,0,2,-1,-1,0,3,-1} },
{ {-1,-1,2,3,-1,0,-1,3,-1,0,1,-1,-1,-1,1,2,-1,0,-1,1,-1,0,3,-1}, {-1,-1,2,3,-1,0,1,3,0,-1,-1,-1,-1,-1,1,2,-1,0,-1,1,-1,0,3,-1} },
{ {-1,-1,1,2,-1,0,3,2,-1,0,2,-1,-1,1,-1,3,-1,-1,3,-1,0,-1,-1,-1,-1,0,1,3}, {-1,-1,1,2,-1,0,3,2,-1,0,2,-1,-1,1,-1,3,-1,1,3,-1,0,-1,-1,-1,-1,0,-1,3}, {-1,0,1,-1,-1,0,3,2,-1,-1,1,2,-1,-1,2,3,-1,0,-1,1,-1,-1,3,-1,-1,0,-1,3}, {-1,0,1,-1,-1,-1,3,2,-1,-1,1,2,-1,0,3,-1,-1,-1,2,3,-1,0,-1,1,-1,0,-1,3}, {-1,0,3,2,-1,-1,1,2,-1,-1,2,3,-1,0,-1,1,-1,-1,3,-1,0,-1,-1,-1,-1,0,1,3}, {-1,-1,3,2,-1,-1,1,2,-1,0,3,-1,-1,-1,2,3,-1,0,-1,1,0,-1,-1,-1,-1,0,1,3}, {-1,-1,1,2,-1,0,3,-1,-1,0,-1,2,-1,0,2,-1,-1,1,-1,3,-1,-1,3,-1,-1,0,1,3}, {-1,-1,1,2,-1,0,3,-1,-1,0,-1,2,-1,0,2,-1,-1,1,-1,3,-1,1,3,-1,-1,0,-1,3} },
{ {-1,-1,2,3,-1,0,1,-1,-1,-1,3,1,-1,0,2,1,0,-1,-1,-1,-1,0,-1,2}, {-1,-1,2,3,-1,0,1,-1,-1,0,2,-1,-1,-1,3,1,-1,0,-1,1,-1,0,-1,2} },
{ {-1,0,1,-1,-1,-1,3,1,-1,-1,-1,2,-1,0,3,2,-1,0,2,-1,-1,-1,2,3,-1,0,-1,1}, {-1,-1,3,1,-1,1,-1,2,-1,0,3,-1,-1,0,-1,2,-1,0,2,-1,-1,1,2,-1,-1,0,-1,3}, {-1,-1,3,1,-1,1,-1,2,-1,0,3,2,-1,0,2,-1,-1,1,2,-1,0,-1,-1,-1,-1,0,-1,3}, {-1,-1,3,1,-1,0,3,-1,-1,0,-1,2,-1,1,2,-1,-1,-1,-1,2,-1,0,-1,3,-1,0,2,1}, {-1,0,1,-1,-1,-1,3,2,-1,-1,3,1,-1,0,-1,2,-1,-1,2,3,-1,0,2,1,0,-1,-1,-1}, {-1,-1,3,1,-1,0,3,2,-1,1,2,-1,-1,-1,-1,2,0,-1,-1,-1,-1,0,-1,3,-1,0,2,1}, {-1,0,1,-1,-1,-1,3,2,-1,-1,3,1,-1,0,-1,2,-1,0,2,-1,-1,-1,2,3,-1,0,-1,1}, {-1,0,1,-1,-1,-1,3,1,-1,0,3,2,-1,-1,2,3,-1,0,2,1,-1,-1,-1,2,0,-1,-1,-1} },
{ {-1,-1,2,3,-1,2,1,-1,-1,0,3,-1,-1,0,-1,1,-1,2,-1,1,-1,0,-1,3,-1,0,1,-1}, {-1,-1,2,3,-1,0,3,-1,-1,2,-1,1,-1,-1,1,-1,-1,0,-1,3,-1,0,1,-1,-1,0,2,1}, {-1,0,1,-1,-1,-1,2,3,-1,0,-1,2,-1,-1,3,1,-1,0,2,1,-1,-1,1,3,0,-1,-1,-1}, {-1,-1,2,3,-1,-1,1,-1,-1,0,-1,2,-1,0,2,-1,-1,-1,3,1,-1,0,-1,1,-1,0,1,3}, {-1,-1,2,3,-1,2,1,-1,-1,0,3,-1,-1,0,-1,1,-1,2,-1,1,0,-1,-1,-1,-1,0,1,3}, {-1,0,1,3,-1,-1,2,3,-1,0,-1,2,-1,-1,3,1,-1,0,2,1,-1,-1,1,-1,0,-1,-1,-1}, {-1,0,1,-1,-1,-1,2,3,-1,0,-1,2,-1,-1,1,3,-1,0,2,-1,-1,-1,3,1,-1,0,-1,1}, {-1,-1,2,3,-1,0,3,-1,-1,2,-1,1,-1,-1,1,-1,0,-1,-1,-1,-1,0,1,3,-1,0,2,1} },
{ {-1,1,2,3,-1,0,2,1,1,-1,-1,-1,-1,-1,3,-1,-1,-1,-1,2,0,-1,-1,-1,-1,0,1,3,-1,0,3,2}, {-1,1,2,3,-1,0,2,1,-1,-1,3,-1,-1,-1,-1,2,-1,-1,1,-1,-1,0,1,3,0,-1,-1,-1,-1,0,3,2}, {-1,1,2,3,-1,0,2,1,-1,2,-1,-1,-1,-1,3,-1,-1,-1,1,-1,0,-1,-1,-1,-1,0,1,3,-1,0,3,2} }
};


// Check if the sum of volumes of tetrahedra is the same as the volume
// calculated by applying divergence theorem on the surface.
// If they are not equal, the mesh must be invalid.
void assertVolumeEqual(
    const std::vector<vec3>& verts,
    const std::vector<ivec4>& tets
) {
    // sum of tetrahedron values
    double Vt = 0.0;
    for (ivec4 tet : tets) {
        double dV = det(
            verts[tet[1]] - verts[tet[0]],
            verts[tet[2]] - verts[tet[0]],
            verts[tet[3]] - verts[tet[0]]
        ) / 6.0;
        assert(dV > 0.0);
        Vt += dV;
    }

    // triangle faces
    std::set<ivec3, decltype(ivec3Cmp)> faces(ivec3Cmp);
    for (ivec4 tet : tets) {
        for (int fi = 0; fi < 4; fi++) {
            ivec3 f;
            for (int _ = 0; _ < 3; _++)
                f[_] = tet[FACES[fi][_]];
            f = rotateIvec3(f);
            assert(faces.find(f) == faces.end());
            ivec3 fo = ivec3(f.x, f.z, f.y);
            if (faces.find(fo) != faces.end())
                faces.erase(fo);
            else faces.insert(f);
        }
    }

    // volume from triangle faces
    double Vs = 0.0;
    for (ivec3 tri : faces) {
        double dV = det(
            verts[tri.x], verts[tri.y], verts[tri.z]
        ) / 6.0;
        Vs += dV;
    }
    assert(Vs > 0);

    // compare
    printf("Vt=%lg Vs=%lg\n", Vt, Vs);
    assert(abs(Vs / Vt - 1.0) < 1e-6);
}


// Refine the mesh, requires positive volumes for all tets
void smoothMesh(
    std::vector<vec3>& verts,
    std::vector<ivec4>& tets,
    int nsteps,
    ScalarFieldFBatch F = nullptr,
    std::function<vec3(vec3)> constraint = nullptr,  // add this to bring point back
    std::vector<int> constraintI = std::vector<int>(),
    std::vector<vec3> constraintN = std::vector<vec3>()
) {
    printf("%d %d\n", constraintI.size(), constraintN.size());
    int vn = (int)verts.size(), svn = 0;  // # of vertices; # on surface
    int tn = (int)tets.size(), stn = 0;  // # of tets; # on surface
    for (int i : constraintI)
        assert(i >= 0 && i < vn);

    // surface
    std::set<ivec3, decltype(ivec3Cmp)> faces(ivec3Cmp);
    for (ivec4 tet : tets) {
        for (int fi = 0; fi < 4; fi++) {
            ivec3 f;
            for (int _ = 0; _ < 3; _++)
                f[_] = tet[FACES[fi][_]];
            f = rotateIvec3(f);
            assert(faces.find(f) == faces.end());
            ivec3 fo = ivec3(f.x, f.z, f.y);
            if (faces.find(fo) != faces.end())
                faces.erase(fo);
            else faces.insert(f);
        }
    }
    for (ivec4 t : tets) {
        for (int _ = 0; _ < 4; _++) {
            ivec3 f(t[_], t[(_ + 1) % 4], t[(_ + 2) % 4]);
            if (_ % 2 == 1) std::swap(f.y, f.z);
            assert(faces.find(rotateIvec3(f)) == faces.end());
        }
    }

    // edges
    std::unordered_set<uint64_t> edges;
    auto edgeKey = [](int a, int b) -> uint64_t {
        return (uint64_t)max(a, b) | ((uint64_t)min(a, b) << 32);
    };
    for (ivec3 f : faces) {
        for (int _ = 0; _ < 3; _++)
            edges.insert(edgeKey(f[_], f[(_ + 1) % 3]));
    }

    // geometric topology
    std::vector<int> compressedIndex(vn, -1);  // [vn] global -> near surface
    std::vector<int> fullIndex;  // [svn] near surface -> global
    std::vector<int> surfaceTets;  // [stn] indices of tets near surface
    std::vector<bool> isConstrained[4];  // [vn] constrained on domain boundary? (any, x, y, z)
    std::vector<bool> applySurfaceConstraints(vn, false);  // [vn] constrained on isosurface?
    auto isOnFace = [&](int i) {
        return isConstrained[0][i] || applySurfaceConstraints[i];
    };
    // geometry and values
    std::vector<vec3> surfaceVertPs;  // [svn] positions
    std::vector<double> surfaceVertVals;  // [svn] function values
    std::vector<vec3> surfaceVertGrads, surfaceTetGrads;  // [svn, stn] gradients
    std::vector<double> surfaceVertGradWeights;  // [svn] used to project tet gradients to verts
    // smoothing
    std::vector<vec3> grads(vn);
    std::vector<double> maxFactor(vn), maxMovement(vn);
    // add mew
    auto addVert = [&](vec3 p, bool isNearFace,
        bool isConstrainedX, bool isConstrainedY, bool isConstrainedZ,
        bool isConstrainedF) {
        verts.push_back(p);
        compressedIndex.push_back(-1);
        if (isNearFace) {
            compressedIndex.back() = svn++;
            fullIndex.push_back(vn);
            surfaceVertPs.push_back(vec3(0));
            surfaceVertVals.push_back(0.0);
            surfaceVertGrads.push_back(vec3(0));
            surfaceVertGradWeights.push_back(0.0);
        }
        isConstrained[0].push_back(
            isConstrainedX || isConstrainedY || isConstrainedZ);
        isConstrained[1].push_back(isConstrainedX);
        isConstrained[2].push_back(isConstrainedY);
        isConstrained[3].push_back(isConstrainedZ);
        applySurfaceConstraints.push_back(isConstrainedF);
        grads.push_back(vec3(0));
        maxFactor.push_back(0), maxMovement.push_back(0);
        return vn++;
    };
    auto addTet = [&](ivec4 t, int i = -1) {
        if (i == -1) tets.push_back(t);
        else tets[i] = t;
        bool isSurfaceTet = false;
        for (int _ = 0; _ < 4; _++)
            if (compressedIndex[t[_]] != -1)
                isSurfaceTet = true;
        if (isSurfaceTet) {
            if (i == -1) {
                surfaceTets.push_back(tn);
                surfaceTetGrads.push_back(vec3(0));
                stn++;
                for (int _ = 0; _ < 4; _++)
                    if (compressedIndex[t[_]] == -1) {
                        compressedIndex[t[_]] = svn++;
                        fullIndex.push_back(t[_]);
                        surfaceVertPs.push_back(vec3(0));
                        surfaceVertVals.push_back(0.0);
                        surfaceVertGrads.push_back(vec3(0));
                        surfaceVertGradWeights.push_back(0.0);
                    }
            }
        }
        return i != -1 ? i : tn++;
    };

    // vertices near surface
    for (int _ = 0; _ < 4; _++)
        isConstrained[_] = std::vector<bool>(vn, false);
    for (int ci = 0; ci < (int)constraintI.size(); ci++) {
        int i = constraintI[ci];
        isConstrained[0][i] = true;
        vec3 t = constraintN[ci];
        if (t == vec3(1, 0, 0))
            isConstrained[1][i] = true;
        else if (t == vec3(0, 1, 0))
            isConstrained[2][i] = true;
        else if (t == vec3(0, 0, 1))
            isConstrained[3][i] = true;
        else assert(false);
    }
    if (F) {
        // on face
        for (ivec3 f : faces) {
            bool isC = isConstrained[0][f[0]]
                && isConstrained[0][f[1]]
                && isConstrained[0][f[2]];
            for (int _ = 0; _ < 3; _++) {
                if (compressedIndex[f[_]] == -1) {
                    compressedIndex[f[_]] = svn;
                    fullIndex.push_back(f[_]);
                    svn++;
                }
                if (!isC)
                    applySurfaceConstraints[f[_]] = true;
            }
        }
        // one layer
        int i0 = 0, i1 = svn;
        for (int ti = 0; ti < tn; ti++) {
            ivec4 tet = tets[ti];
            int onface = 0;
            for (int _ = 0; _ < 4; _++)
                if (compressedIndex[tet[_]] >= i0 &&
                    compressedIndex[tet[_]] < i1)
                    onface += 1;
            if (onface == 0)
                continue;
            surfaceTets.push_back(ti);
            for (int _ = 0; _ < 4; _++)
                if (compressedIndex[tet[_]] == -1) {
                    compressedIndex[tet[_]] = svn;
                    fullIndex.push_back(tet[_]);
                    svn++;
                }
            stn++;
        }
        // make sure isOnFace() works
        std::vector<bool> onFace(vn, false);
        for (ivec3 f : faces)
            for (int _ = 0; _ < 3; _++)
                onFace[f[_]] = true;
        for (int i = 0; i < vn; i++)
            assert(onFace[i] == isOnFace(i));
    }
    surfaceVertPs.resize(svn);
    surfaceVertVals.resize(svn);
    surfaceVertGrads.resize(svn);
    surfaceVertGradWeights.resize(svn);
    surfaceTetGrads.resize(stn);

    for (int stepi = 0; stepi < nsteps; stepi++) {

        /* Smoothing */

        // accumulate gradient
        for (int i = 0; i < vn; i++)
            grads[i] = vec3(0.0);
        for (ivec4 tet : tets) {
            vec3 v[4], g[4];
            for (int _ = 0; _ < 4; _++)
                v[_] = verts[tet[_]];
            const double* vd = (const double*)&v[0];
            double val, size2;
            double* res[3] = { &val, (double*)g, &size2 };
            MeshgenTetLoss::meshgen_tet_loss(&vd, res, nullptr, nullptr, 0);
            for (int _ = 0; _ < 4; _++)
                grads[tet[_]] -= 0.01 * g[_] * size2;
        }

        // force the mesh on the surface
        if (F) {
            // evaluate
            for (int i = 0; i < svn; i++) {
                int j = fullIndex[i];
                if (j != -1)
                    surfaceVertPs[i] = verts[j] + grads[j];
            }
            F(svn, &surfaceVertPs[0], &surfaceVertVals[0]);
            // gradient on tets
            for (int i = 0; i < stn; i++) {
                vec3 x[4]; double v[4];
                for (int _ = 0; _ < 4; _++) {
                    int j = compressedIndex[tets[surfaceTets[i]][_]];
                    assert(j >= 0 && j < svn);
                    x[_] = surfaceVertPs[j];
                    v[_] = surfaceVertVals[j];
                }
                mat3 m(x[1]-x[0], x[2]-x[0], x[3]-x[0]);
                vec3 b(v[1]-v[0], v[2]-v[0], v[3]-v[0]);
                surfaceTetGrads[i] = inverse(transpose(m)) * b;
            }
            // gradient on verts
            for (int i = 0; i < svn; i++) {
                surfaceVertGradWeights[i] = 0.0;
                surfaceVertGrads[i] = vec3(0.0);
            }
            for (int i = 0; i < stn; i++) {
                for (int _ = 0; _ < 4; _++) {
                    int j = compressedIndex[tets[surfaceTets[i]][_]];
                    surfaceVertGrads[j] += surfaceTetGrads[i];
                    surfaceVertGradWeights[j] += 1.0;
                }
            }
            for (int i = 0; i < svn; i++) {
                if (surfaceVertGradWeights[i] <= 0.0) printf("%d %lf\n", i, surfaceVertGradWeights[i]);
                assert(surfaceVertGradWeights[i] > 0.0);
                surfaceVertGrads[i] /= surfaceVertGradWeights[i];
            }
            // boundary constraints
            for (int ci = 0; ci < (int)constraintI.size(); ci++) {
                int i = compressedIndex[constraintI[ci]];
                if (i == -1) continue;
                vec3 n = constraintN[ci];
                surfaceVertGrads[i] -= dot(surfaceVertGrads[i], n) * n;
            }
            // move the vertex to the surface
            for (int i = 0; i < svn; i++) {
                if (fullIndex[i] == -1)
                    continue;
                if (!applySurfaceConstraints[fullIndex[i]])
                    continue;
                double v = surfaceVertVals[i];
                vec3 g = surfaceVertGrads[i];
                grads[fullIndex[i]] -= v * g / dot(g, g);
            }
        }

        // apply boundary constraints
        for (int i = 0; i < vn; i++) {
            for (int _ = 0; _ < 3; _++)
                if (isConstrained[_ + 1][i])
                    ((double*)&grads[i])[_] = 0.0;
        }
        if (constraint) {
            for (int i = 0; i < vn; i++)
                grads[i] += constraint(verts[i] + grads[i]);
        }

        // calculate maximum allowed vertex movement factor
        for (int i = 0; i < vn; i++)
            maxFactor[i] = 1.0, maxMovement[i] = 0.0;
        for (ivec4 tet : tets) {
            // prevent going negative by passing through a face
            const static int fvp[4][4] = {
                {0,1,2,3}, {0,3,1,2}, {0,2,3,1}, {1,3,2,0}
            };
            // check faces
            vec3 v[4], g[4];
            double mf[4] = { 1.0, 1.0, 1.0, 1.0 };
            for (int i = 0; i < 4; i++) {
                for (int _ = 0; _ < 4; _++) {
                    int j = tet[fvp[i][_]];
                    v[_] = verts[j], g[_] = grads[j];
                }
                // plane normal and distance to the vertex
                vec3 n = normalize(cross(v[1] - v[0], v[2] - v[0]));
                double d = dot(n, v[3] - v[0]);
                assert(d > 0.0);
                // how far you need to go to make it negative
                double d3 = max(-dot(n, g[3]), 0.0);
                double k[4] = { 1, 1, 1, 1 };
                for (int _ = 0; _ < 3; _++) {
                    double d_ = max(dot(n, g[_]), 0.0);
                    double ds = d_ + d3;
                    if (ds == 0.0) continue;
                    k[_] = min(k[_], d / ds);
                }
                k[3] = min(min(k[0], k[1]), k[2]);
                for (int _ = 0; _ < 4; _++)
                    mf[fvp[i][_]] = min(mf[fvp[i][_]], k[_]);
            }
            for (int _ = 0; _ < 4; _++)
                maxFactor[tet[_]] = min(maxFactor[tet[_]],
                    mf[_] > 0.0 ? mf[_] : 1.0);
            // prevent going crazy
            double sl = cbrt(abs(det(v[1] - v[0], v[2] - v[0], v[3] - v[0]) / 6.0));
            for (int _ = 0; _ < 4; _++)
                maxMovement[tet[_]] = max(maxMovement[tet[_]], sl);
        }

        // displacements
        for (int i = 0; i < vn; i++) {
            vec3 g = 0.9 * maxFactor[i] * grads[i];
            double gl = length(g);
            if (gl != 0.0) {
                double a = maxMovement[i];
                g *= a * tanh(gl / a) / gl;
            }
            grads[i] = g;
        }

        // expect this to drop to 0.1x after 20 iterations
        // if not, adjust step size
        double meanDisp = 0.0;
        for (int i = 0; i < vn; i++)
            meanDisp += length(grads[i]) / vn;
        printf("%lf\n", meanDisp);

        // reduce displacement if negative volume occurs
        std::vector<bool> reduce(vn, true);
        for (int iter = 0; iter < 4; iter++) {
            // update vertex position
            const double r = 0.8;
            double k = (iter == 0 ? 1.0 : (r - 1.0) * pow(r, iter - 1.0));
            for (int i = 0; i < vn; i++) if (reduce[i])
                verts[i] += k * grads[i];
            // check if negative volume occurs
            reduce = std::vector<bool>(vn, false);
            bool found = false;
            for (ivec4 tet : tets) {
                vec3 v[4] = {
                    verts[tet[0]], verts[tet[1]], verts[tet[2]], verts[tet[3]]
                };
                if (det(v[1] - v[0], v[2] - v[0], v[3] - v[0]) < 0.0) {
                    reduce[tet[0]] = reduce[tet[1]] =
                        reduce[tet[2]] = reduce[tet[3]] = true;
                    found = true;
                    // printf("%d\n", iter);
                }
            }
            if (!found) break;
        }

        /* Mesh Change */

        if (stepi + 2 >= nsteps || stepi % 3 != 2)
            continue;

        // edges
        std::unordered_map<uint64_t, int> splitEdges;
        std::unordered_set<uint64_t> bridgeEdges;
        int edgeCount = 0;
        auto addSplitEdge = [&](int a, int b, bool isBridge) {
            uint64_t key = (uint64_t)max(a, b) | ((uint64_t)min(a, b) << 32);
            splitEdges[key] += 1;
            edgeCount += 1;
            if (isBridge) bridgeEdges.insert(key);
        };
        auto getSplitEdge = [&](int a, int b) {
            uint64_t key = (uint64_t)max(a, b) | ((uint64_t)min(a, b) << 32);
            if (splitEdges.find(key) == splitEdges.end())
                return -1;
            return splitEdges[key];
        };

        for (int ti = 0; ti < tn; ti++) {
            ivec4 tet = tets[ti];
            vec3 p[4];
            for (int _ = 0; _ < 4; _++) {
                int vi = tet[_];
                assert(vi >= 0 && vi < vn);
                p[_] = verts[vi];
            }

            // large solid angle: split face, max 1
            // doesn't really work - results in more large angles
            if (0) {
                double maxang = 0.0;
                int maxi = -1;
                for (int _ = 0; _ < 4; _++) {
                    double ang = solidAngle(
                        p[(_+1)%4] - p[_],
                        p[(_+2)%4] - p[_],
                        p[(_+3)%4] - p[_]
                    ) * (_ & 1 ? -1.0 : 1.0);
                    assert(ang > 0.0);
                    if (ang > maxang)
                        maxang = ang, maxi = _;
                }
                if (maxang > PI) {
                    const int* f = FACES[maxi];
                    for (int j = 0; j < 3; j++)
                        addSplitEdge(tet[f[j]], tet[f[(j+1)%3]], false);
                }
            }

            // large face angle: split opposite edge
            // small face angle: split adjacent edge
            if (1) {
                for (int fi = 0; fi < 4; fi++) {
                    ivec3 f;
                    for (int _ = 0; _ < 3; _++)
                        f[_] = tet[FACES[fi][_]];
                    f = rotateIvec3(f);
                    assert(faces.find(ivec3(f.x, f.z, f.y)) == faces.end());
                    if (faces.find(f) == faces.end())
                        continue;
                    for (int _ = 0; _ < 3; _++)
                        assert(edges.find(edgeKey(f[_], f[(_ + 1) % 3])) != edges.end());
                    for (int _ = 0; _ < 3; _++) {
                        int i0 = f[_];
                        int i1 = f[(_ + 1) % 3];
                        int i2 = f[(_ + 2) % 3];
                        vec3 a = verts[i1] - verts[i0];
                        vec3 b = verts[i2] - verts[i0];
                        double ang = acos(dot(a, b) / (length(a) * length(b)));
                        if (ang > 0.9 * PI) {
                            if (isConstrained[0][i0] && applySurfaceConstraints[i0] &&
                                isConstrained[0][i1] && applySurfaceConstraints[i1] &&
                                isConstrained[0][i2] && applySurfaceConstraints[i2])
                                applySurfaceConstraints[i0] = false;
                            else
                                addSplitEdge(i1, i2, false);
                        }
                        if (ang < 0.1 * PI) {
                            addSplitEdge(i0, i1, false);
                            addSplitEdge(i0, i2, false);
                        }
                    }
                }
            }
        }
        // printf("%d\n", edgeCount);

        edgeCount = 0;
        for (std::pair<uint64_t, int> ec : splitEdges) {
            assert(ec.second > 0);
            // printf("%d", ec.second);
            int a = (int)(ec.first >> 32);
            int b = (int)ec.first;
            vec3 p = 0.5 * (verts[a] + verts[b]);
            bool hasEdge = edges.find(edgeKey(a, b)) != edges.end();
            assert(hasEdge);
            bool boundConstrained[4] = { false };
            for (int _ = 1; _ < 4; _++) {
                boundConstrained[_] = isConstrained[_][a] && isConstrained[_][b];
                boundConstrained[0] |= boundConstrained[_];
            }
            int vi = addVert(p,
                compressedIndex[a] != -1 || compressedIndex[b] != -1,
                // false, false, false, false
                // true, true, true, true
                hasEdge && boundConstrained[1],
                hasEdge && boundConstrained[2],
                hasEdge && boundConstrained[3],
                hasEdge && ((//bridgeEdges.find(ec.first) != bridgeEdges.end() ? false :
                    (applySurfaceConstraints[a] || applySurfaceConstraints[b])
                ) || !boundConstrained[0])  // exception: "bridge" ?
            );
            assert(isOnFace(vi));
            if (hasEdge)
                edges.erase(edgeKey(a, b));
            else assert(false);
            splitEdges[ec.first] = vi;
            edgeCount++;
        }
        printf("%d\n", edgeCount);

        edgeCount = 0;
        int tn0 = tn;
        for (int ti = 0; ti < tn0; ti++) {
            ivec4 tet = tets[ti];
            // index + subtet vertex indices
            int e2v[6];
            int idx = 0;
            for (int _ = 0; _ < 6; _++) {
                int a = tet[EDGES[_][0]];
                int b = tet[EDGES[_][1]];
                e2v[_] = getSplitEdge(a, b);
                if (e2v[_] != -1)
                    idx |= (1 << _);
            }
            if (idx == 0) continue;
            // remove faces
            bool hasFace[4];
            for (int _ = 0; _ < 4; _++) {
                ivec3 f;
                for (int __ = 0; __ < 3; __++)
                    f[__] = tet[FACES[_][__]];
                f = rotateIvec3(f);
                assert(faces.find(ivec3(f.x, f.z, f.y)) == faces.end());
                hasFace[_] = faces.find(f) != faces.end();
                if (hasFace[_]) {
                    assert(isOnFace(f[0]) && isOnFace(f[1]) && isOnFace(f[2]));
                    faces.erase(f);
                }
            }
            // add tets
            for (int i = 0; i < 32; i += 4) {
                const int* subtets = &LUT_EDGE_SPLIT[idx][0][i];
                const int* subofs = &LUT_EDGE_SPLIT_OF[idx][0][i];
                if (subtets[0] == -1)
                    break;
                ivec4 newtet;
                for (int _ = 0; _ < 4; _++) {
                    int v = subtets[_];
                    if (v < 4) v = tet[v];
                    else v = e2v[v - 4];
                    newtet[_] = v;
                }
                if (i == 0) addTet(newtet, ti);
                else addTet(newtet);
                // add faces
                for (int _ = 0; _ < 4; _++) {
                    if (subofs[_] == -1 || !hasFace[subofs[_]])
                        continue;
                    ivec3 f;
                    for (int __ = 0; __ < 3; __++)
                        f[__] = newtet[FACES[_][__]];
                    f = rotateIvec3(f);
                    assert(isOnFace(f[0]) && isOnFace(f[1]) && isOnFace(f[2]));
                    for (int __ = 0; __ < 3; __++)
                        edges.insert(edgeKey(f[__], f[(__ + 1) % 3]));
                    assert(faces.find(f) == faces.end());
                    assert(faces.find(ivec3(f.x, f.z, f.y)) == faces.end());
                    faces.insert(f);
                }
            }
            edgeCount++;
        }
        printf("%d\n", edgeCount);

        /* Sanity Check */

        if (false)
            continue;

        assertVolumeEqual(verts, tets);

        // make sure isOnFace() works
        std::vector<bool> onFaceF(vn, false);
        for (ivec3 f : faces)
            for (int _ = 0; _ < 3; _++)
                onFaceF[f[_]] = true;
        std::vector<bool> onFaceE(vn, false);
        for (uint64_t e : edges)
            for (int _ = 0; _ < 2; _++)
                onFaceE[((int*)&e)[_]] = true;
        for (int i = 0; i < vn; i++) {
            assert(onFaceF[i] == onFaceE[i]);
            if (onFaceF[i] != isOnFace(i)) {
            // if (onFaceF[i] != onFaceE[i]) {
                printf("%d/%d  %d %d  %d %d\n", i, vn,
                    (int)onFaceF[i], (int)isOnFace(i),
                    (int)isConstrained[0][i], (int)applySurfaceConstraints[i]
                );
            }
        }

        // solid angle check??
        std::vector<double> sa(vn, 0.0);
        for (ivec4 t : tets) {
            for (int _  = 0; _ < 4; _++) {
                vec3 p = verts[t[_]];
                vec3 v[3];
                for (int __ = 0; __ < 3; __++)
                    v[__] = verts[t[FACES[_][__]]];
                double a = solidAngle(v[0]-p, v[1]-p, v[2]-p);
                assert(a > 0.0);
                sa[t[_]] += a;
            }
        }
        for (int i = 0; i < vn; i++)
            if ((abs(sa[i] - 4.0*PI) < 1e-6) == isOnFace(i)) {
                printf("%d  %.2lf  %d %d\n", i, sa[i]/PI,
                    (int)isConstrained[0][i], (int)applySurfaceConstraints[i]);
                assert(false);
            }

        // make sure all faces are good
        std::map<ivec3, int, decltype(ivec3Cmp)> fcount(ivec3Cmp);
        for (int i = 0; i < tn; i++) {
            for (int _ = 0; _ < 4; _++) {
                ivec3 f;
                for (int __ = 0; __ < 3; __++)
                    f[__] = tets[i][FACES[_][__]];
                f = rotateIvec3(f);
                assert(faces.find(ivec3(f.x, f.z, f.y)) == faces.end());
                fcount[ivec3(f.x, min(f.y, f.z), max(f.y, f.z))] += 1;
            }
        }
        for (std::pair<ivec3, int> p : fcount) {
            ivec3 f1 = p.first;
            ivec3 f2 = ivec3(f1.x, f1.z, f1.y);
            if (p.second == 1)
                assert(faces.find(f1) != faces.end() ||
                       faces.find(f2) != faces.end());
            else if (p.second == 2)
                assert(faces.find(f1) == faces.end() &&
                       faces.find(f2) == faces.end());
            else assert(false);
        }
        for (ivec3 f : faces) {
            if (f.z < f.y) std::swap(f.y, f.z);
            assert(fcount.find(f) != fcount.end());
        }

        // make sure all surface edges are good
        std::map<uint64_t, int> ecount;
        for (ivec3 f : faces) {
            for (int _ = 0; _ < 3; _++) {
                ecount[edgeKey(f[_], f[(_ + 1) % 3])] += 1;
            }
        }
        for (std::pair<uint64_t, int> ec : ecount) {
            assert(ec.second == 2);
            assert(edges.find(ec.first) != edges.end());
            ecount[ec.first] = 0;
        }
        for (uint64_t e : edges) {
            assert(ecount.find(e) != ecount.end());
        }
        for (ivec4 t : tets) {
            for (int i = 0; i < 6; i++) {
                uint64_t key = edgeKey(t[EDGES[i][0]], t[EDGES[i][1]]);
                if (ecount.find(key) != ecount.end())
                    ecount[key] += 1;
            }
        }
        for (std::pair<uint64_t, int> ec : ecount) {
            assert(ec.second >= 2);
            ecount[ec.first] = 0;
        }
        for (ivec4 t : tets) {
            for (int fi = 0; fi < 4; fi++) {
                ivec3 f;
                for (int _ = 0; _ < 3; _++)
                    f[_] = t[FACES[fi][_]];
                f = rotateIvec3(f);
                if (faces.find(f) == faces.end())
                    continue;
                for (int _ = 0; _ < 3; _++)
                    ecount[edgeKey(f[_], f[(_ + 1) % 3])] += 1;
            }
        }
        for (std::pair<uint64_t, int> ec : ecount) {
            assert(ec.second == 2);
        }
    }

}


MESHGEN_TET_IMPLICIT_NS_END
