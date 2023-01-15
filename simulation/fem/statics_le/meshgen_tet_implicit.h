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
const int FACES[4][3] = {  // ccw, missing the index
    {1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}
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



double vanDerCorput(int n, int b) {
    double x = 0.0, e = 1.0 / b;
    while (n) {
        x += (n % b) * e;
        e /= b, n /= b;
    }
    return x;
}


void generateInitialTetrahedraInBox(
    ScalarFieldF F0, vec3 bc, vec3 br, double tetSize,
    std::vector<MeshVertex>& vertices, std::vector<ivec4>& tets
) {
    assert(vertices.empty() && tets.empty());
    ScalarFieldF F = [&](double x, double y, double z) {
        double f = F0(x, y, z);
        double b = std::max({
            abs(x - bc.x) - br.x,
            abs(y - bc.y) - br.y,
            abs(z - bc.z) - br.z });
        return max(f, b);
    };
    vec3 c = bc;
    for (int i = 1; i < 65536 && !(F0(c.x, c.y, c.z) < 0.); i++) {
        vec3 s(vanDerCorput(i, 2), vanDerCorput(i, 3), vanDerCorput(i, 5));
        c = bc + br * (2.0 * s - 1.0);
    }
    vertices.assign({
        MeshVertex{ c },
        MeshVertex{ c + tetSize * vec3(1, 1, 0) },
        MeshVertex{ c + tetSize * vec3(0, 1, 1) },
        MeshVertex{ c + tetSize * vec3(1, 0, 1) }
        });
    tets.push_back(ivec4(0, 1, 2, 3));
    generateInitialTetrahedra(F, vertices, tets);
}


MESHGEN_TET_IMPLICIT_NS_END
