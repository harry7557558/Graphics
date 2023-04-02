// miscellaneous mesh generation functions

#pragma once

#include <functional>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
#include <initializer_list>
#include <bitset>
#include "elements.h"


#define MESHGEN_MISC_NS_START namespace MeshgenMisc {
#define MESHGEN_MISC_NS_END }

MESHGEN_MISC_NS_START


typedef std::function<double(double, double, double)> ScalarFieldF;
typedef std::function<void(int, const vec3*, double*)> ScalarFieldFBatch;

struct MeshVertex {
    vec3 x;
    double fv;
};


// for maps/sets

// normalize a triangle to be unique while reserving the orientation
ivec3 rotateIvec3(ivec3 v) {
    assert(v.x != v.y && v.x != v.z && v.y != v.z);
    int i = v.x < v.y && v.x < v.z ? 0 :
        v.y < v.x && v.y < v.z ? 1 : 2;
    return ivec3(v[i], v[(i + 1) % 3], v[(i + 2) % 3]);
}
// compare two triangles component wise
auto ivec3Cmp = [](ivec3 a, ivec3 b) {
    return a.z != b.z ? a.z < b.z :
        *((uint64_t*)&a.x) < *((uint64_t*)&b.x);
};


// disjoint set union
class DisjointSet {
    int N;
    int *parent;
    uint8_t *rank;
public:
    DisjointSet(int N) :N(N) {
        parent = new int[N];
        rank = new uint8_t[N];
        for (int i = 0; i < N; i++)
            parent[i] = -1, rank[i] = 0;
    }

    // find representatitve
    int findRep(int i) {
        if (parent[i] == -1)
            return i;
        int ans = findRep(parent[i]);
        parent[i] = ans;
        return ans;
    }

    // set union, returns False if already merged
    bool unionSet(int i, int j) {
        int i_rep = findRep(i);
        int j_rep = findRep(j);
        if (i_rep == j_rep) return false;
        if (rank[i_rep] < rank[j_rep])
            parent[i_rep] = j_rep;
        else if (rank[i_rep] > rank[j_rep])
            parent[j_rep] = i_rep;
        else parent[j_rep] = i_rep, rank[i_rep]++;
        return true;
    }

    // map index to compressed index, returns size
    int getCompressedMap(std::vector<int> &res) {
        std::vector<int> uncompressed(N, -1);
        int count = 0;
        for (int i = 0; i < N; i++)
            if (findRep(i) == i)
                uncompressed[i] = count++;
        res.resize(N);
        for (int i = 0; i < N; i++) {
            int rep = findRep(i);
            res[i] = uncompressed[rep];
        }
        return count;
    }

};


// memory saver
template<typename T>
void freeVector(std::vector<T> &v) {
    v.clear();
    v.shrink_to_fit();
}


// math
double solidAngle(vec3 a, vec3 b, vec3 c) {
    // https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
    double la = length(a), lb = length(b), lc = length(c);
    double n = dot(a, cross(b, c));
    double m = la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
    double ans = 2.0 * atan(n / m);
    if (n * ans < 0.0) ans += (n < 0.0 ? -2.0 * PI : 2.0 * PI);
    return ans;
}


MESHGEN_MISC_NS_END
