// FEM elements

#pragma once

#include <cmath>
#include <initializer_list>
#include <cassert>

#if SUPPRESS_ASSERT
#undef assert
#define assert(x) 0
#endif

#ifndef PI
#define PI 3.1415926535897932384626
#endif

#undef max
#undef min
#undef abs
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))
template<typename T> T abs(T x) { return x > 0 ? x : -x; }

#include "glm/glm.hpp"
using glm::clamp; using glm::mix; using glm::sign;
using glm::vec2; using glm::vec3; using glm::vec4;
using glm::dot; using glm::cross; using glm::outerProduct;
using glm::mat2; using glm::mat3; using glm::mat4; using glm::mat2x3; using glm::mat3x2;
using glm::inverse; using glm::transpose; using glm::determinant;


// timer
#include <chrono>
auto _TIME_START = std::chrono::high_resolution_clock::now();
float getTimePast() {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float>(t1 - _TIME_START).count();
}


/* Indices */

struct ivec2 {
    int x, y;
    ivec2(int a = 0):x(a), y(a) {}
    ivec2(int a, int b):x(a), y(b) {}
    int& operator[](int i) { return (&x)[i]; }
    const int& operator[](int i) const { return (&x)[i]; }
    ivec2 operator + (const ivec2& v) const { return ivec2(x + v.x, y + v.y); }
};

struct ivec3 {
    int x, y, z;
    ivec3(int a = 0):x(a), y(a), z(a) {}
    ivec3(int a, int b, int c):x(a), y(b), z(c) {}
    int& operator[](int i) { return (&x)[i]; }
    const int& operator[](int i) const { return (&x)[i]; }
    ivec3 operator - () const { return ivec3(-x, -y, -z); }
    ivec3 operator + (const ivec3& v) const { return ivec3(x + v.x, y + v.y, z + v.z); }
    ivec3 operator - (const ivec3& v) const { return ivec3(x - v.x, y - v.y, z - v.z); }
    ivec3 operator * (const ivec3& v) const { return ivec3(x * v.x, y * v.y, z * v.z); }
    ivec3 operator / (const ivec3& v) const { return ivec3(x / v.x, y / v.y, z / v.z); }
    ivec3 operator % (const ivec3& v) const { return ivec3(x % v.x, y % v.y, z % v.z); }
    ivec3 operator & (const ivec3& v) const { return ivec3(x & v.x, y & v.y, z & v.z); }
    ivec3 operator | (const ivec3& v) const { return ivec3(x | v.x, y | v.y, z | v.z); }
    ivec3 operator ^ (const ivec3& v) const { return ivec3(x ^ v.x, y ^ v.y, z ^ v.z); }
    ivec3 operator * (const int& k) const { return ivec3(x * k, y * k, z * k); }
    ivec3 operator / (const int& k) const { return ivec3(x / k, y / k, z / k); }
    ivec3 operator % (const int& k) const { return ivec3(x % k, y % k, z % k); }
    friend ivec3 operator * (const int& a, const ivec3& v) { return ivec3(a * v.x, a * v.y, a * v.z); }
    friend int dot(ivec3 u, ivec3 v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
    friend ivec3 cross(ivec3 u, ivec3 v) { return ivec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x); }
    friend int det(ivec3 a, ivec3 b, ivec3 c) { return dot(a, cross(b, c)); }  // be careful about overflow
    bool operator == (const ivec3& v) const { return x == v.x && y == v.y && z == v.z; }
    bool operator != (const ivec3& v) const { return x != v.x || y != v.y || z != v.z; }
};

struct ivec4 {
    int x, y, z, w;
    ivec4(int a = 0):x(a), y(a), z(a), w(a) {}
    ivec4(int a, int b, int c, int d):x(a), y(b), z(c), w(d) {}
    ivec4(const int* p):x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}
    int& operator[](int i) { return (&x)[i]; }
    const int& operator[](int i) const { return (&x)[i]; }
};

struct ivec8 {
    int a, b, c, d, e, f, g, h;
    ivec8(int x = 0): a(x), b(x), c(x), d(x), e(x), f(x), g(x), h(x) {}
    ivec8(int a, int b, int c, int d, int e, int f, int g, int h)
        :a(a), b(b), c(c), d(d), e(e), f(f), g(g), h(g) {}
    ivec8(const int* p) {
        for (int i = 0; i < 8; i++)
            ((int*)this)[i] = p[i];
    }
};



/* FEM Elements */


#define MAX_AREA_ELEMENT_N 4
#define MAX_AREA_ELEMENT_TN 1
#define MAX_AREA_ELEMENT_EN 3

struct AreaElement {
protected:


public:
    virtual ~AreaElement() {};

    // get the number of vertices
    virtual int getN() const = 0;

    // get vertice indices
    virtual const int* getVi() const = 0;

    // get triangles and edges for rendering
    virtual int getNumTriangles() const = 0;
    virtual void getTriangles(ivec3 ts[]) const = 0;
    virtual int getNumEdges() const = 0;
    virtual void getEdges(ivec2 es[]) const = 0;

    // get the area of the element
    virtual float getArea() const = 0;

    // evaluate the stiffness matrix
    // X: a list of global vertex initial positions
    // K: the NxN stiffness matrix
    virtual void evalK(const vec2* X, float* K) const = 0;

};



struct LinearTrigElement: AreaElement {
private:
    // 3 vertices, positive area
    int vi[3];

    // area of the element
    float dA;

public:

    // get volume
    float getArea() const { return dA; }

    // vi: the global indices of vertices
    // X: the initial global vertex positions
    LinearTrigElement(int vi[3], const vec2* X) {
        for (int i = 0; i < 3; i++)
            this->vi[i] = vi[i];
        // if (X != nullptr) calcDV(X);
    }
    LinearTrigElement(std::initializer_list<int> vi, const vec2* X) {
        for (int i = 0; i < 3; i++)
            this->vi[i] = vi.begin()[i];
        // if (X != nullptr) calcDV(X);
    }
    LinearTrigElement() { }

    // get the number of vertices
    int getN() const { return 3; }

    // get vertice indices
    const int* getVi() const { return &vi[0]; }

    // get triangles and edges for rendering
    int getNumTriangles() const { return 1; }
    void getTriangles(ivec3 ts[1]) const {
        ts[0] = ivec3(vi[0], vi[1], vi[2]);
    }
    int getNumEdges() const { return 3; }
    void getEdges(ivec2 es[3]) const {
        es[0] = ivec2(vi[0], vi[1]);
        es[1] = ivec2(vi[0], vi[2]);
        es[2] = ivec2(vi[1], vi[2]);
    }

    // evaluate the stiffness matrix (3x3)
    void evalK(const vec2* Xs, float* K) const {
        assert(sizeof(mat3) == 9*sizeof(float));
        mat2 X = inverse(transpose(mat2(
            Xs[vi[1]]-Xs[vi[0]],
            Xs[vi[2]]-Xs[vi[0]]
            )));
        mat3x2 D = X * mat3x2(-1, -1, 1, 0, 0, 1);
        *((mat3*)K) = 0.5f * transpose(D) * D;
    }

};
