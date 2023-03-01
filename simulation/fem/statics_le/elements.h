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
template<typename T> T max(T x, T y) { return (x > y ? x : y); }
template<typename T> T min(T x, T y) { return (x < y ? x : y); }
template<typename T, typename t> T clamp(T x, t a, t b) { return (x<a ? a : x>b ? b : x); }
template<typename T, typename f> T mix(T x, T y, f a) { return (x * (f(1) - a) + y * a); }  // lerp
template<typename T> T abs(T x) { return x > 0 ? x : -x; }
template<typename T> T sign(T x) { return (T)(x > 0 ? 1 : x < 0 ? -1 : 0); }


/* Indices */

struct ivec2 {
    int x, y;
    ivec2(int a = 0):x(a), y(a) {}
    ivec2(int a, int b):x(a), y(b) {}
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


/* Matrix/Vector */

struct vec3 {
    double x, y, z;
    vec3() {}
    explicit vec3(double a):x(a), y(a), z(a) {}
    explicit vec3(double x, double y, double z):x(x), y(y), z(z) {}
    explicit vec3(ivec3 p):x(p.x), y(p.y), z(p.z) {}
    vec3 operator - () const { return vec3(-x, -y, -z); }
    vec3 operator + (const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    vec3 operator - (const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    vec3 operator * (const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }  // element wise
    vec3 operator / (const vec3& v) const { return vec3(x / v.x, y / v.y, z / v.z); }
    vec3 operator + (const double& k) const { return vec3(x + k, y + k, z + k); }
    vec3 operator - (const double& k) const { return vec3(x - k, y - k, z - k); }
    vec3 operator * (const double& k) const { return vec3(x * k, y * k, z * k); }
    vec3 operator / (const double& a) const { return vec3(x / a, y / a, z / a); }
    void operator += (const vec3& v) { x += v.x, y += v.y, z += v.z; }
    void operator -= (const vec3& v) { x -= v.x, y -= v.y, z -= v.z; }
    void operator *= (const vec3& v) { x *= v.x, y *= v.y, z *= v.z; }
    void operator /= (const vec3& v) { x /= v.x, y /= v.y, z /= v.z; }
    friend vec3 operator * (const double& a, const vec3& v) { return vec3(a * v.x, a * v.y, a * v.z); }
    void operator += (const double& a) { x += a, y += a, z += a; }
    void operator -= (const double& a) { x -= a, y -= a, z -= a; }
    void operator *= (const double& a) { x *= a, y *= a, z *= a; }
    void operator /= (const double& a) { x /= a, y /= a, z /= a; }
    double sqr() const { return x * x + y * y + z * z; }
    friend double length(vec3 v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
    friend vec3 normalize(vec3 v) { double m = 1.0 / sqrt(v.x * v.x + v.y * v.y + v.z * v.z); return vec3(v.x * m, v.y * m, v.z * m); }
    friend double dot(vec3 u, vec3 v) { return u.x * v.x + u.y * v.y + u.z * v.z; }
    friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x); }
    friend double det(vec3 a, vec3 b, vec3 c) { return dot(a, cross(b, c)); }
    friend double ndot(const vec3& u, const vec3& v) { return dot(u, v) / sqrt(u.sqr() * v.sqr()); }
    friend vec3 ncross(vec3 u, vec3 v) { return normalize(cross(u, v)); }
    bool operator == (const vec3& v) const { return x == v.x && y == v.y && z == v.z; }
    bool operator != (const vec3& v) const { return x != v.x || y != v.y || z != v.z; }
};

struct mat3 {
    double v[3][3];
    mat3() {}
    explicit mat3(double k) {  // diagonal matrix
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            v[i][j] = k * (i == j);
    }
    explicit mat3(vec3 lambda) {  // diagonal matrix
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
            v[i][j] = i == j ? ((double*)&lambda)[i] : 0;
    }
    explicit mat3(vec3 lambda, double xy, double xz, double yz) {  // symmetric matrix
        v[0][0] = lambda.x, v[1][1] = lambda.y, v[2][2] = lambda.z;
        v[0][1] = v[1][0] = xy, v[0][2] = v[2][0] = xz, v[1][2] = v[2][1] = yz;
    }
    explicit mat3(vec3 i, vec3 j, vec3 k) {  // matrix by column vectors
        for (int u = 0; u < 3; u++) v[u][0] = ((double*)&i)[u], v[u][1] = ((double*)&j)[u], v[u][2] = ((double*)&k)[u];
    }
    explicit mat3(double _00, double _10, double _20, double _01, double _11, double _21, double _02, double _12, double _22) {  // ordered in column-wise
        v[0][0] = _00, v[0][1] = _01, v[0][2] = _02, v[1][0] = _10, v[1][1] = _11, v[1][2] = _12, v[2][0] = _20, v[2][1] = _21, v[2][2] = _22;
    }
    double* operator[] (int d) { return &v[d][0]; }
    const double* operator[] (int d) const { return &v[d][0]; }
    vec3 row(int i) const { return vec3(v[i][0], v[i][1], v[i][2]); }
    vec3 column(int i) const { return vec3(v[0][i], v[1][i], v[2][i]); }
    vec3 diag() const { return vec3(v[0][0], v[1][1], v[2][2]); }
    void operator += (const mat3& m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] += (&m.v[0][0])[i]; }
    void operator -= (const mat3& m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] -= (&m.v[0][0])[i]; }
    void operator *= (double m) { for (int i = 0; i < 9; i++) (&v[0][0])[i] *= m; }
    mat3 operator + (const mat3& m) const { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] + (&m.v[0][0])[i]; return r; }
    mat3 operator - (const mat3& m) const { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] - (&m.v[0][0])[i]; return r; }
    mat3 operator * (double m) const { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = (&v[0][0])[i] * m; return r; }
    friend mat3 operator * (double a, const mat3& m) { mat3 r; for (int i = 0; i < 9; i++) (&r.v[0][0])[i] = a * (&m.v[0][0])[i]; return r; }
    friend double determinant(const mat3& m) { return m.v[0][0] * (m.v[1][1] * m.v[2][2] - m.v[1][2] * m.v[2][1]) - m.v[0][1] * (m.v[1][0] * m.v[2][2] - m.v[1][2] * m.v[2][0]) + m.v[0][2] * (m.v[1][0] * m.v[2][1] - m.v[1][1] * m.v[2][0]); }
    mat3 transpose() const { mat3 r; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.v[i][j] = v[j][i]; return r; }
    friend mat3 transpose(const mat3& m) { return mat3(m.v[0][0], m.v[0][1], m.v[0][2], m.v[1][0], m.v[1][1], m.v[1][2], m.v[2][0], m.v[2][1], m.v[2][2]); }
    friend double trace(const mat3& m) { return m.v[0][0] + m.v[1][1] + m.v[2][2]; }
    friend double sumsqr(const mat3& m) { double r = 0; for (int i = 0; i < 9; i++) r += (&m.v[0][0])[i] * (&m.v[0][0])[i]; return r; }  // sum of square of elements

    mat3 operator * (const mat3& A) const {
        mat3 R;
        for (int m = 0; m < 3; m++) for (int n = 0; n < 3; n++) {
            R.v[m][n] = 0;
            for (int i = 0; i < 3; i++) R.v[m][n] += v[m][i] * A.v[i][n];
        }
        return R;
    }
    vec3 operator * (const vec3& a) const {
        return vec3(
            v[0][0] * a.x + v[0][1] * a.y + v[0][2] * a.z,
            v[1][0] * a.x + v[1][1] * a.y + v[1][2] * a.z,
            v[2][0] * a.x + v[2][1] * a.y + v[2][2] * a.z);
    }
    mat3 inverse_() const {
        mat3 R;
        R.v[0][0] = v[1][1] * v[2][2] - v[1][2] * v[2][1], R.v[0][1] = v[0][2] * v[2][1] - v[0][1] * v[2][2], R.v[0][2] = v[0][1] * v[1][2] - v[0][2] * v[1][1];
        R.v[1][0] = v[1][2] * v[2][0] - v[1][0] * v[2][2], R.v[1][1] = v[0][0] * v[2][2] - v[0][2] * v[2][0], R.v[1][2] = v[0][2] * v[1][0] - v[0][0] * v[1][2];
        R.v[2][0] = v[1][0] * v[2][1] - v[1][1] * v[2][0], R.v[2][1] = v[0][1] * v[2][0] - v[0][0] * v[2][1], R.v[2][2] = v[0][0] * v[1][1] - v[0][1] * v[1][0];
        double m = 1.0 / (v[0][0] * R.v[0][0] + v[0][1] * R.v[1][0] + v[0][2] * R.v[2][0]);
        R.v[0][0] *= m, R.v[0][1] *= m, R.v[0][2] *= m, R.v[1][0] *= m, R.v[1][1] *= m, R.v[1][2] *= m, R.v[2][0] *= m, R.v[2][1] *= m, R.v[2][2] *= m;
        return R;
    }
    friend mat3 inverse(const mat3& A) { return A.inverse_(); }

};

mat3 tensor(vec3 u, vec3 v) { return mat3(u * v.x, u * v.y, u * v.z); }
mat3 axis_angle(vec3 n, double a) {
    n = normalize(n); double ct = cos(a), st = sin(a);
    return mat3(
        ct + n.x * n.x * (1 - ct), n.y * n.x * (1 - ct) + n.z * st, n.z * n.x * (1 - ct) - n.y * st,
        n.x * n.y * (1 - ct) - n.z * st, ct + n.y * n.y * (1 - ct), n.z * n.y * (1 - ct) + n.x * st,
        n.x * n.z * (1 - ct) + n.y * st, n.y * n.z * (1 - ct) - n.x * st, ct + n.z * n.z * (1 - ct)
    );
}


// eigenvalues of a symmetric matrix
vec3 eigvalsh(mat3 A) {
    double tj[3], ti[3];
    for (int d = 0; d < 64; d++) {
        double err = 0.;
        for (int i = 0; i < 3; i++) for (int j = 0; j < i; j++) {
            err += A[i][j] * A[i][j];
            // calculate given rotation
            double a = A[j][j], b = A[i][i], d = A[i][j];
            double t = 0.5 * atan2(2.0 * d, a - b);
            double c = cos(t), s = sin(t);
            // apply inverse rotation to left side
            for (int k = 0; k < 3; k++) {
                tj[k] = c * A[j][k] + s * A[i][k];
                ti[k] = c * A[i][k] - s * A[j][k];
            }
            for (int k = 0; k < 3; k++)
                A[j][k] = tj[k], A[i][k] = ti[k];
            // apply rotation to right side
            for (int k = 0; k < 3; k++) {
                tj[k] = c * A[k][j] + s * A[k][i];
                ti[k] = c * A[k][i] - s * A[k][j];
            }
            for (int k = 0; k < 3; k++)
                A[k][j] = tj[k], A[k][i] = ti[k];
        }
        if (err < 1e-32) break;
    }
    return vec3(A[0][0], A[1][1], A[2][2]);
}


/* Applied Forces */

// triangular surface: ccw outward normal
struct ElementForce3: ivec3 {
    vec3 F;
    ElementForce3(ivec3 v, vec3 f) :
        ivec3(v), F(f) {}
};

// tetrahedral volume: any order
// quad surface: ccw outward normal
struct ElementForce4: ivec4 {
    vec3 F;
    ElementForce4(ivec4 v, vec3 f) :
        ivec4(v), F(f) {}
};

// brick volume:
// 000, 100, 110, 010, 001, 101, 111, 011
struct ElementForce8: ivec8 {
    vec3 F;
    ElementForce8(ivec8 v, vec3 f) :
        ivec8(v), F(f) {}
};



/* FEM Elements */


#define MAX_SOLID_ELEMENT_N 20
#define MAX_SOLID_ELEMENT_TN 48
#define MAX_SOLID_ELEMENT_EN 24

struct SolidElement {
protected:

    // evaluate the gradient of the linearized strain
    // N: number of vertices in the FEM element
    // D: gradient of deformation gradient, 3xN
    // SD: 6x3N
    static void getLinearizedStrainGradient(int N, const double* D, double* SD) {
        for (int i = 0; i < 18 * N; i++)
            SD[i] = 0.0;
        for (int i = 0; i < 3 * N; i += 3) {
            double* sd[6] = {
                &SD[0 * N + i], &SD[3 * N + i], &SD[6 * N + i],
                &SD[9 * N + i], &SD[12 * N + i], &SD[15 * N + i]
            };
            sd[0][0] = sd[4][2] = sd[5][1] = D[i / 3];
            sd[1][1] = sd[3][2] = sd[5][0] = D[N + i / 3];
            sd[2][2] = sd[3][1] = sd[4][0] = D[2 * N + i / 3];
        }
    }

    // evaluate the stiffness matrix of an element
    // N: number of vertices in the element
    // V: volume of the element
    // SD: getLinearizedStrainGradient(), 6x3N
    // C: getStressStrainMatrix(), 6x6
    // K: stiffness matrix, 3Nx3N
    static void getElementStiffnessMatrix(
        int N, double V, const double* SD, const double* C, double* K
    ) {
        for (int i = 0; i < 3 * N; i++) {
            for (int j = 0; j < 3 * N; j++) {
                double s = 0.0;
                for (int u = 0; u < 6; u++) for (int v = 0; v < 6; v++)
                    s += C[6 * u + v] * SD[3 * N * u + i] * SD[3 * N * v + j];
                K[3 * N * i + j] = V * s;
            }
        }
    }

    // convert 6-component stress to 3x3 stress tensor
    static mat3 stress2tensor(const double sigma[6]) {
        return mat3(
            sigma[0], sigma[5], sigma[4],
            sigma[5], sigma[1], sigma[3],
            sigma[4], sigma[3], sigma[2]
        );
    }

public:

    // get the number of vertices
    virtual int getN() const = 0;

    // get vertice indices
    virtual const int* getVi() const = 0;

    // get triangles and edges for rendering
    virtual int getNumTriangles() const = 0;
    virtual void getTriangles(ivec3 ts[]) const = 0;
    virtual int getNumEdges() const = 0;
    virtual void getEdges(ivec2 es[]) const = 0;

    // precompute D and dV
    // X: global list of vertices
    virtual void calcDV(const vec3* X) = 0;

    // get the undeformed volume of the element
    virtual double getVolume() const = 0;

    // evaluate the stiffness matrix
    // X: a list of global vertex initial positions
    // C: a 6x6 matrix transforming strain to stress
    // K: the 3Nx3N stiffness matrix
    virtual void evalK(const vec3* X, const double* C, double* K) const = 0;

    // add the forces and stresses at joints exerted by the element
    // X: a list of global vertex initial positions
    // C: strain to stress matrix
    // U: a list of global vertex deflections
    // To calculate the stresses of the structure at joints:
    //   - Initialize Sigma and SigmaW to zero, same length as vertex list
    //   - Call this function for each element
    //   - Divide Sigma by SigmaW for stresses
    virtual void addStress(
        const vec3* X, const double* C, const vec3* U,
        mat3* Sigma, double* SigmaW) const = 0;
};



struct LinearTetrahedralElement: SolidElement {
private:
    // 4 vertices, positive volume
    int vi[4];

    // precomputed D, ∂∇u/∂u
    double D[3][4];
    // volume of the element
    double dV;

public:

    // calculate D and dV
    void calcDV(const vec3* X) {
        mat3 X3 = transpose(mat3(
            X[vi[1]] - X[vi[0]],
            X[vi[2]] - X[vi[0]],
            X[vi[3]] - X[vi[0]]
        ));
        dV = determinant(X3) / 6.0;
        assert(dV > 0.0);
        mat3 invX = inverse(X3);
        for (int i = 0; i < 3 * 4; i++) (&D[0][0])[i] = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                D[i][0] -= invX[i][j];
                D[i][j + 1] += invX[i][j];
            }
        }
    }

    // get volume
    double getVolume() const { return dV; }

    // vi: the global indices of vertices
    // X: the initial global vertex positions
    LinearTetrahedralElement(int vi[4], const vec3* X) {
        for (int i = 0; i < 4; i++)
            this->vi[i] = vi[i];
        if (X != nullptr) calcDV(X);
    }
    LinearTetrahedralElement(std::initializer_list<int> vi, const vec3* X) {
        for (int i = 0; i < 4; i++)
            this->vi[i] = vi.begin()[i];
        if (X != nullptr) calcDV(X);
    }
    LinearTetrahedralElement() { }

    // get the number of vertices
    int getN() const { return 4; }

    // get vertice indices
    const int* getVi() const { return &vi[0]; }

    // get triangles and edges for rendering
    int getNumTriangles() const { return 4; }
    void getTriangles(ivec3 ts[4]) const {
        ts[0] = ivec3(vi[0], vi[2], vi[1]);
        ts[1] = ivec3(vi[0], vi[1], vi[3]);
        ts[2] = ivec3(vi[0], vi[3], vi[2]);
        ts[3] = ivec3(vi[1], vi[2], vi[3]);
    }
    int getNumEdges() const { return 6; }
    void getEdges(ivec2 es[6]) const {
        es[0] = ivec2(vi[0], vi[1]);
        es[1] = ivec2(vi[0], vi[2]);
        es[2] = ivec2(vi[0], vi[3]);
        es[3] = ivec2(vi[1], vi[2]);
        es[4] = ivec2(vi[1], vi[3]);
        es[5] = ivec2(vi[2], vi[3]);
    }

    // evaluate the stiffness matrix (12x12)
    void evalK(const vec3* X, const double* C, double* K) const {
        double SD[6 * 3 * 4];
        getLinearizedStrainGradient(4, &D[0][0], SD);
        getElementStiffnessMatrix(4, dV, SD, C, K);
    }

    // used to calculate stresses at joints
    void addStress(
        const vec3* X, const double* C, const vec3* U,
        mat3* Sigma, double* SigmaW) const
    {
        vec3 SD[6][4];
        getLinearizedStrainGradient(4, &D[0][0], (double*)&SD[0][0]);
        // strain
        double epsilon[6];
        for (int i = 0; i < 6; i++) {
            epsilon[i] = 0.0;
            for (int j = 0; j < 4; j++)
                epsilon[i] += dot(SD[i][j], U[vi[j]]);
        }
        // stress
        double sigma[6];
        for (int i = 0; i < 6; i++) {
            sigma[i] = 0.0;
            for (int j = 0; j < 6; j++)
                sigma[i] += C[6 * i + j] * epsilon[j];
        }
        for (int i = 0; i < 4; i++) {
            double w = 1.0 / cbrt(dV);
            Sigma[vi[i]] += w * stress2tensor(sigma);
            SigmaW[vi[i]] += w;
        }
    }

};



struct QuadraticTetrahedralElement: SolidElement {
private:
    const static vec3 params[11];
    const static double weights[11];
    const static double W[11][3][10];
    const static double WV[10][3][10];

    // 10 vertices
    // 0, 1, 2, 3, 01, 02, 03, 12, 13, 23
    int vi[10];

    // precomputed D, ∂∇u/∂u
    double D[11][3][10];
    // volume of the element
    double dV[11];

public:

    // calculate D and dV
    void calcDV(const vec3* X) {
        for (int _ = 0; _ < 11; _++) {
            vec3 dX[3];
            for (int i = 0; i < 3; i++) {
                dX[i] = vec3(0);
                for (int j = 0; j < 10; j++)
                    dX[i] += W[_][i][j] * X[vi[j]];
            }
            mat3 X3 = transpose(mat3(
                dX[0], dX[1], dX[2]
            ));
            dV[_] = determinant(X3) / 6.0;
            assert(dV[_] > 0.0);
            mat3 invX = inverse(X3);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 10; j++) {
                    double s = 0.0;
                    for (int k = 0; k < 3; k++)
                        s += invX[i][k] * W[_][k][j];
                    D[_][i][j] = s;
                }
            }
        }
    }

    // get volume
    double getVolume() const {
        double V = 0.0;
        for (int i = 0; i < 11; i++)
            V += weights[i] * dV[i];
        return V;
    }

    // vi: the global indices of vertices
    // 0, 1, 2, 3, 01, 02, 03, 12, 13, 23
    // X: the initial global vertex positions
    QuadraticTetrahedralElement(int vi[10], const vec3* X) {
        for (int i = 0; i < 10; i++)
            this->vi[i] = vi[i];
        if (X != nullptr) calcDV(X);
    }
    // vi: the global indices of vertices
    // 0, 1, 2, 3, 01, 02, 03, 12, 13, 23
    // X: the initial global vertex positions
    QuadraticTetrahedralElement(std::initializer_list<int> vi, const vec3* X) {
        for (int i = 0; i < 10; i++)
            this->vi[i] = vi.begin()[i];
        if (X != nullptr) calcDV(X);
    }
    QuadraticTetrahedralElement() { }

    // get the number of vertices
    int getN() const { return 10; }

    // get vertice indices
    const int* getVi() const { return &vi[0]; }

    // get triangles and edges for rendering
#if 0
    int getNumTriangles() const { return 4; }
    void getTriangles(ivec3 ts[4]) const {
        ts[0] = ivec3(vi[0], vi[2], vi[1]);
        ts[1] = ivec3(vi[0], vi[1], vi[3]);
        ts[2] = ivec3(vi[0], vi[3], vi[2]);
        ts[3] = ivec3(vi[1], vi[2], vi[3]);
    }
#else
    int getNumTriangles() const { return 16; }
    void getTriangles(ivec3 t[16]) const {
        t[0] = ivec3(0, 4, 6), t[1] = ivec3(4, 1, 8), t[2] = ivec3(6, 8, 3), t[3] = ivec3(4, 8, 6);
        t[4] = ivec3(1, 7, 8), t[5] = ivec3(7, 2, 9), t[6] = ivec3(8, 9, 3), t[7] = ivec3(8, 7, 9);
        t[8] = ivec3(2, 5, 9), t[9] = ivec3(5, 0, 6), t[10] = ivec3(9, 6, 3), t[11] = ivec3(9, 5, 6);
        t[12] = ivec3(0, 5, 4), t[13] = ivec3(4, 7, 1), t[14] = ivec3(5, 2, 7), t[15] = ivec3(5, 7, 4);
        for (int i = 0; i < 16; i++) {
            t[i].x = vi[t[i].x];
            t[i].y = vi[t[i].y];
            t[i].z = vi[t[i].z];
        }
    }
#endif
    int getNumEdges() const { return 12; }
    void getEdges(ivec2 es[12]) const {
        int e[24] = {
            0, 4, 4, 1, 0, 5, 5, 2, 0, 6, 6, 3,
            1, 7, 7, 2, 1, 8, 8, 3, 3, 9, 9, 2
        };
        for (int i = 0; i < 24; i++)
            ((int*)&es[0])[i] = vi[e[i]];
    }

    // evaluate the stiffness matrix (30x30)
    void evalK(const vec3* X, const double* C, double* K) const {
        double SD[6 * 3 * 10];
        double Kt[9 * 10 * 10];
        for (int i = 0; i < 900; i++) K[i] = 0.0;
        for (int _ = 0; _ < 11; _++) {
            getLinearizedStrainGradient(10, &D[_][0][0], SD);
            getElementStiffnessMatrix(10, dV[_], SD, C, Kt);
            for (int i = 0; i < 900; i++)
                K[i] += weights[_] * Kt[i];
        }
    }

    // used to calculate force densities and stresses at joints
    void addStress(
        const vec3* X, const double* C, const vec3* U,
        mat3* Sigma, double* SigmaW) const
    {
#if 1
        // strain gradient
        double D[10][3][10], dV[10];
        for (int _ = 0; _ < 10; _++) {
            vec3 dX[3];
            for (int i = 0; i < 3; i++) {
                dX[i] = vec3(0);
                for (int j = 0; j < 10; j++)
                    dX[i] += WV[_][i][j] * X[vi[j]];
            }
            mat3 X3 = transpose(mat3(
                dX[0], dX[1], dX[2]
            ));
            dV[_] = determinant(X3) / 6.0;
            assert(dV[_] > 0.0);
            mat3 invX = inverse(X3);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 10; j++) {
                    double s = 0.0;
                    for (int k = 0; k < 3; k++)
                        s += invX[i][k] * WV[_][k][j];
                    D[_][i][j] = s;
                }
            }
        }
        for (int _ = 0; _ < 10; _++) {
            vec3 SD[6][10];
            getLinearizedStrainGradient(10, &D[_][0][0], (double*)&SD[0][0]);
            // strain
            double epsilon[6] = { 0,0,0,0,0,0 };
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 10; j++)
                    epsilon[i] += dot(SD[i][j], U[vi[j]]);
            // stress
            double sigma[6];
            for (int i = 0; i < 6; i++) {
                sigma[i] = 0.0;
                for (int j = 0; j < 6; j++)
                    sigma[i] += C[6 * i + j] * epsilon[j];
            }
            double w = 1.0 / cbrt(dV[_]);
            Sigma[vi[_]] += w * stress2tensor(sigma);
            SigmaW[vi[_]] += w;
        }
#else
        // more stable but might involve some diffusion?
        vec3 SD[6][10]; double sd[6 * 3 * 10];
        for (int i = 0; i < 60; i++) (&SD[0][0])[i] = vec3(0);
        double V = 0.0;
        for (int _ = 0; _ < 11; _++) {
            getLinearizedStrainGradient(10, &D[_][0][0], sd);
            for (int i = 0; i < 6 * 3 * 10; i++)
                ((double*)&SD[0][0])[i] += weights[_] * sd[i];
            V += dV[_];
        }
        double epsilon[6] = { 0,0,0,0,0,0 };
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 10; j++)
                epsilon[i] += dot(SD[i][j], U[vi[j]]);
        double sigma[6] = { 0,0,0,0,0,0 };
        for (int i = 0; i < 6; i++)
            for (int j = 0; j < 6; j++)
                sigma[i] += C[6 * i + j] * epsilon[j];
        mat3 sigmaT = stress2tensor(sigma);
        for (int _ = 0; _ < 10; _++) {
            double w = 1.0 / cbrt(V);
            Sigma[vi[_]] += w * sigmaT;
            SigmaW[vi[_]] += w;
        }
#endif
    }

};

const vec3 QuadraticTetrahedralElement::params[11] = {
    vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1),
    vec3(0.5, 0, 0), vec3(0.5, 0.5, 0), vec3(0, 0.5, 0), vec3(0, 0, 0.5),
    vec3(0.5, 0, 0.5), vec3(0, 0.5, 0.5), vec3(0.25, 0.25, 0.25)
};
const double QuadraticTetrahedralElement::weights[11] = {
    0.016666666666666666,0.016666666666666666,0.016666666666666666,0.016666666666666666,
    0.06666666666666667,0.06666666666666667,0.06666666666666667,0.06666666666666667,
    0.06666666666666667,0.06666666666666667,0.5333333333333333
};
const double QuadraticTetrahedralElement::W[11][3][10] = {
    {{-3,-1,0,0,4,0,0,0,0,0}, {-3,0,-1,0,0,4,0,0,0,0}, {-3,0,0,-1,0,0,4,0,0,0}},
    {{1,3,0,0,-4,0,0,0,0,0}, {1,0,-1,0,-4,0,0,4,0,0}, {1,0,0,-1,-4,0,0,0,4,0}},
    {{1,-1,0,0,0,-4,0,4,0,0}, {1,0,3,0,0,-4,0,0,0,0}, {1,0,0,-1,0,-4,0,0,0,4}},
    {{1,-1,0,0,0,0,-4,0,4,0}, {1,0,-1,0,0,0,-4,0,0,4}, {1,0,0,3,0,0,-4,0,0,0}},
    {{-1,1,0,0,0,0,0,0,0,0}, {-1,0,-1,0,-2,2,0,2,0,0}, {-1,0,0,-1,-2,0,2,0,2,0}},
    {{1,1,0,0,-2,-2,0,2,0,0}, {1,0,1,0,-2,-2,0,2,0,0}, {1,0,0,-1,-2,-2,0,0,2,2}},
    {{-1,-1,0,0,2,-2,0,2,0,0}, {-1,0,1,0,0,0,0,0,0,0}, {-1,0,0,-1,0,-2,2,0,0,2}},
    {{-1,-1,0,0,2,0,-2,0,2,0}, {-1,0,-1,0,0,2,-2,0,0,2}, {-1,0,0,1,0,0,0,0,0,0}},
    {{1,1,0,0,-2,0,-2,0,2,0}, {1,0,-1,0,-2,0,-2,2,0,2}, {1,0,0,1,-2,0,-2,0,2,0}},
    {{1,-1,0,0,0,-2,-2,2,2,0}, {1,0,1,0,0,-2,-2,0,0,2}, {1,0,0,1,0,-2,-2,0,0,2}},
    {{0,0,0,0,0,-1,-1,1,1,0}, {0,0,0,0,-1,0,-1,1,0,1}, {0,0,0,0,-1,-1,0,0,1,1}},
};
const double QuadraticTetrahedralElement::WV[10][3][10] = {
    {{-3,-1,0,0,4,0,0,0,0,0}, {-3,0,-1,0,0,4,0,0,0,0}, {-3,0,0,-1,0,0,4,0,0,0}},
    {{1,3,0,0,-4,0,0,0,0,0}, {1,0,-1,0,-4,0,0,4,0,0}, {1,0,0,-1,-4,0,0,0,4,0}},
    {{1,-1,0,0,0,-4,0,4,0,0}, {1,0,3,0,0,-4,0,0,0,0}, {1,0,0,-1,0,-4,0,0,0,4}},
    {{1,-1,0,0,0,0,-4,0,4,0}, {1,0,-1,0,0,0,-4,0,0,4}, {1,0,0,3,0,0,-4,0,0,0}},
    {{-1,1,0,0,0,0,0,0,0,0}, {-1,0,-1,0,-2,2,0,2,0,0}, {-1,0,0,-1,-2,0,2,0,2,0}},
    {{-1,-1,0,0,2,-2,0,2,0,0}, {-1,0,1,0,0,0,0,0,0,0}, {-1,0,0,-1,0,-2,2,0,0,2}},
    {{-1,-1,0,0,2,0,-2,0,2,0}, {-1,0,-1,0,0,2,-2,0,0,2}, {-1,0,0,1,0,0,0,0,0,0}},
    {{1,1,0,0,-2,-2,0,2,0,0}, {1,0,1,0,-2,-2,0,2,0,0}, {1,0,0,-1,-2,-2,0,0,2,2}},
    {{1,1,0,0,-2,0,-2,0,2,0}, {1,0,-1,0,-2,0,-2,2,0,2}, {1,0,0,1,-2,0,-2,0,2,0}},
    {{1,-1,0,0,0,-2,-2,2,2,0}, {1,0,1,0,0,-2,-2,0,0,2}, {1,0,0,1,0,-2,-2,0,0,2}},
};



struct LinearBrickElement: SolidElement {
private:
    const static vec3 params[8];
    const static double weights[8];
    const static double W[8][3][8];
    const static double WV[8][3][8];  // for vertices

    // 8 vertices
    // 000, 100, 110, 010, 001, 101, 111, 011
    // three parametric axes in order -> positive volume
    int vi[8];

    // precomputed D, ∂∇u/∂u
    double D[8][3][8];
    // volume of the element
    double dV[8];

public:

    // calculate D and dV
    void calcDV(const vec3* X) {
        for (int _ = 0; _ < 8; _++) {
            vec3 dX[3];
            for (int i = 0; i < 3; i++) {
                dX[i] = vec3(0);
                for (int j = 0; j < 8; j++)
                    dX[i] += W[_][i][j] * X[vi[j]];
            }
            mat3 X3 = transpose(mat3(
                dX[0], dX[1], dX[2]
            ));
            dV[_] = determinant(X3) * 8.0;
            assert(dV[_] > 0.0);
            mat3 invX = inverse(X3);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    double s = 0.0;
                    for (int k = 0; k < 3; k++)
                        s += invX[i][k] * W[_][k][j];
                    D[_][i][j] = s;
                }
            }
        }
    }

    // get triangles and edges for rendering
    int getNumTriangles() const { return 12; }
    void getTriangles(ivec3 ts[12]) const {
        ts[0] = ivec3(0, 1, 5), ts[1] = ivec3(0, 5, 4);
        ts[2] = ivec3(1, 2, 6), ts[3] = ivec3(1, 6, 5);
        ts[4] = ivec3(2, 3, 7), ts[5] = ivec3(2, 7, 6);
        ts[6] = ivec3(3, 0, 4), ts[7] = ivec3(3, 4, 7);
        ts[8] = ivec3(4, 5, 6), ts[9] = ivec3(4, 6, 7);
        ts[10] = ivec3(0, 3, 2), ts[11] = ivec3(0, 2, 1);
        for (int i = 0; i < 12; i++) {
            ts[i].x = vi[ts[i].x];
            ts[i].y = vi[ts[i].y];
            ts[i].z = vi[ts[i].z];
        }
    }
    int getNumEdges() const { return 12; }
    void getEdges(ivec2 es[12]) const {
        int e[24] = {
            0, 1, 1, 2, 2, 3, 3, 0,
            0, 4, 1, 5, 2, 6, 3, 7,
            4, 5, 5, 6, 6, 7, 7, 4
        };
        for (int i = 0; i < 24; i++)
            ((int*)&es[0])[i] = vi[e[i]];
    }

    // get volume
    double getVolume() const {
        double V = 0.0;
        for (int i = 0; i < 8; i++)
            V += weights[i] * dV[i];
        return V;
    }

    // vi: the global indices of vertices
    // 000, 100, 110, 010, 001, 101, 111, 011
    // X: the initial global vertex positions
    LinearBrickElement(int vi[8], const vec3* X) {
        for (int i = 0; i < 8; i++)
            this->vi[i] = vi[i];
        if (X != nullptr) calcDV(X);
    }
    // vi: the global indices of vertices
    // 000, 100, 110, 010, 001, 101, 111, 011
    // X: the initial global vertex positions
    LinearBrickElement(std::initializer_list<int> vi, const vec3* X) {
        for (int i = 0; i < 8; i++)
            this->vi[i] = vi.begin()[i];
        if (X != nullptr) calcDV(X);
    }
    LinearBrickElement() { }

    // get the number of vertices
    int getN() const { return 8; }

    // get vertice indices
    const int* getVi() const { return &vi[0]; }

    // evaluate the stiffness matrix (24x24)
    void evalK(const vec3* X, const double* C, double* K) const {
        double SD[6 * 3 * 8];
        double Kt[9 * 8 * 8];
        for (int i = 0; i < 576; i++) K[i] = 0.0;
        for (int _ = 0; _ < 8; _++) {
            getLinearizedStrainGradient(8, &D[_][0][0], SD);
            getElementStiffnessMatrix(8, dV[_], SD, C, Kt);
            for (int i = 0; i < 576; i++)
                K[i] += weights[_] * Kt[i];
        }
    }

    // used to calculate force densities and stresses at joints
    void addStress(
        const vec3* X, const double* C, const vec3* U,
        mat3* Sigma, double* SigmaW) const
    {
        // strain gradient
        double D[8][3][8], dV[8];
        for (int _ = 0; _ < 8; _++) {
            vec3 dX[3];
            for (int i = 0; i < 3; i++) {
                dX[i] = vec3(0);
                for (int j = 0; j < 8; j++)
                    dX[i] += WV[_][i][j] * X[vi[j]];
            }
            mat3 X3 = transpose(mat3(
                dX[0], dX[1], dX[2]
            ));
            dV[_] = determinant(X3) * 8.0;
            assert(dV[_] > 0.0);
            mat3 invX = inverse(X3);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 8; j++) {
                    double s = 0.0;
                    for (int k = 0; k < 3; k++)
                        s += invX[i][k] * WV[_][k][j];
                    D[_][i][j] = s;
                }
            }
        }
        for (int _ = 0; _ < 8; _++) {
            vec3 SD[6][8];
            getLinearizedStrainGradient(8, &D[_][0][0], (double*)&SD[0][0]);
            // strain
            double epsilon[6] = { 0,0,0,0,0,0 };
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 8; j++)
                    epsilon[i] += dot(SD[i][j], U[vi[j]]);
            // stress
            double sigma[6];
            for (int i = 0; i < 6; i++) {
                sigma[i] = 0.0;
                for (int j = 0; j < 6; j++)
                    sigma[i] += C[6 * i + j] * epsilon[j];
            }
            double w = 1.0 / cbrt(dV[_]);
            Sigma[vi[_]] += w * stress2tensor(sigma);
            SigmaW[vi[_]] += w;
        }
    }
};

#define C 0.5773502691896257
const vec3 LinearBrickElement::params[8] = {
    vec3(-C, -C, -C), vec3(C, -C, -C), vec3(C, C, -C), vec3(-C, C, -C),
    vec3(-C, -C, C), vec3(C, -C, C), vec3(C, C, C), vec3(-C, C, C)
};
#undef C
const double LinearBrickElement::weights[8] = {
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125
};
const double LinearBrickElement::W[8][3][8] = {
    {{-0.3110042339640731,0.3110042339640731,0.08333333333333333,-0.08333333333333333,-0.08333333333333333,0.08333333333333333,0.022329099369260225,-0.022329099369260225}, {-0.3110042339640731,-0.08333333333333333,0.08333333333333333,0.3110042339640731,-0.08333333333333333,-0.022329099369260225,0.022329099369260225,0.08333333333333333}, {-0.3110042339640731,-0.08333333333333333,-0.022329099369260225,-0.08333333333333333,0.3110042339640731,0.08333333333333333,0.022329099369260225,0.08333333333333333}},
    {{-0.3110042339640731,0.3110042339640731,0.08333333333333333,-0.08333333333333333,-0.08333333333333333,0.08333333333333333,0.022329099369260225,-0.022329099369260225}, {-0.08333333333333333,-0.3110042339640731,0.3110042339640731,0.08333333333333333,-0.022329099369260225,-0.08333333333333333,0.08333333333333333,0.022329099369260225}, {-0.08333333333333333,-0.3110042339640731,-0.08333333333333333,-0.022329099369260225,0.08333333333333333,0.3110042339640731,0.08333333333333333,0.022329099369260225}},
    {{-0.08333333333333333,0.08333333333333333,0.3110042339640731,-0.3110042339640731,-0.022329099369260225,0.022329099369260225,0.08333333333333333,-0.08333333333333333}, {-0.08333333333333333,-0.3110042339640731,0.3110042339640731,0.08333333333333333,-0.022329099369260225,-0.08333333333333333,0.08333333333333333,0.022329099369260225}, {-0.022329099369260225,-0.08333333333333333,-0.3110042339640731,-0.08333333333333333,0.022329099369260225,0.08333333333333333,0.3110042339640731,0.08333333333333333}},
    {{-0.08333333333333333,0.08333333333333333,0.3110042339640731,-0.3110042339640731,-0.022329099369260225,0.022329099369260225,0.08333333333333333,-0.08333333333333333}, {-0.3110042339640731,-0.08333333333333333,0.08333333333333333,0.3110042339640731,-0.08333333333333333,-0.022329099369260225,0.022329099369260225,0.08333333333333333}, {-0.08333333333333333,-0.022329099369260225,-0.08333333333333333,-0.3110042339640731,0.08333333333333333,0.022329099369260225,0.08333333333333333,0.3110042339640731}},
    {{-0.08333333333333333,0.08333333333333333,0.022329099369260225,-0.022329099369260225,-0.3110042339640731,0.3110042339640731,0.08333333333333333,-0.08333333333333333}, {-0.08333333333333333,-0.022329099369260225,0.022329099369260225,0.08333333333333333,-0.3110042339640731,-0.08333333333333333,0.08333333333333333,0.3110042339640731}, {-0.3110042339640731,-0.08333333333333333,-0.022329099369260225,-0.08333333333333333,0.3110042339640731,0.08333333333333333,0.022329099369260225,0.08333333333333333}},
    {{-0.08333333333333333,0.08333333333333333,0.022329099369260225,-0.022329099369260225,-0.3110042339640731,0.3110042339640731,0.08333333333333333,-0.08333333333333333}, {-0.022329099369260225,-0.08333333333333333,0.08333333333333333,0.022329099369260225,-0.08333333333333333,-0.3110042339640731,0.3110042339640731,0.08333333333333333}, {-0.08333333333333333,-0.3110042339640731,-0.08333333333333333,-0.022329099369260225,0.08333333333333333,0.3110042339640731,0.08333333333333333,0.022329099369260225}},
    {{-0.022329099369260225,0.022329099369260225,0.08333333333333333,-0.08333333333333333,-0.08333333333333333,0.08333333333333333,0.3110042339640731,-0.3110042339640731}, {-0.022329099369260225,-0.08333333333333333,0.08333333333333333,0.022329099369260225,-0.08333333333333333,-0.3110042339640731,0.3110042339640731,0.08333333333333333}, {-0.022329099369260225,-0.08333333333333333,-0.3110042339640731,-0.08333333333333333,0.022329099369260225,0.08333333333333333,0.3110042339640731,0.08333333333333333}},
    {{-0.022329099369260225,0.022329099369260225,0.08333333333333333,-0.08333333333333333,-0.08333333333333333,0.08333333333333333,0.3110042339640731,-0.3110042339640731}, {-0.08333333333333333,-0.022329099369260225,0.022329099369260225,0.08333333333333333,-0.3110042339640731,-0.08333333333333333,0.08333333333333333,0.3110042339640731}, {-0.08333333333333333,-0.022329099369260225,-0.08333333333333333,-0.3110042339640731,0.08333333333333333,0.022329099369260225,0.08333333333333333,0.3110042339640731}},
};
const double LinearBrickElement::WV[8][3][8] = {
    {{-0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0}, {-0.5,0.0,0.0,0.5,0.0,0.0,0.0,0.0}, {-0.5,0.0,0.0,0.0,0.5,0.0,0.0,0.0}},
    {{-0.5,0.5,0.0,0.0,0.0,0.0,0.0,0.0}, {0.0,-0.5,0.5,0.0,0.0,0.0,0.0,0.0}, {0.0,-0.5,0.0,0.0,0.0,0.5,0.0,0.0}},
    {{0.0,0.0,0.5,-0.5,0.0,0.0,0.0,0.0}, {0.0,-0.5,0.5,0.0,0.0,0.0,0.0,0.0}, {0.0,0.0,-0.5,0.0,0.0,0.0,0.5,0.0}},
    {{0.0,0.0,0.5,-0.5,0.0,0.0,0.0,0.0}, {-0.5,0.0,0.0,0.5,0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,-0.5,0.0,0.0,0.0,0.5}},
    {{0.0,0.0,0.0,0.0,-0.5,0.5,0.0,0.0}, {0.0,0.0,0.0,0.0,-0.5,0.0,0.0,0.5}, {-0.5,0.0,0.0,0.0,0.5,0.0,0.0,0.0}},
    {{0.0,0.0,0.0,0.0,-0.5,0.5,0.0,0.0}, {0.0,0.0,0.0,0.0,0.0,-0.5,0.5,0.0}, {0.0,-0.5,0.0,0.0,0.0,0.5,0.0,0.0}},
    {{0.0,0.0,0.0,0.0,0.0,0.0,0.5,-0.5}, {0.0,0.0,0.0,0.0,0.0,-0.5,0.5,0.0}, {0.0,0.0,-0.5,0.0,0.0,0.0,0.5,0.0}},
    {{0.0,0.0,0.0,0.0,0.0,0.0,0.5,-0.5}, {0.0,0.0,0.0,0.0,-0.5,0.0,0.0,0.5}, {0.0,0.0,0.0,-0.5,0.0,0.0,0.0,0.5}},
};
