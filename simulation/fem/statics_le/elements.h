// FEM elements


#include <cmath>
#include <initializer_list>

#undef max
#undef min
template<typename T> inline T max(T x, T y) { return (x > y ? x : y); }
template<typename T> inline T min(T x, T y) { return (x < y ? x : y); }
template<typename T, typename t> inline T clamp(T x, t a, t b) { return (x<a ? a : x>b ? b : x); }
template<typename T, typename f> inline T mix(T x, T y, f a) { return (x * (f(1) - a) + y * a); }  // lerp


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
};

struct ivec4 {
    int x, y, z, w;
    ivec4(int a = 0):x(a), y(a), z(a), w(a) {}
    ivec4(int a, int b, int c, int d):x(a), y(b), z(c), w(d) {}
};


/* Matrix/Vector */

struct vec3 {
    double x, y, z;
    vec3() {}
    explicit vec3(double a):x(a), y(a), z(a) {}
    explicit vec3(double x, double y, double z):x(x), y(y), z(z) {}
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



/* FEM Elements */


#define MAX_SOLID_ELEMENT_N 6

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

public:

    // get the number of vertices
    virtual int getN() const = 0;

    // get vertice indices
    virtual const int* getVi() const = 0;

    // evaluate the gradient of deformation gradient to deflection
    // \partial \nabla u \over \partial u
    // X: a list of global vertex initial positions
    // D: a 3xN matrix, the gradient, before a tensor product with I3
    // t: interpolation parameter, where to evaluate
    virtual void evalD(const vec3* X, double* D, vec3 t) const = 0;

    // evaluate the stiffness matrix
    // X: a list of global vertex initial positions
    // C: a 6x6 matrix transforming strain to stress
    // K: the 3Nx3N stiffness matrix
    virtual void evalK(const vec3* X, const double* C, double* K) const = 0;

};


struct LinearTetrahedralElement: SolidElement {
private:
    // 4 vertices
    int vi[4];

    // precomputed X^{-1}
    mat3 invX;
    // volume of the element
    double dV;

    void calcXV(const vec3 *X) {
        mat3 X3 = transpose(mat3(
            X[vi[1]] - X[vi[0]],
            X[vi[2]] - X[vi[0]],
            X[vi[3]] - X[vi[0]]
        ));
        dV = abs(determinant(X3)) / 6.0;
        invX = inverse(X3);
    }

    // get an element in the matrix D with row and column indices
    static double& getDij(double* D, int i, int j) {
        return D[4 * i + j];
    }

public:

    // vi: the global indices of vertices
    // X: the initial global vertex positions
    LinearTetrahedralElement(int vi[4], const vec3* X) {
        for (int i = 0; i < 4; i++)
            this->vi[i] = vi[i];
        calcXV(X);
    }
    LinearTetrahedralElement(std::initializer_list<int> vi, const vec3 *X) {
        for (int i = 0; i < 4; i++)
            this->vi[i] = vi.begin()[i];
        calcXV(X);
    }
    LinearTetrahedralElement() { }

    // get the number of vertices
    int getN() const { return 4; }

    // get vertice indices
    const int* getVi() const { return &vi[0]; }

    // evaluate the gradient (3x4)
    void evalD(const vec3* X, double* D, vec3 t) const {
        for (int i = 0; i < 3 * 4; i++) D[i] = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                getDij(D, i, 0) -= invX[i][j];
                getDij(D, i, j + 1) += invX[i][j];
            }
        }
    }

    // evaluate the stiffness matrix (12x12)
    void evalK(const vec3* X, const double* C, double* K) const {
        double D[12], SD[72];
        this->evalD(&X[0], D, vec3(0.25));
        this->getLinearizedStrainGradient(4, D, SD);
        this->getElementStiffnessMatrix(4, dV, SD, C, K);
    }

};

