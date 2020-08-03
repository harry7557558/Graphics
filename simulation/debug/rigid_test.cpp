// test the correctness of some analytically integrated inertia tensors

#pragma GCC optimize "Ofast"

#include <stdio.h>
#include <chrono>

// vector/matrix, random
#include "D:\Coding\Github\Graphics\fitting\numerical\random.h"

void printfloatv(double x, const char* end = "\n") { if (x*x < 1e-8) printf("0"); else printf("%.4lg", x); printf(end); }
void printvec3(vec3 p, const char* end = "\n") { printf("(%lf,%lf,%lf)%s", p.x, p.y, p.z, end); }
void printvec3v(vec3 p, const char* end = "\n") { printf("("); printfloatv(p.x, ","); printfloatv(p.y, ","); printfloatv(p.z, ")"); printf(end); }
void printmat3(const mat3 &m, const char* end = "\n") { printf("\\begin{bmatrix}"); for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) printf("%lf%s", m.v[i][j], j == 2 ? "\\\\" : "&"); printf("\\end{bmatrix}%s", end); }
void printmat3v(const mat3 &m, const char* end = "\n") { for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) printfloatv(m.v[i][j], j == 2 ? "\n" : "\t"); printf(end); }



// numerical integral in one dimension
// s: type specifier for function template, should be initialized to zero
template<typename Fun, typename T> T Integral(Fun f, double a, double b, int n, T s) {
	n *= 2;
	double u = (b - a) / n;
	for (int i = 1; i < n; i += 2) s += f(a + u * i);
	s *= 2;
	for (int i = 2; i < n; i += 2) s += f(a + u * i);
	s = 2 * s + f(a) + f(b);
	return s * (u / 3);
}
void testIntegral(int N = 100){
    // check single variable analytical integrals numerically
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		double r = randf(0,5);
		double a = randf(-r,r);
		double x0 = a;
		double x1 = r;
		auto F = [&](double x) ->double {
			//return 2*(x*x*sqrt(r*r-x*x)+1./3.*pow(r*r-x*x,1.5));
			return 2*(x*x*sqrt(r*r-x*x));
		};
		double I0 = 1./4.*r*r*r*r*(acos(a/r)+1./4.*sin(4*asin(a/r)));
		double I = Integral(F, x0, x1, 1000, 0.0);
		printf("%4d\t", i + 1); printf("%lf\n", I/I0);
	}
}




// test inertia calculation of basic primitives

mat3 segmentInertia(vec3 a, vec3 b) {
	return (1. / 3) * (mat3(dot(a, a) + dot(b, b) + dot(a, b)) \
		- ((tensor(a, a) + tensor(b, b)) + 0.5*(tensor(a, b) + tensor(b, a))));
}
mat3 segmentInertiaN(vec3 a, vec3 b) {
	vec3 d = b - a;
	mat3 I(0.0); double L = 0.0;
	const int N = 1000; const double dt = 1. / N;
	for (int i = 0; i < N; i++) {
		double t = i * dt;
		vec3 u = a + d * t, v = u + d * dt;
		double dL = length(v - u);
		vec3 r = (u + v)*0.5;
		I += (mat3(dot(r, r)) - tensor(r, r)) * dL;
		L += dL;
	}
	return I * (1. / L);
}
void testSegmentInertia(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		vec3 a = rand3(5.0);
		vec3 b = rand3(5.0);
		mat3 I = segmentInertia(a, b);
		mat3 J = segmentInertiaN(a, b);
		mat3 d = I - J;
		printf("%lf\t%lf\t%lf\n", cbrt(abs(determinant(d))), abs(trace(d)), sqrt(sumsqr(d)));
	}
}

mat3 triangleInertia(vec3 a, vec3 b) {
	return 0.5*segmentInertia(a, b);
}
mat3 triangleInertiaN(vec3 a, vec3 b) {
	mat3 I(0.0); vec3 C(0.0); double A = 0.0;
	const int N = 1000; const double du = 1. / N, dv = du;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double u0 = i * du, u1 = u0 + du;
			double v0 = j * dv, v1 = v0 + dv;
			vec3 p00 = u0 * ((1 - v0)*a + v0 * b);
			vec3 p01 = u0 * ((1 - v1)*a + v1 * b);
			vec3 p10 = u1 * ((1 - v0)*a + v0 * b);
			vec3 p11 = u1 * ((1 - v1)*a + v1 * b);
			double dA = 0.5 * (length(cross(p01 - p00, p10 - p00)) + length(cross(p10 - p11, p01 - p11))); A += dA;
			vec3 p = 0.25*(p00 + p01 + p10 + p11); C += p * dA;
			I += (mat3(dot(p, p)) - tensor(p, p)) * dA;
		}
	}
	C *= (1. / A);
	return I * (1. / A);
}
void testTriangleInertia(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		vec3 a = rand3(5.0);
		vec3 b = rand3(5.0);
		mat3 I = triangleInertia(a, b);
		mat3 J = triangleInertiaN(a, b);
		mat3 d = I - J;
		printf("%lf\t%lf\t%lf\n", cbrt(abs(determinant(d))), abs(trace(d)), sqrt(sumsqr(d)));
	}
}

mat3 TriangleInertia(vec3 a, vec3 b, vec3 c) {
	return (1. / 6.)*(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
		(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
}
mat3 TriangleInertiaN(vec3 a, vec3 b, vec3 c) {
	mat3 I(0.0); vec3 C(0.0); double A = 0.0;
	const int N = 1000; const double du = 1. / N, dv = du;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			double u0 = i * du, u1 = u0 + du;
			double v0 = j * dv, v1 = v0 + dv;
			vec3 p00 = (1 - u0)*((1 - v0)*a + v0 * b) + u0 * ((1 - v0)*a + v0 * c);
			vec3 p01 = (1 - u0)*((1 - v1)*a + v1 * b) + u0 * ((1 - v1)*a + v1 * c);
			vec3 p10 = (1 - u1)*((1 - v0)*a + v0 * b) + u1 * ((1 - v0)*a + v0 * c);
			vec3 p11 = (1 - u1)*((1 - v1)*a + v1 * b) + u1 * ((1 - v1)*a + v1 * c);
			double dA = 0.5 * (length(cross(p01 - p00, p10 - p00)) + length(cross(p10 - p11, p01 - p11))); A += dA;
			vec3 p = 0.25*(p00 + p01 + p10 + p11); C += p * dA;
			I += (mat3(dot(p, p)) - tensor(p, p)) * dA;
		}
	}
	C *= (1. / A);
	return I * (1. / A);
}
void testTriangleAInertia(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		vec3 a = rand3(5.0);
		vec3 b = rand3(5.0);
		vec3 c = rand3(5.0);
		mat3 I = TriangleInertia(a, b, c);
		mat3 J = TriangleInertiaN(a, b, c);
		mat3 d = I - J;
		printf("%lf\t%lf\t%lf\n", cbrt(abs(determinant(d))), abs(trace(d)), sqrt(sumsqr(d)));
	}
}

mat3 tetrahedronInertia(vec3 a, vec3 b, vec3 c) {
	//return 0.6*TriangleInertia(a, b, c);
	return 0.1*(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
		(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
}
mat3 tetrahedronInertiaN(vec3 a, vec3 b, vec3 c) {
	mat3 I(0.0); vec3 C(0.0); double V = 0.0;
	const int N = 100; const double du = 1. / N, dv = du, dw = du;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				double u0 = i * du, u1 = u0 + du;
				double v0 = j * dv, v1 = v0 + dv;
				double w0 = k * dw, w1 = w0 + dw;
				vec3 p000 = u0 * (1 - v0)*w0 * a + u0 * v0*w0*b + (1 - u0)*w0*c;
				vec3 p001 = u0 * (1 - v0)*w1 * a + u0 * v0*w1*b + (1 - u0)*w1*c;
				vec3 p010 = u0 * (1 - v1)*w0 * a + u0 * v1*w0*b + (1 - u0)*w0*c;
				vec3 p011 = u0 * (1 - v1)*w1 * a + u0 * v1*w1*b + (1 - u0)*w1*c;
				vec3 p100 = u1 * (1 - v0)*w0 * a + u1 * v0*w0*b + (1 - u1)*w0*c;
				vec3 p101 = u1 * (1 - v0)*w1 * a + u1 * v0*w1*b + (1 - u1)*w1*c;
				vec3 p110 = u1 * (1 - v1)*w0 * a + u1 * v1*w0*b + (1 - u1)*w0*c;
				vec3 p111 = u1 * (1 - v1)*w1 * a + u1 * v1*w1*b + (1 - u1)*w1*c;
				// divide the cube into five tetrahedrons
				double V0 = det(p001 - p010, p111 - p010, p100 - p010); vec3 p0 = p010 + p001 + p111 + p100;
				double V1 = det(p001 - p000, p010 - p000, p100 - p000); vec3 p1 = p000 + p001 + p010 + p100;
				double V2 = det(p010 - p011, p001 - p011, p111 - p011); vec3 p2 = p011 + p010 + p001 + p111;
				double V3 = det(p100 - p110, p010 - p110, p111 - p110); vec3 p3 = p110 + p100 + p010 + p111;
				double V4 = det(p001 - p101, p100 - p101, p111 - p101); vec3 p4 = p101 + p001 + p100 + p111;
				double dV = (V0 + V1 + V2 + V3 + V4) / 6; V += dV;
				vec3 p = (.25 / (V0 + V1 + V2 + V3 + V4))*(p0 * V0 + p1 * V1 + p2 * V2 + p3 * V3 + p4 * V4); C += p * dV;
				I += (mat3(dot(p, p)) - tensor(p, p)) * dV;
			}
		}
	}
	C *= (1. / V);
	return I * (1. / V);
}
void testTetrahedronInertia(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		vec3 a = rand3(5.0);
		vec3 b = rand3(5.0);
		vec3 c = rand3(5.0);
		mat3 I = tetrahedronInertia(a, b, c);
		mat3 J = tetrahedronInertiaN(a, b, c);
		//double s = 0; for (int i = 0; i < 9; i++) s += (&J.v[0][0])[i] / (&I.v[0][0])[i]; printf("%lf\n", s / 9);
		mat3 d = I - J;
		printf("%d\t%lf\t%lf\t%lf\n", i + 1, cbrt(abs(determinant(d))), abs(trace(d)), sqrt(sumsqr(d)));
	}
}




// test analytically integrated inertias

struct object {
	double m = 0;  // mass
	vec3 c = vec3(0);  // center of mass
	mat3 I = mat3(0);  // inertia tensor
};
void printDifference(object a, object b) {
    // b divide by a element-wise
	printf("%lf    ", b.m / a.m);
	for (int i = 0; i < 3; i++) {
		double ai = ((double*)&a.c)[i], bi = ((double*)&b.c)[i];
		if (ai != 0.0 && bi != 0.0) printf("%lf ", bi / ai);
		else printf("' ");
		//printf("%lf %lf ", ai, bi);
	}
	if (a.c.sqr()==0.0) printf("%lf", length(b.c-a.c));
	printf("   ");
	for (int i = 0; i < 3; i++) for (int j = i; j < 3; j++) {  // assume symmetric matrix
		double ai = a.I.v[i][j], bi = b.I.v[i][j];
		if (ai != 0.0 && bi != 0.0) printf("%lf ", bi / ai);
		else printf("' ");
	}
	printf("\n");
}

typedef _geometry_segment<vec3> segment;
typedef _geometry_triangle<vec3> triangle;

// L: line integral
// A: surface integral
// S: volume integral using Gauss divergence theorem
// V: volume integral

object calcSegment(segment s) {
	object r;
	r.m = length(s.q - s.p);
	r.c = 0.5*(s.p + s.q);
	r.I = r.m*segmentInertia(s.p, s.q);
	return r;
}
object calcObject_Ls(const segment *s, int N) {
	object R;
	for (int i = 0; i < N; i++) {
		object r = calcSegment(s[i]);
		R.m += r.m;
		R.c += r.m*r.c;
		R.I += r.I;
	}
	R.c /= R.m;
	return R;
}

object calcTriangle(triangle s) {
	vec3 a = s.a, b = s.b, c = s.c;
	object r;
	r.m = 0.5*length(cross(b - a, c - a));
	r.c = (a + b + c) / 3.;
	r.I = r.m*TriangleInertia(a, b, c);
	return r;
}
object calcObject_As(const triangle *s, int N) {
	object R;
	for (int i = 0; i < N; i++) {
		object r = calcTriangle(s[i]);
		R.m += r.m;
		R.c += r.m*r.c;
		R.I += r.I;
	}
	R.c /= R.m;
	return R;
}
void testObject_As(int N = 100) {
	triangle T[1024];
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		double r = randf(0, 5);
		int n = randi(3, 10);
		object O0;
		O0.m = .5*r*r*sin(2*PI/n)*n;
		O0.c = vec3(0,0,0);
		O0.I = O0.m * (r*r/12.*(2+cos(2*PI/n))) * mat3(vec3(1,1,2));
		for (int i=0;i<n;i++) T[i] = triangle{vec3(0),r*vec3(cos(i*2*PI/n),sin(i*2*PI/n),0),r*vec3(cos((i+1)*2*PI/n),sin((i+1)*2*PI/n),0)};
		object O = calcObject_As(T, n);
		printf("%4d\t", i + 1); printDifference(O0, O);
	}
}

object calcTetrahedron(triangle s) {
	vec3 a = s.a, b = s.b, c = s.c;
	object r;
	r.m = det(a, b, c) / 6.;  // signed
	r.c = (a + b + c) / 4.;
	r.I = r.m*tetrahedronInertia(a, b, c);
	return r;
}
object calcObject_Ss(const triangle *s, int N) {  // be careful about the face normal of the triangles
	object R;
	for (int i = 0; i < N; i++) {
		object r = calcTetrahedron(s[i]);
		R.m += r.m;
		R.c += r.m*r.c;
		R.I += r.I;
	}
	R.c /= R.m;
	return R;
}

template<typename Fun> object calcObject_L(Fun F, double t0, double t1, int N = 1024) {
	double t, dt = (t1 - t0) / N;
	segment *S = new segment[N];
	for (int i = 0; i < N; i++) {
		t = t0 + i * dt;
		vec3 a = F(t), b = F(t + dt);
		S[i] = segment{ a, b };
	}
	object r = calcObject_Ls(S, N);
	delete S;
	return r;
}
void testObject_L(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		double r = randf(0, 5);
		double a = randf(0,PI);
		double t0 = -a;
		double t1 = a;
		auto F = [&](double t) {
			return vec3(r*cos(t),r*sin(t),0);
		};
		object O0;
		O0.m = 2*a*r;
		O0.c = vec3(r*sin(a)/a, 0, 0);
		O0.I = O0.m*(.5*r*r) * mat3(vec3(1-sin(2*a)/(2.*a),1+sin(2*a)/(2.*a),2));
		object O = calcObject_L(F, t0, t1);
		printf("%4d\t", i + 1); printDifference(O0, O);
	}
}

template<typename Fun> object calcObject_A(Fun F, double u0, double u1, double v0, double v1, int uN = 1024, int vN = 1024) {
	double u, v, du = (u1 - u0) / uN, dv = (v1 - v0) / vN;
	triangle *S = new triangle[2 * uN*vN]; int sD = 0;
	for (int i = 0; i < uN; i++) {
		u = u0 + i * du;
		for (int j = 0; j < vN; j++) {
			v = v0 + j * dv;
			vec3 p00 = F(u, v);
			vec3 p01 = F(u, v + dv);
			vec3 p10 = F(u + du, v);
			vec3 p11 = F(u + du, v + dv);
			S[sD++] = triangle{ p00, p01, p10 };
			S[sD++] = triangle{ p00, p11, p01 };
		}
	}
	object r = calcObject_As(S, sD);
	delete S;
	return r;
}
void testObject_A(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		vec3 a = rand3(5);
		vec3 b = rand3(5);
		vec3 p = rand3(5);
		double u0 = 0;
		double u1 = 1;
		double v0 = 0;
		double v1 = 1;
		auto F = [&](double u, double v) {
			return p+u*a+v*b;
		};
		object O0;
		O0.m = length(cross(a,b));
		O0.c = p+.5*a+.5*b;
		O0.I = O0.m*(mat3(1./3.*(dot(a,a)+dot(b,b)+3*dot(p,p))+dot(a,p)+dot(b,p)+.5*dot(a,b))-(1./3.*(tensor(a,a)+tensor(b,b)+3*tensor(p,p))+.5*(tensor(a,p)+tensor(p,a)+tensor(b,p)+tensor(p,b)+.5*(tensor(a,b)+tensor(b,a)))));
		object O = calcObject_A(F, u0, u1, v0, v1);
		printf("%4d\t", i + 1); printDifference(O0, O);
	}
}

template<typename Fun> object calcObject_S(Fun F, double u0, double u1, double v0, double v1, int uN = 1024, int vN = 1024) {
	// must be closed surface with normal facing outside
	double u, v, du = (u1 - u0) / uN, dv = (v1 - v0) / vN;
	triangle *S = new triangle[2 * uN*vN]; int sD = 0;
	for (int i = 0; i < uN; i++) {
		u = u0 + i * du;
		for (int j = 0; j < vN; j++) {
			v = v0 + j * dv;
			vec3 p00 = F(u, v);
			vec3 p01 = F(u, v + dv);
			vec3 p10 = F(u + du, v);
			vec3 p11 = F(u + du, v + dv);
			S[sD++] = triangle{ p00, p10, p11 };
			S[sD++] = triangle{ p00, p11, p01 };
		}
	}
	object r = calcObject_Ss(S, sD);
	delete S;
	return r;
}
void testObject_S(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		double a = randf(0, 5);
		double b = randf(0, 5);
		double c = randf(0, 5);
		double u0 = 0;
		double u1 = 2 * PI;
		double v0 = PI;
		double v1 = 0;
		auto F = [&](double u, double v) {
			return vec3(a*sin(v)*cos(u), b*sin(v)*sin(u), c*cos(v));
		};
		object O0;
		O0.m = 4. / 3.*PI*a*b*c;
		O0.c = vec3(0, 0, 0);
		O0.I = O0.m*(.2) * mat3(vec3(b*b + c * c, a*a + c * c, a*a + b * b));
		object O = calcObject_S(F, u0, u1, v0, v1);
		printf("%4d\t", i + 1); printDifference(O0, O);
	}
}

template<typename Fun> object calcObject_V(Fun F, double u0, double u1, double v0, double v1, double w0, double w1, int uN = 128, int vN = 128, int wN = 128) {
	double u, v, w, u_, v_, w_, du = (u1 - u0) / uN, dv = (v1 - v0) / vN, dw = (w1 - w0) / wN;
	mat3 I(0.0); vec3 C(0.0); double V = 0.0;
	for (int i = 0; i < uN; i++) {
		for (int j = 0; j < vN; j++) {
			for (int k = 0; k < wN; k++) {
				u = u0 + i * du, u_ = u + du;
				v = v0 + j * dv, v_ = v + dv;
				w = w0 + k * dw, w_ = w + dw;
				vec3 p000 = F(u, v, w);
				vec3 p001 = F(u, v, w_);
				vec3 p010 = F(u, v_, w);
				vec3 p011 = F(u, v_, w_);
				vec3 p100 = F(u_, v, w);
				vec3 p101 = F(u_, v, w_);
				vec3 p110 = F(u_, v_, w);
				vec3 p111 = F(u_, v_, w_);
				double V0 = det(p001 - p010, p100 - p010, p111 - p010); vec3 p0 = (p010 + p001 + p111 + p100);
				double V1 = det(p001 - p000, p100 - p000, p010 - p000); vec3 p1 = (p000 + p001 + p010 + p100);
				double V2 = det(p010 - p011, p111 - p011, p001 - p011); vec3 p2 = (p011 + p010 + p001 + p111);
				double V3 = det(p100 - p110, p111 - p110, p010 - p110); vec3 p3 = (p110 + p100 + p010 + p111);
				double V4 = det(p001 - p101, p111 - p101, p100 - p101); vec3 p4 = (p101 + p001 + p100 + p111);
				double dV = (V0 + V1 + V2 + V3 + V4) / 6.; V += dV;
				vec3 p = (1. / (24.*dV))*(p0 * V0 + p1 * V1 + p2 * V2 + p3 * V3 + p4 * V4); C += p * dV;
				I += (mat3(dot(p, p)) - tensor(p, p)) * dV;
			}
		}
	}
	if (V <= 0) fprintf(stderr, "[%u] WARNING: negative Jacobian\n", __LINE__);
	return object{ V, C / V, I };
}
void testObject_V(int N = 100) {
	for (int i = 0; i < N; i++) {
		_SRAND(i);
		double R = randf(0,5);
		double r1 = randf(0,R);
		double r0 = randf(0,r1);
		double r_ = r1*r1+r0*r0;
		double u0 = 0;
		double u1 = 2*PI;
		double v0 = 0;
		double v1 = 2*PI;
		double w0 = r0;
		double w1 = r1;
		auto F = [&](double u, double v, double w) {
			return vec3(cos(u)*(R+w*cos(v)),sin(u)*(R+w*cos(v)),w*sin(v));
		};
		object O0;
		O0.m = 2*PI*PI*R*(r1*r1-r0*r0);
		O0.c = vec3(0,0,0);
		O0.I = O0.m*(1)*mat3(vec3(1./2.*R*R+5./8.*r_,1./2.*R*R+5./8.*r_,R*R+3./4.*r_));
		object O = calcObject_V(F, u0, u1, v0, v1, w0, w1);
		printf("%4d\t", i + 1); printDifference(O0, O);
	}
}


int main() {
	auto t0 = std::chrono::high_resolution_clock::now();
	testObject_As(100);
	printf("%lfs elapsed\n", std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
	//system("pause");
	return 0;
}

