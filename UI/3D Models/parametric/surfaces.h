// parametric surface equations for generating objects in run-time
// most surfaces are designed from aesthetic view


#ifndef __INC_GEOMETRY_H
#include "numerical/geometry.h"
#endif

// Parametric surface class
#include <functional>
#include <vector>
template<typename Fun>
class ParametricSurface {
public:
	double u0, u1, v0, v1;  // parameter intervals
	int uD, vD;  // recommended discretization splits
	const Fun P;  // equation, vec3 P(u,v)
	uint32_t id;  // optional
	ParametricSurface(Fun P,
		double u0 = NAN, double u1 = NAN, double v0 = NAN, double v1 = NAN, int uD = 0, int vD = 0,
		const char* name = nullptr)
		:u0(u0), u1(u1), v0(v0), v1(v1), uD(uD), vD(vD), P(P) {
		id = hash(name);
	}
	static uint32_t hash(const char* s) {
		uint32_t h = 0;
		while (*s) h = 1664525u * h + 1013904223u * *(s++);
		return h;
	}

	int param2trigs(std::vector<triangle> &p,
		vec3 translate = vec3(0.), double scale = 1.) const {
		auto F = [&](double u, double v) {
			return scale * P(u, v) + translate;
		};
		double du = (u1 - u0) / uD, dv = (v1 - v0) / vD;
		for (int ui = 0; ui < uD; ui++) {
			for (int vi = 0; vi < vD; vi++) {
				double u = u0 + ui * du, v = v0 + vi * dv;
				vec3 p00 = F(u, v);
				vec3 p01 = F(u, v + dv);
				vec3 p10 = F(u + du, v);
				vec3 p11 = F(u + du, v + dv);
				p.push_back(triangle{ p10, p00, p01 });
				p.push_back(triangle{ p01, p11, p10 });
			}
		}
		return 2 * uD * vD;
	}
};
typedef ParametricSurface<vec3(*)(double, double)> ParametricSurfaceP;
typedef ParametricSurface<std::function<vec3(double, double)>> ParametricSurfaceL;




// function templates

namespace ParametricSurfaceTemplates {

	vec3 Archimedean_snail(double u, double v,
		double vert, double offset, double layer_n) {

		/* 0 < u < 2π, 0 < v < 1
		   For a right-handed snail, all of the following are positive:
			- vert: height/sharpness, vertice=(0,0,vert)
			- offset: offset from z-axis, 1.0 for a "standard" snail
			- layer_n: number of layers of the snail
		*/

		return (1. - v)*vec3(
			(cos(u) + offset)*cossin(2.*PI*layer_n*v),
			sin(u)) + vec3(0, 0, vert*v);
	};

	vec3 Logarithmic_snail(double u, double v,
		double exp_k, double vert, double offset, double layer_n) {

		/* 0 < u < 2π, v0 < v < v1
			- exp_k: r=exp(k/2π*θ), positive
			- vert: z_vertice = exp(exp_k)*vert
			- offset: offset from z-axis, 1.0 for a "standard" snail
			- layer_n: number of layers of the snail when -1<v<1
			- v0, v1: custom parameters, v0 can be -INFINITY
		*/

		return exp(exp_k*v) * vec3(
			cossin(PI*layer_n*v)*(offset + cos(u)),
			vert*(exp(exp_k - exp_k * v) - 2.) + sin(u));

		// right-handed version
		return exp(exp_k*v) * vec3(
			sincos(PI*layer_n*v)*(offset + cos(u)),
			vert*(exp(exp_k - exp_k * v) - 2.) - sin(u));
	};


}



const std::vector<ParametricSurfaceL> ParamSurfaces({

	/*[0]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Archimedean_snail(u, v, 1.2, 1.0, 3.0);
	}, 0., 2.*PI, 0., 1., 40, 120, "land snail"),

	/*[1]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Archimedean_snail(u, v, 2.2, 1.0, 4.5);
	}, 0., 2.*PI, 0., 1., 40, 120, "river snail"),

	/*[2]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Archimedean_snail(u, v, 1.1, 0.5, 2.0);
	}, 0., 2.*PI, 0., 1., 40, 120, "field snail"),

	/*[3]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Archimedean_snail(u, v, 1.8, 0.2, 2.4);
	}, 0., 2.*PI, 0., 1., 40, 120, "pond snail"),

	/*[4]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Archimedean_snail(u, v, 3.5, 0.2, 8.0);
	}, 0., 2.*PI, 0., 1., 40, 120, "cone snail"),

	/*[5]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Logarithmic_snail(u, v, 0.45, 1.7, 0.35, 2.1);
	}, 0., 2.*PI, -7., 1., 40, 120, "pot snail"),
		
	/*[6]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Logarithmic_snail(u, v, 0.65, 0.55, 0.86, 1.0);
	}, 0., 2.*PI, -6., 1., 40, 160, "moon snail"),
		
	/*[7]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Logarithmic_snail(u, v, 0.9, 0.0, 2.0, 1.0);
	}, 0., 2.*PI, -6., -1., 40, 160, "snail (out)"),

	/*[8]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Logarithmic_snail(u, v, 1.8, 1.2, 2.7, 1.0);
	}, 0., 2.*PI, -3., 1., 40, 160, "snail (outer)"),

	/*[9]*/ ParametricSurfaceL([](double u, double v) {
		return ParametricSurfaceTemplates::Logarithmic_snail(u, v, 0.7, 0.0, 1.05, 1.0);
	}, 0., 2.*PI, -5., 1., 40, 160, "snail (inner)"),


	/*[10]*/ ParametricSurfaceL([](double u, double v) {
		return vec3(
			cossin(3.*PI*v)*(2.0 + cos(u)) + 0.05*cossin(60.*PI*v),
			sin(u) + 0.05*sin(10.*u))*exp(v);
	}, 0., 2.*PI, -4., 1., 80, 1000, "textured snail 1"),
		
	/*[11]*/ ParametricSurfaceL([](double u, double v) {
		return vec3(
			sincos(3.*PI*v)*(1. + cos(u)) + 0.1*cossin(-10.*u),
			4. - (v + 4.) - 0.9*sin(u))*exp(0.8*v);
	}, 0., 2.*PI, -4., 1., 100, 800, "textured snail 2"),
		
	/*[12]*/ ParametricSurfaceL([](double u, double v) {
		vec3 p = vec3(
			cossin(PI*3.*v)*(1. + cos(u)) + 0.1*cossin(40.*PI*v),
			(exp(1. - v) - 2.) + sin(u));
		p += 0.06*vec3(sin(10.*p.x)*sin(10.*p.y)*cos(10.*p.z));
		return p * exp(v);
	}, 0., 2.*PI, -3., 1., 100, 1600, "textured snail 3"),

	/*[13]*/ ParametricSurfaceL([](double u, double v) {
		return vec3(
			sincos(3.*PI*v)*(0.9 + cos(u)) + 0.05*cossin(40.*PI*v)*cos(10.*u)*(1. + cos(u)),
			(exp(1.5 - 0.9*v) - 3.) - 1.1*sin(u))*exp(0.9*v);
	}, 0., 2.*PI, -4., 1., 100, 800, "textured snail 4"),

	/*[14]*/ ParametricSurfaceL([](double u, double v) {
		return vec3(sincos(3.*PI*v)*(1. + cos(u)), -1.1*sin(u))
			*exp(0.8*v)*(.5*cos(20.*PI*v) + .9)*(.05*cos(10.*u) + 1.)
			+ vec3(0, 0, 5. - 4.*exp(v));
	}, 0., 2.*PI, -4., 1., 100, 1000, "textured snail #1"),

	/*[15]*/ ParametricSurfaceL([](double u, double v) {
		return vec3(
			(1 - .05*exp(sin(u)))*(cossin(3.*PI*v)*(2.0 + cos(u)) + 0.05*cossin(60.*PI*v)),
			-exp(1. - .5*v) + sin(u) + 0.1*sin(10.*u)*sin(20.*PI*v)*exp(v))*exp(v)
			+ vec3(0, 0, 4.);
	}, 0., 2.*PI, -4., 1., 100, 1000, "textured snail #2"),
		
	/*[16]*/ ParametricSurfaceL([](double u, double v) {
		return exp(v) * vec3(
			cossin(PI*3.*v)*(1. + cos(u)),
			(exp(1. - v) - 2.) + exp(v)*sin(u));
	}, 0., 2.*PI, -3., 1., 40, 200, "melon snail"),

	/*[17]*/ ParametricSurfaceL([](double u, double v) {
		vec3 p = exp(v) * vec3(
			cossin(PI*3.*v)*(1. + cos(u)),
			.55*(exp(1. - v) - 2.) + sin(u));
		return p + vec3(0, 0, 3. - exp(-.55*p.z));
	}, 0., 2.*PI, -3., 1., 40, 200, "bailer snail"),
});





// functions that may be useful for normalizing test shapes

// calculate the center of mass of an object
// assume the object is a surface with uniform surface density
vec3 calcCOM_shell(const triangle* T, int N) {
	double A = 0; vec3 C(0.);
	for (int i = 0; i < N; i++) {
		double dA = T[i].area();
		vec3 dC = (1. / 3.)*(T[i].A + T[i].B + T[i].C);
		A += dA, C += dA * dC;
	}
	return C * (1. / A);
}

// calculate the axis-aligned bounding box, return the center
vec3 calcAABB(const triangle* T, int N, vec3* rad = 0) {
	const vec3* P = (vec3*)T; N *= 3;
	vec3 Min(INFINITY), Max(-INFINITY);
	for (int i = 0; i < N; i++) {
		Max = pMax(Max, P[i]);
		Min = pMin(Min, P[i]);
	}
	if (rad) *rad = .5*(Max - Min);
	return .5*(Max + Min);
}

// translate the object so its center is the origin
void translateToCOM_shell(triangle* T, int N) {
	vec3 D = -calcCOM_shell(T, N);
	for (int i = 0; i < N; i++) T[i].translate(D);
}
void translateToAABBCenter(triangle* T, int N) {
	vec3 D = -calcAABB(T, N);
	for (int i = 0; i < N; i++) T[i].translate(D);
}

// calculate the maximum "radius" of the object from the origin
double calcRadius(const triangle* T, int N) {
	const vec3* P = (vec3*)T; N *= 3;
	double maxR = 0.;
	for (int i = 0; i < N; i++) {
		double r = P[i].sqr();
		if (r > maxR) maxR = r;
	}
	return sqrt(maxR);
}
