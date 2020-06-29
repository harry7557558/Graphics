// function minimization experiment
// macro _DEBUG_OPTIMIZATION is defined in _debug_optimization_2d.cpp


#include "geometry.h"


// numerical differentiation
// not necessary when gradients are given
template<typename Fun> vec2 nGrad(Fun F, vec2 x, double e = .0001) {
	return (.5 / e) * vec2(F(x + vec2(e, 0)) - F(x - vec2(e, 0)), F(x + vec2(0, e)) - F(x - vec2(0, e)));
}
template<typename Fun> void nGrad2(Fun F, vec2 x, double &Fx, vec2 &grad, vec2 &grad2, double &dxy, double e = .0001) {
	double D[3][3];
	for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
		D[i][j] = F(x + e * vec2(j - 1, i - 1));
	Fx = D[1][1];
	grad = (.5 / e) * vec2(D[1][2] - D[1][0], D[2][1] - D[0][1]);
	grad2 = (1. / (e*e)) * vec2(D[1][2] + D[1][0] - 2.*Fx, D[2][1] + D[0][1] - 2.*Fx);
	dxy = (.25 / (e*e)) * ((D[0][0] + D[2][2]) - (D[0][2] + D[2][0]));
}


// these methods try to find a point with zero-gradient
// can be minimum, local maximum, or saddle point

// this method performs Newton's iteration along gradient direction
template<typename Fun> vec2 Newton_Gradient_2d(Fun F, vec2 x0) {
	const double e = 0.0001;
	vec2 x = x0;
	double F0x = INFINITY;
	for (int i = 0; i < 10000; i++) {
		vec2 g = nGrad(F, x, e);
		// second derivative in gradient direction
		vec2 eg = normalize(g);
		double Fx = F(x);
		double gm = (1. / (e*e)) * (F(x + e * eg) + F(x - e * eg) - 2.*Fx);
		// Newton's iteration step in gradient direction
		vec2 dx = g * (1. / gm);
		if (!(dx.sqr() > 1e-16 && abs(Fx - F0x) > 1e-12)) break;
		x -= dx;
		F0x = Fx;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 128,0,128 }));
		drawDot(x, 5, COLOR({ 128,0,128 }));
#endif
	}
	return x;
}

// this method uses Newton-Raphson method to find a point with zero-gradient
template<typename Fun> vec2 Newton_Iteration_2d(Fun F, vec2 x0) {
	vec2 x = x0;
	double z0 = INFINITY, z;
	for (int i = 0; i < 10000; i++) {
		vec2 g, g2; double gxy;
		nGrad2(F, x, z, g, g2, gxy, 0.0001);
		double m = 1. / (g2.x*g2.y - gxy * gxy);
		vec2 dx = m * vec2(g.x*g2.y - g.y*gxy, g2.x*g.y - g.x*gxy);
		if (!(dx.sqr() > 1e-16 && abs(z - z0) > 1e-12)) break;
		x -= dx;
		z0 = z;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 128,128,0 }));
		drawDot(x, 5, COLOR({ 128,128,0 }));
#endif
	}
	return x;
}

