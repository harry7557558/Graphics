/* INCOMPLETE */
// function minimization experiment
// macro _DEBUG_OPTIMIZATION is defined in _debug_optimization_2d.cpp



// One-dimensional optimization
// These functions reveive parameters as reference instead of returning a number
// because they are often used in multivariable optimizations and the functions may be expensive to evaluate.

// before calling this function: evaluate y0=F(x0) and y1=F(x1) and make sure y1<y0
template<typename Fun> void bracketMinimum_1d(Fun F, double &x0, double &x1, double &y0, double &y1) { /* Not Implemented */ }
// minimize a 1d function where the minima is bracketed x0 and x1
template<typename Fun> void GoldenSectionSearch_1d(Fun F, double &x0, double &x1, double &y0, double &y1, double eps = 1e-12) {
	const double g1 = 0.6180339887498949, g0 = 1.0 - g1;
	double t0 = g1 * x0 + g0 * x1;
	double t1 = g0 * x0 + g1 * x1;
	y0 = F(t0), y1 = F(t1);
	for (int i = 0; i < 64; i++) {
		if (y0 < y1) {
			x1 = t1, y1 = y0;
			t1 = t0, t0 = g1 * x0 + g0 * x1;
			y0 = F(t0);
		}
		else {
			x0 = t0, y0 = y1;
			t0 = t1, t1 = g0 * x0 + g1 * x1;
			y1 = F(t1);
		}
		if (x1 - x0 < eps) break;
	}
}




#include "geometry.h"


// numerical differentiation in 2d
// not necessary when analytical gradient is given
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




// Non-standard methods


// this method performs Newton's iteration in gradient direction
template<typename Fun> vec2 Newton_Gradient_2d(Fun F, vec2 x0) {
	const double e = 0.0001;
	vec2 x = x0;
	double F0x = INFINITY;
	for (int i = 0; i < 10000; i++) {
		double Fx = F(x);
		double xp = F(x + vec2(e, 0)), xm = F(x - vec2(e, 0));
		double yp = F(x + vec2(0, e)), ym = F(x - vec2(0, e));
		vec2 g = (.5 / e)*vec2(xp - xm, yp - ym), eg = normalize(g);  // gradient
		double gm = (1. / (e*e)) * (F(x + e * eg) + F(x - e * eg) - 2.*Fx);  // second derivative in gradient direction
		vec2 dx = g * abs(1. / gm);  // simply taking an abs prevents it from reaching a maxima
		if (!(dx.sqr() > 1e-12 && abs(Fx - F0x) > 1e-8)) {
			// break the balance when it reaches a saddle point (doesn't always work)
			// this makes it generally faster but more like to stuck on valleys
			if (Fx > xp) x += vec2(e, 0);
			else if (Fx > xm) x -= vec2(e, 0);
			else if (Fx > yp) x += vec2(0, e);
			else if (Fx > ym) x -= vec2(0, e);
			// termination - small step, small difference, or nan
			else if (!(dx.sqr() > 1e-16 && abs(Fx - F0x) > 1e-12)) {
				if (0.0*dot(dx, dx) == 0.0) return x - dx;
				else return x;
			}
		}
		else x -= dx;
		F0x = Fx;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 128,0,128 }));
		drawDot(x, 5, COLOR({ 128,0,128 }));
#endif
	}
	return x;
}

// this method uses Newton-Raphson method to find a point with zero-gradient
// it might be a local maxima, a minima, or a saddle point
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
// this version doesn't get maxima and saddle but is slower and has a higher risk of failture
// ideas of improvement include checking signs of eigenvalues and detecting infinite loops
template<typename Fun> vec2 Newton_Iteration_2d_(Fun F, vec2 x0) {
	const double e = 0.0001;
	vec2 x = x0;
	double z0 = INFINITY, z;
	for (int i = 0; i < 10000; i++) {
		// numerical differentiation
		double D[3][3];
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			D[i][j] = F(x + e * vec2(j - 1, i - 1));
		z = D[1][1];
		vec2 g = (.5 / e) * vec2(D[1][2] - D[1][0], D[2][1] - D[0][1]);
		vec2 g2 = (1. / (e*e)) * vec2(D[1][2] + D[1][0] - 2.*z, D[2][1] + D[0][1] - 2.*z);
		double gxy = (.25 / (e*e)) * ((D[0][0] + D[2][2]) - (D[0][2] + D[2][0]));
		// solve the linear system
		double m = 1. / (g2.x*g2.y - gxy * gxy);
		vec2 dx = m * vec2(g.x*g2.y - g.y*gxy, g2.x*g.y - g.x*gxy);
		if (2.*dot(dx, g) < dot(dx, vec2(g2.x*dx.x + gxy * dx.y, gxy*dx.x + g2.y*dx.y))) dx = -dx;  // prevent reaching maxima
		if (!(dx.sqr() > 1e-12 && abs(z - z0) > 1e-8)) {
			// prevent reaching saddle
			// this may cause it to enter an infinite loop
			int mi = -1, mj = -1;
			double mz = z;
			for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
				if (D[i][j] < mz) mz = D[i][j], mi = i, mj = j;
			if (mi != -1) x += e * vec2(mj - 1, mi - 1);
			else if (!(dx.sqr() > 1e-16 && abs(z - z0) > 1e-12)) {  // termination
				if (0.0*dx.x == 0.0*dx.y) return x - dx;
				else return x;
			}
		}
		if (0.0*dx.x == 0.0*dx.y) x -= dx;
		else break;
		z0 = z;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 160,40,40 }));
		drawDot(x, 5, COLOR({ 160,40,40 }));
#endif
	}
	return x;
}





// debugging method
template<typename Fun> vec2 Untitled_Method(Fun F, vec2 x0) {
	const double e = 0.0001;
	vec2 x = x0;
	double F0x = INFINITY;
	for (int i = 0; i < 10000; i++) {
#if 0
		double Fx = F(x);
		double xp = F(x + vec2(e, 0)), xm = F(x - vec2(e, 0));
		double yp = F(x + vec2(0, e)), ym = F(x - vec2(0, e));
		vec2 g = (.5 / e)*vec2(xp - xm, yp - ym), eg = normalize(g);
		double gm = (1. / (e*e)) * (F(x + e * eg) + F(x - e * eg) - 2.*Fx);
		double t0 = 0, t1 = abs(1. / gm), phi0 = Fx, phi1 = F(x - t1 * g);
		GoldenSectionSearch_1d([&](double t) { return F(x - t * g); }, t0, t1, phi0, phi1, 1e-4);
		vec2 dx = g * (phi0 < phi1 ? t0 : t1);
#else
		double D[3][3];
		for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
			D[i][j] = F(x + e * vec2(j - 1, i - 1));
		double Fx = D[1][1];
		vec2 g = (.5 / e) * vec2(D[1][2] - D[1][0], D[2][1] - D[0][1]);
		vec2 g2 = (1. / (e*e)) * vec2(D[1][2] + D[1][0] - 2.*Fx, D[2][1] + D[0][1] - 2.*Fx);
		double gxy = (.25 / (e*e)) * ((D[0][0] + D[2][2]) - (D[0][2] + D[2][0]));
		double m = 1. / (g2.x*g2.y - gxy * gxy);
		vec2 dx = m * vec2(g.x*g2.y - g.y*gxy, g2.x*g.y - g.x*gxy);
		if (2.*dot(dx, g) < dot(dx, vec2(g2.x*dx.x + gxy * dx.y, gxy*dx.x + g2.y*dx.y))) dx = -dx;
		if (dot(dx, dx) > 1e-4) {
			double t0 = 0, t1 = 2, phi0 = Fx, phi1 = F(x - t1 * dx);
			GoldenSectionSearch_1d([&](double t) { return F(x - t * dx); }, t0, t1, phi0, phi1, 1e-4);
			dx = dx * (phi0 < phi1 ? t0 : t1);
		}
#endif

		if (!(dx.sqr() > 1e-16 && abs(Fx - F0x) > 1e-12)) {
			if (0.0*dot(dx, dx) == 0.0) return x - dx;
			else return x;
		}
		else x -= dx;
		F0x = Fx;
#ifdef _DEBUG_OPTIMIZATION
		drawLine(x, x + dx, COLOR({ 0,0,0 }));
		drawDot(x, 5, COLOR({ 0,0,0 }));
#endif
	}
	return x;
}




// Numerical Recipe
// http://user.it.uu.se/~matsh/opt/f8/f8.html
