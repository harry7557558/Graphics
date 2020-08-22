
#ifndef __INC_INTERPOLATION_H

#define __INC_INTERPOLATION_H


// linear interpolation
template<typename T>
T lerp(T a, T b, double u) {
	return a + (b - a) * u;
	return a * (1 - u) + b * u;
}

// cubic interpolation by derivative
template<typename T>
T intp_d(T a, T b, T dadt, T dbdt, double u) {
	return a * (1 + u * u*(-3 + 2 * u)) + b * (u*u*(3 - 2 * u)) + dadt * (u*(1 + u * (u - 2))) + dbdt * (u*u*(u - 1));
}

// slerp, second derivative, Bezier, B, Hermite, Catmull-Rom, Lagrange, bicubic, rational, trigonometric, fitting... 


#endif  // __INC_INTERPOLATION_H

