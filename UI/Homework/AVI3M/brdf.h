// Monte-Carlo path tracing
#include "numerical/random.h"

// importance sampling
vec3 randdir_cosWeighted(vec3 n, uint32_t &seed) {
	vec3 u = ncross(n, vec3(1.2345, 2.3456, -3.4561));
	vec3 v = cross(u, n);
	double rn = rand01(seed);
	vec2 rh = sqrt(rn) * cossin(2.*PI*rand01(seed));
	double rz = sqrt(1. - rn);
	return rh.x * u + rh.y * v + rz * n;
}
// the ray comes from a medium with reflective index n1 to a medium with reflective index n2
vec3 randdir_Fresnel(vec3 rd, vec3 n, double n1, double n2, uint32_t &seed) {
	double eta = n1 / n2;
	double ci = -dot(n, rd);
	if (ci < 0.) ci = -ci, n = -n;
	double ct = 1.0 - eta * eta * (1.0 - ci * ci);
	if (ct < 0.) return rd + 2.*ci*n;
	ct = sqrt(ct);
	double Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
	double Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
	double R = 0.5 * (Rs * Rs + Rp * Rp);
	return rand01(seed) > R ?
		rd * eta + n * (eta * ci - ct)  // refraction
		: rd + 2.*ci*n;  // reflection
}
