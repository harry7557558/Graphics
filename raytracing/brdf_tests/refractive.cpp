#include "_.h"

#include "raytracing/cubemap/cubemap.h"
#include "raytracing/bvh_ply.h"


// which model to use
#define SPHERE 0


Cubemap cubemap;
BVH *model;

void mainInit() {
	Center = vec3(0.0, 0.0, 0.7);

	// load model
#if !SPHERE
	//model = loadModel("modeling/pbrt_dragon_oriented.ply");
	//model = loadModel("modeling/stanford_dragon_oriented.ply");
	model = loadModel("UI/Homework/AVI3M/eye_full_stand.stl");
#endif

	// load cubemap
	char filenames[6][256];
	for (int i = 0; i < 6; i++)
		sprintf(filenames[i], "raytracing/cubemap/shadertoy_uffizi_gallery/%d.jpg", i);
	cubemap = Cubemap(filenames[0], filenames[1], filenames[2], filenames[3], filenames[4], filenames[5]);

}


vec3f light(vec3f rd) {
	vec3f col = cubemap.sample(rd);
	//col = vec3f(0.5f, 0.5f, 0.6f);
	vec3f bri = vec3f(1.0f) + vec3f(2.0f, 1.5f, 1.0f) * pow(max(dot(rd, normalize(vec3f(-0.2f, -0.5f, 0.5f))), 0.), 4.);
	return col * bri;
}


// sphere intersection function
bool intersectSphere(vec3f O, float r, vec3f ro, vec3f rd, float &t, vec3f &n) {
	ro -= O;
	float b = -dot(ro, rd), c = dot(ro, ro) - r * r;
	float delta = b * b - c;
	if (delta < 0.0) return false;
	delta = sqrt(delta);
	float t1 = b - delta, t2 = b + delta;
	if (t1 > t2) std::swap(t1, t2);
	if (t1 > t || t2 < 0.) return false;
	t = t1 > 0. ? t1 : t2;
	n = normalize(ro + rd * t);
	return true;
}


vec3f sampleCosWeighted(vec3f n, uint32_t &seed) {
	vec3f u = normalize(cross(n, vec3f(1.2345f, 2.3456f, -3.4561f)));
	vec3f v = cross(u, n);
	float rn = rand01f(seed);
	vec2f rh = sqrt(rn) * cossin(2.0f*PIf*rand01f(seed));
	double rz = sqrt(1. - rn);
	return rh.x * u + rh.y * v + rz * n;
}

vec3f sampleFresnelDielectric(vec3f rd, vec3f n, float n1, float n2, uint32_t &seed) {
	float eta = n1 / n2;
	float ci = -dot(n, rd);
	if (ci < 0.0f) ci = -ci, n = -n;
	float ct = 1.0f - eta * eta * (1.0f - ci * ci);
	if (ct < 0.0f) return rd + 2.0f*ci*n;
	ct = sqrt(ct);
	float Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
	float Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
	float R = 0.5f * (Rs * Rs + Rp * Rp);
	//R = 0.15f;
	return rand01f(seed) > R ?
		rd * eta + n * (eta * ci - ct)  // refraction
		: rd + 2.0f*ci*n;  // reflection
}



vec3f mainRender(vec3f ro, vec3f rd, uint32_t &seed) {

	enum Material {
		background,
		diffuse,
		refractive
	};

	vec3f m_col = vec3f(1.0), col;
	bool is_inside = false;

	for (int iter = 0; iter < 64; iter++) {
		ro += 1e-4f*rd;
		vec3f n, min_n;
		float t, min_t = INFINITY;
		Material material = background;

		// plane
		t = -ro.z / rd.z;
		if (t > 0.0f) {
			min_t = t, min_n = vec3f(0, 0, 1);
			col = vec3f(0.8f, 0.9f, 1.0f);
			material = diffuse;
		}

		// object
		t = min_t;
#if SPHERE
		if (intersectSphere(vec3f(0.0f, 0.0f, 1.0f), 1.0f, ro, rd, t, n)) {
#else
		if (intersectBVH(model, ro, rd, t, n)) {
#endif
			min_t = t, min_n = n;
			col = is_inside ? exp(-0.4f*vec3f(0.8f, 0.6f, 0.05f)*t) : vec3f(1.0f);
			material = refractive;
		}

		// update ray
		if (material == background) {
			//if (iter == 0) return vec3f(0.f);
			col = light(rd);
			return m_col * col;
		}
		m_col *= col;
		min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
		ro = ro + rd * min_t;
		if (material == diffuse) {  // diffuse
			rd = sampleCosWeighted(min_n, seed);
		}
		else if (material == refractive) {  // steel ball
			vec2f eta = is_inside ? vec2f(1.5f, 1.0f) : vec2f(1.0f, 1.5f);
			rd = sampleFresnelDielectric(rd, min_n, eta.x, eta.y, seed);
		}
		if (dot(rd, min_n) < 0.0) {
			is_inside ^= 1;
		}
	}
	return m_col;
}
