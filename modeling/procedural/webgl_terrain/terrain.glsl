// #include <noise.glsl>

#define BaseNoise  SimplexNoise2D
#define BaseNoiseG SimplexNoise2Dg
#define BaseNoiseV SimplexNoise2Dv

vec3 NoiseG(vec2 xy) {
	const float var_grad = BaseNoiseV.x;
	const float var_val = BaseNoiseV.z;
	const float EXP_H = 0.5;  // required average height
	const float EXP_G = 0.8;  // required average gradient
	const float a = EXP_H / sqrt(var_val);
	const float k = EXP_G / (a * sqrt(var_grad));
	return BaseNoiseG(k*xy) * vec3(vec2(a*k), a);
}

// Simple FBM
float Fbm1(vec2 xy) {
    float h = 0.0;
    float ampl = 1.0;
    for (int k = 0; k < 8; k++) {
        ampl *= 0.5;
        h += ampl * NoiseG(xy).z;
        xy = 2.0 * mat2(0.6, -0.8, 0.8, 0.6) * xy + 2.0;
    }
    return h;
}

// Terrain with faked erotion
float Fbm2(vec2 xy) {
	float h = 0.0;
	float ampl = 1.0;
	vec2 sum_grad = vec2(0.0);
	for (int i = 0; i < 8; i++) {
		vec3 gradval = NoiseG(xy);
		sum_grad += gradval.xy;
		ampl *= 0.5;
		h += ampl * gradval.z / (1.0 + dot(sum_grad, sum_grad));
        xy = 2.0 * mat2(0.6, -0.8, 0.8, 0.6) * xy + 2.0;
	}
	return h;
}