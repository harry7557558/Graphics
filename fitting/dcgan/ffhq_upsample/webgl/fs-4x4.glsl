#version 300 es
precision highp float;

uniform sampler2D uWeights;

uniform int iFrame;


// Hash without Sine, David Hoskins, MIT License
// https://www.shadertoy.com/view/4djSRW
vec2 hash23(vec3 p3) {
	p3 = fract(p3 * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

// Random normal distribution
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float randn(vec3 seed) {
    float t = 0.04 * float(iFrame);
    vec3 seed1 = seed + floor(t);
    vec3 seed2 = seed1 + 1.0;
    vec2 uv = mix(hash23(seed1), hash23(seed2), fract(t));
    float a = sqrt(-2.*log(1.0-uv.x));
    float b = 6.283185*uv.y;
    return a*sin(b);
}


int getIRes() {
    ivec2 p = ivec2(gl_FragCoord.xy) % 64;
    int i = p.y * 64 + p.x;
    return i * 4;
}
float getSrc(int i) {
    ivec2 id = ivec2(gl_FragCoord.xy) / 64;
    return randn(vec3(id, i));
}
float getW(int i) {
    int w = textureSize(uWeights, 0).x;
    vec4 r = texelFetch(uWeights, ivec2((i/4)%w, (i/4)/w), 0);
    return i%4==0? r.x : i%4==1 ? r.y : i%4==2 ? r.z : r.w;
}


out vec4 fragColor;
void main() {

    int i0 = getIRes();
    if (i0 >= 256) discard;

    float v[4];
    for (int i = i0; i < i0 + 4; i++) {
        float s = 0.0;
        for (int j = 0; j < 32; j++) {
            s += getW(i*32+j) * getSrc(j);
        }
        v[i-i0] = max(0.1*s, s);
    }

    fragColor = vec4(v[0], v[1], v[2], v[3]);

}
