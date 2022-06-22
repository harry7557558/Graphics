#version 300 es
precision highp float;

// advect color

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform float eps;
uniform float dt;

uniform sampler2D samplerU;
uniform sampler2D samplerC;

#define PI 3.1415926


vec2 getU(vec2 coord) {
    return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).xy;
}

vec4 getC(vec2 coord) {
    return texelFetch(samplerC, ivec2(mod(coord,iResolution)), 0);
}
vec4 getCbillinear(vec2 uv) {
    vec2 size = iResolution;  // vec2(textureSize(samplerC))
    vec2 xy = uv * size - 0.5;
    vec2 f = fract(xy);
    vec2 p = floor(xy);
    vec4 c00 = getC(p+vec2(0,0));
    vec4 c10 = getC(p+vec2(1,0));
    vec4 c01 = getC(p+vec2(0,1));
    vec4 c11 = getC(p+vec2(1,1));
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}


void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 xy = coord / iResolution.xy;

    vec2 u = getU(coord);
    vec2 p = xy - u * dt;

    if (iFrame==0) {
        const float k = 10.0 * PI;
        fragColor = 0.0 + 1.0 * vec4(
            step(sin(k*p)-cos(k*p),cos(k*p)+sin(k*p)),
            step(cos(k*p)-cos(k*p),sin(k*p)+sin(k*p))
        ).zxyw;
    }

    else fragColor = getCbillinear(p);
}
