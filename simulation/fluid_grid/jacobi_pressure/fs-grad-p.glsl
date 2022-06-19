#version 300 es
precision highp float;

// subtract the gradient of pressure from the velocity

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform float eps;
uniform float dt;
uniform int iterIndex;

uniform sampler2D samplerU;
uniform sampler2D samplerP;


vec2 getU(vec2 coord) {
    return texelFetch(samplerU, ivec2(coord)%ivec2(iResolution), 0).xy;
}
float getDivU(vec2 coord) {
    vec2 ddx = (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.0*eps);
    vec2 ddy = (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.0*eps);
    return ddx.x + ddy.y;
}
float getP(vec2 coord) {
    return texelFetch(samplerP, ivec2(coord)%ivec2(iResolution), 0).x;
}
vec2 getGradP(vec2 coord) {
    return vec2(getP(coord+vec2(1,0))-getP(coord-vec2(1,0)),
                getP(coord+vec2(0,1))-getP(coord-vec2(0,1)))/(2.0*eps);
}

void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 u = getU(coord);
    vec2 du = -getGradP(coord);
    fragColor = vec4(u+du, 0,1);
}
