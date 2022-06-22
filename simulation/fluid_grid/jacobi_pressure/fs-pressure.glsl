#version 300 es
precision highp float;

// iteratively solves for ∇²p = div(u)

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform float eps;
uniform float dt;
uniform int iterIndex;

uniform sampler2D samplerU;
uniform sampler2D samplerP;


float getDivU(vec2 coord) {
    return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).z;
}
float getP(vec2 coord) {
    if (iterIndex == 0) return 0.0;
    return texelFetch(samplerP, ivec2(mod(coord,iResolution)), 0).x;
}


void main() {
    vec2 c = gl_FragCoord.xy;
    float div = getDivU(c);
    float p1 = 0.25*(getP(c+vec2(1,0))+getP(c-vec2(1,0))+getP(c+vec2(0,1))+getP(c-vec2(0,1))-eps*eps*div);
    // float p1 = 0.25*(getP(c+vec2(2,0))+getP(c-vec2(2,0))+getP(c+vec2(0,2))+getP(c-vec2(0,2))-4.0*eps*eps*div);
    fragColor = vec4(p1,0,0,1);
}
