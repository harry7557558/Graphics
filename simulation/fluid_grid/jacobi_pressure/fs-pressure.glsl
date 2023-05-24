#version 300 es
precision highp float;

// iteratively solves for ∇²p = div(u)

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;
uniform int iterIndex;

uniform sampler2D samplerU;
uniform sampler2D samplerP;


float getDivU(vec2 coord) {
    return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).z;
}
float getP(vec2 coord) {
    // return texelFetch(samplerP, ivec2(mod(coord,iResolution)), 0).x;
    ivec2 c = ivec2(coord), r = ivec2(iResolution);
    c = (r-1)-abs(r-1-abs(c));
    return texelFetch(samplerP, c, 0).x;
}


void main() {
    vec2 c = gl_FragCoord.xy;
    float div = getDivU(c);
    vec2 e2 = eps*eps;
    // float p1 = 0.25*(getP(c+vec2(1,0))+getP(c-vec2(1,0))+getP(c+vec2(0,1))+getP(c-vec2(0,1))-e2.x*div);  // eps.x==eps.y
    float p1 = ((getP(c+vec2(1,0))+getP(c-vec2(1,0)))/e2.x+(getP(c+vec2(0,1))+getP(c-vec2(0,1)))/e2.y-div)/(2./e2.x+2./e2.y);
    fragColor = vec4(p1,0,0,1);
}
