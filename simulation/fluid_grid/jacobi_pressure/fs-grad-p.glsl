#version 300 es
precision highp float;

// subtract the gradient of pressure from the velocity

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;
uniform int iterIndex;

uniform sampler2D samplerU;
uniform sampler2D samplerP;


vec2 getU(vec2 coord) {
    // return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).xy;
    ivec2 c = ivec2(coord), r = ivec2(iResolution);
    vec2 s = vec2(
        c.x<0||c.x>=r.x ? -1.0 : 1.0,
        c.y<0||c.y>=r.y ? -1.0 : 1.0 );
    c = (r-1)-abs(r-1-abs(c));
    return s * texelFetch(samplerU, c, 0).xy;
}
float getDivU(vec2 coord) {
    vec2 ddx = (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.0*eps.x);
    vec2 ddy = (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.0*eps.y);
    return ddx.x + ddy.y;
}
float getP(vec2 coord) {
    // return texelFetch(samplerP, ivec2(mod(coord,iResolution)), 0).x;
    ivec2 c = ivec2(coord), r = ivec2(iResolution);
    c = (r-1)-abs(r-1-abs(c));
    return texelFetch(samplerP, c, 0).x;
}
vec2 getGradP(vec2 coord) {
    return vec2(
        (getP(coord+vec2(1,0))-getP(coord-vec2(1,0)))/(2.0*eps.x),
        (getP(coord+vec2(0,1))-getP(coord-vec2(0,1)))/(2.0*eps.y)
    );
}

void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 u = getU(coord);
    vec2 du = -getGradP(coord);
    if (isnan(du.x+du.y)) du = vec2(0);
    // du = vec2(0);
    fragColor = vec4(u+du, 0,1);
}
