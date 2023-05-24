#version 300 es
precision highp float;

// calculate the divergence of velocity and store in z

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;

uniform sampler2D samplerU;


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


void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 u = getU(coord);
    float div = getDivU(coord);
    fragColor = vec4(u, div, 1);
}
