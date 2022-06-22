#version 300 es
precision highp float;

// calculate the divergence of velocity and store in z

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform float eps;
uniform float dt;

uniform sampler2D samplerU;


vec2 getU(vec2 coord) {
    return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).xy;
}
float getDivU(vec2 coord) {
    vec2 ddx = (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.0*eps);
    vec2 ddy = (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.0*eps);
    return ddx.x + ddy.y;
}


void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 u = getU(coord);
    float div = getDivU(coord);
    fragColor = vec4(u, div, 1);
}
