#version 300 es
precision highp float;

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
float getDivU(vec2 coord) {
    vec2 ddx = (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.0*eps);
    vec2 ddy = (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.0*eps);
    return ddx.x + ddy.y;
}


// https://www.shadertoy.com/view/3tjGWm
vec3 hueShift(vec3 c, float s) {
    vec3 m=vec3(cos(s),s=sin(s)*.5774,-s);
    return c*mat3(m+=(1.-m.x)/3.,m.zxy,m.yzx);
}

void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 xy = coord / iResolution.xy;

    vec2 u = getU(coord);
    float mag = length(u.xy);
    float dir = atan(u.y,u.x);
    vec3 col = mag/(mag+0.4)*hueShift(vec3(1,0,0), dir);
    // col = vec3(0.5+getDivU(coord));

    col = texture(samplerC, xy).xyz;

    fragColor = vec4(col, 1);
}
