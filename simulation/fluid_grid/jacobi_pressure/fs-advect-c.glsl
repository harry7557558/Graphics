#version 300 es
precision highp float;

// advect color

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;

uniform sampler2D samplerU;
uniform sampler2D samplerC;

#define PI 3.1415926


vec2 getU(vec2 coord) {
    // return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).xy;
    ivec2 c = ivec2(coord), r = ivec2(iResolution);
    vec2 s = vec2(
        c.x<0||c.x>=r.x ? -1.0 : 1.0,
        c.y<0||c.y>=r.y ? -1.0 : 1.0 );
    c = (r-1)-abs(r-1-abs(c));
    return s * texelFetch(samplerU, c, 0).xy;
}

vec4 getC(vec2 coord) {
    // return texelFetch(samplerC, ivec2(mod(coord,iResolution)), 0);
    ivec2 c = ivec2(coord), r = ivec2(iResolution);
    c = (r-1)-abs(r-1-abs(c));
    return texelFetch(samplerC, c, 0);
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
    if (isnan(u.x+u.y)) u = vec2(0.0);
    vec2 p = xy - u * dt * eps.x/eps;

    if (u==vec2(0) || iFrame==0) {
        const float k = 4.0 * PI;
        fragColor = 0.0 + 1.0 * vec4(
            step(sin(k*p)-cos(k*p),cos(k*p)+sin(k*p)),
            step(cos(k*p)-cos(k*p),sin(k*p)+sin(k*p))
        ).zxyw;
        fragColor = mix(fragColor, vec4(step(0.0,sin(k*p.x)*sin(k*p.y))), 0.5);
    }

    else fragColor = getCbillinear(p);
}
