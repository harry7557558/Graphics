#version 300 es
precision highp float;

// iteratively solves for Δu = ν ∇²(u+Δu) Δt

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;
uniform int iterIndex;

uniform sampler2D samplerU;
uniform sampler2D samplerDu;


// u+Δu
vec2 getU(vec2 coord) {
    ivec2 c = ivec2(coord), r = ivec2(iResolution);
    c = (r-1)-abs(r-1-abs(c));
    vec2 u = texelFetch(samplerU, c, 0).xy;
    vec2 du = texelFetch(samplerDu, c, 0).xy;
    return u + du;
}

// ∇²(u+Δu)
vec2 getLapU(vec2 coord) {
    return (
        (getU(coord+vec2(1,0))+getU(coord-vec2(1,0))-2.*getU(coord))/(eps.x*eps.x) +
        (getU(coord+vec2(0,1))+getU(coord-vec2(0,1))-2.*getU(coord))/(eps.y*eps.y)
    );
}
vec2 getLapUE(vec2 coord) {
    return (
        (getU(coord+vec2(1,0))+getU(coord-vec2(1,0)))/(eps.x*eps.x) +
        (getU(coord+vec2(0,1))+getU(coord-vec2(0,1)))/(eps.y*eps.y)
    );
}


void main() {
    if (iFrame == 0) {
        fragColor = vec4(0,0,0,1);
        return;
    }
    vec2 coord = gl_FragCoord.xy;
    float k_vis = 0.0001;
#if 0
    vec2 lap = getLapU(coord);
    vec2 du = k_vis * lap * dt;
    du /= 1.0 + 2. * k_vis * dot(1./(eps*eps), vec2(1)) * dt;
    fragColor = vec4(du,0,1);
#else
    vec2 u0 = texelFetch(samplerU, ivec2(coord), 0).xy;
    vec2 lape = getLapUE(coord);
    vec2 u1 = u0 + k_vis * lape * dt;
    u1 /= 1.0 + 2. * k_vis * dot(1./(eps*eps), vec2(1)) * dt;
    fragColor = vec4(u1-u0,0,1);
#endif
}
