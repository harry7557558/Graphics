#version 300 es
precision highp float;

// advect u = u -u*∇u

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;

uniform sampler2D samplerU;
uniform sampler2D samplerDu;  // due to viscosity

uniform vec4 iMouse;

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

// Forward
mat2 getGradU(vec2 coord) {
    return mat2(
        (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.*eps.x),
        (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.*eps.y)
    );
}

// Semi-Lagrangian
vec2 getUBillinear(vec2 xy) {
    vec2 coord = xy * iResolution.xy - 0.5;
    vec2 f = fract(coord);
    vec2 p = floor(coord);
    vec2 c00 = getU(p+vec2(0,0));
    vec2 c10 = getU(p+vec2(1,0));
    vec2 c01 = getU(p+vec2(0,1));
    vec2 c11 = getU(p+vec2(1,1));
    vec2 c = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
    vec2 s = vec2(
        xy.x<0.||xy.x>=1. ? -1.0 : 1.0,
        xy.y<0.||xy.y>=1. ? -1.0 : 1.0 );
    // return s * c;
    return c;
}

void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 xy = coord / iResolution.xy;

    //vec2 u0 = getU(coord);
    vec2 u0 = getUBillinear(xy);
    mat2 gradU = getGradU(coord);
    vec2 u, dudt;

    if (iFrame==0) {
        const float k = 4.0 * PI;
        u = sin(vec2(1,1)*k*xy).yx + vec2(1,0)*exp(-pow(4.*(xy.y-0.5),2.));
        // u = vec2(1.5, 0.5*sin(k*xy.x))*exp(-pow(12.*(xy.y-0.5),2.));
        // u = vec2(0.0);
        u *= 2.0*pow(16.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 1.0);
        fragColor = vec4(u, 0,1);
        return;
    }

#if 0
    // foward, sometimes unstable
    u = u0 - gradU * u * dt;
#else
    // semi-Lagrangian, sometimes looks weird
    u = getUBillinear(xy-eps/iResolution*u0*dt);
#endif

    // mouse drag
    vec2 mouse = iMouse.xy / iResolution.xy;
    vec2 d = mouse - xy;
    dudt = 10.0 * iMouse.zw * exp(-40.0*dot(d,d));
    u += dudt * dt;

    // velocity diffusion (viscosity)
    u += texelFetch(samplerDu, ivec2(coord), 0).xy;

    // viscous drag
    float k_drag = 0.01;
#if 0
    u -= k_drag * u0 * dt;
#else
    u -= k_drag * u0 * dt / (1.0 + k_drag * dt);
#endif

    fragColor = vec4(u, 0,1);
}
