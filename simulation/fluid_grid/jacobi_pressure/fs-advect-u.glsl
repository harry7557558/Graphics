#version 300 es
precision highp float;

// advect u = u -u*∇u

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform float eps;
uniform float dt;

uniform sampler2D samplerU;

uniform vec3 iMouse;

#define PI 3.1415926


vec2 getU(vec2 coord) {
    return texelFetch(samplerU, ivec2(mod(coord,iResolution)), 0).xy;
}
mat2 getGradU(vec2 coord) {
    return mat2(
        getU(coord+vec2(1,0))-getU(coord-vec2(1,0)),
        getU(coord+vec2(0,1))-getU(coord-vec2(0,1))
    ) / (2.0*eps);
}
vec2 getLapU(vec2 coord) {
    return (getU(coord+vec2(1,0))+getU(coord-vec2(1,0))+getU(coord+vec2(0,1))+getU(coord-vec2(0,1))
        - 4.0*getU(coord)) / (eps*eps);
}


void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 xy = coord / iResolution.xy;

    vec2 u = getU(coord);
    mat2 gradU = getGradU(coord);
    
    if (iFrame==0) {
        const float k = 4.0 * PI;
        u = sin(k*xy).yx + vec2(1,0)*exp(-pow(4.*(xy.y-0.5),2.));
        // gradU = mat2(0,k*cos(k*xy.x),k*cos(k*xy.y),0);
        gradU = mat2(1.0);
    }

    vec2 dudt = -gradU * u;
    dudt += 0.1 * getLapU(coord);
    dudt -= 0.1 * u;

    if (iMouse.z > 0.) {
        vec2 mouse = iMouse.xy / iResolution.xy;
        vec2 d = mouse - xy;
        dudt += vec2(5, 5) * exp(-20.0*dot(d,d));
    }

    u += dudt * dt;

    fragColor = vec4(u, 0,1);
}
