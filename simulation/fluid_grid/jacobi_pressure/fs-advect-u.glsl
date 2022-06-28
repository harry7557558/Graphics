#version 300 es
precision highp float;

// advect u = u -u*∇u

out vec4 fragColor;

uniform vec2 iResolution;
uniform int iFrame;
uniform vec2 eps;
uniform float dt;

uniform sampler2D samplerU;

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
mat2 getGradU(vec2 coord) {
    return mat2(
        (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.*eps.x),
        (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.*eps.y)
    );
}
vec2 getLapU(vec2 coord) {
    return (
        (getU(coord+vec2(1,0))+getU(coord-vec2(1,0))-2.*getU(coord))/(eps.x*eps.x) +
        (getU(coord+vec2(0,1))+getU(coord-vec2(0,1))-4.*getU(coord))/(eps.y*eps.y) );
}
float getCurlU(vec2 coord) {
    vec2 ddx = (getU(coord+vec2(1,0))-getU(coord-vec2(1,0)))/(2.0*eps.x);
    vec2 ddy = (getU(coord+vec2(0,1))-getU(coord-vec2(0,1)))/(2.0*eps.y);
    return ddy.x - ddx.y;
}


void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 xy = coord / iResolution.xy;

    vec2 u = getU(coord);
    mat2 gradU = getGradU(coord);

    if (iFrame==0) {
        const float k = 4.0 * PI;
        u = sin(vec2(1,1)*k*xy).yx + vec2(1,0)*exp(-pow(4.*(xy.y-0.5),2.));
        // u = vec2(1.5, 0.5*sin(k*xy.x))*exp(-pow(12.*(xy.y-0.5),2.));
        // u = vec2(0.0);
        u *= 2.0*pow(16.0*xy.x*xy.y*(1.0-xy.x)*(1.0-xy.y), 1.0);
        // gradU = mat2(0,k*cos(k*xy.x),k*cos(k*xy.y),0);
        gradU = mat2(1.0);
    }

    vec2 dudt = -gradU * u;
    if (iFrame > 0) {
        // velocity diffusion (viscosity)
        dudt += 0.01 * getLapU(coord);
        // viscous drag
        dudt -= 0.01 * u;
        // vorticity confinement
        vec2 vt = vec2(abs(getCurlU(coord+vec2(1,0)))-abs(getCurlU(coord-vec2(1,0))),
                       abs(getCurlU(coord+vec2(0,1)))-abs(getCurlU(coord-vec2(0,1)))
                 ) / (2.0*eps);
        vt = vec2(-1,1)*vt.yx / (length(vt)+1e-8);
        // dudt += 0.1*getCurlU(coord)*vt;
    }
    else dudt = vec2(0.0);

    vec2 mouse = iMouse.xy / iResolution.xy;
    vec2 d = mouse - xy;
    dudt += 10.0 * iMouse.zw * exp(-40.0*dot(d,d));

    u += dudt * dt;

    fragColor = vec4(u, 0,1);
}
