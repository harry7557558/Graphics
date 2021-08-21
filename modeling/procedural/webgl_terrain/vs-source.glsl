precision highp float;

attribute vec4 aVertexPosition;

uniform float iRx, iRz, iRy;
uniform float iDist;
uniform vec2 iResolution;

varying vec3 vPos;
varying vec3 vNor;

#define Fbm Fbm2

void main(void) {

    vec2 xy = aVertexPosition.xy;
    float z = Fbm(xy);
    vec2 grad = vec2(Fbm(xy+vec2(0.01,0))-Fbm(xy-vec2(0.01,0)),Fbm(xy+vec2(0,0.01))-Fbm(xy-vec2(0,0.01)))/0.02;
    vPos = vec3(xy, z);
    vNor = vec3(grad, 1.0);
    if (z < 0.0) {
        vPos = vec3(xy, 0.0);
        vNor = vec3(0.0, 0.0, 1.0);
    }
    vec4 pos = vec4(vPos, 1.0);

    vec3 w = vec3(cos(iRx)*vec2(cos(iRz),sin(iRz)), sin(iRx));
    vec3 u = vec3(-sin(iRz), cos(iRz), 0.0);
    vec3 v = cross(w, u);

    mat4 T = mat4(
        u.x, v.x, w.x, 0.0,
        u.y, v.y, w.y, 0.0,
        u.z, v.z, w.z, 0.0,
        0.0, 0.0, 0.0, 1.0 * iDist
    );
    T = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, -0.5,
        0.0, 0.0, 0.0, 1.0
    ) * T;
    pos = T * pos;
    pos.xy *= 0.5 * length(iResolution.xy) / iResolution;
    pos.z *= 0.01;

    gl_Position = pos;
}
