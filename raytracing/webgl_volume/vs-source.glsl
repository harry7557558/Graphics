#version 300 es
precision highp float;

in vec4 vertexPosition;

uniform float iRz, iRx, iRy;
uniform float iSc;
uniform vec2 iResolution;

uniform float uIso;
uniform vec3 uBoxRadius;

out vec3 vRo;
out vec3 vRd;


mat3 calcCamera(float rz, float rx, float ry) {
    float cz = cos(rz), sz = sin(rz),
          cx = cos(rx), sx = sin(rx),
          cy = cos(ry), sy = sin(ry);
    return mat3(cy, 0, sy, 0, 1, 0, -sy, 0, cy) *
        mat3(-sz, cz, 0, -cz*sx, -sz*sx, cx, -cz*cx, -sz*cx, -sx);
}


void main(void) {
    vec2 sc = iResolution.xy / (iSc*min(iResolution.x, iResolution.y));

    // camera rotation
    mat3 R = calcCamera(iRz, iRx, iRy);
    vec3 u = R[0], v = R[1], w = R[2];

    // calculate the minimum and maximum depth
    float mind = 8., maxd = -8.;
    for (float z=-1.0; z<=1.001; z+=2.0)
    for (float y=-1.0; y<=1.001; y+=2.0)
    for (float x=-1.0; x<=1.001; x+=2.0) {
        vec3 p = uBoxRadius * vec3(x, y, z);
        float depth = dot(p, w);
        mind = min(mind, depth), maxd = max(maxd, depth);
    }

    // center of graph, minus control pad
    vec2 pad = 2.0 * vec2(290, 230) / iResolution.xy;
    float area = pad.x*pad.y;
    vec2 com = vec2(1.0)-0.5*pad;
    vec2 ctr = -com*area / (4.0-area);
    vec2 pos = (vertexPosition.xy - clamp(ctr, vec2(-0.2), vec2(0.2))) * sc;

    // set ray origin and direction
    vRo = u * pos.x + v * pos.y + mix(mind, maxd, uIso)*w;
    vRd = w;

    gl_Position = vertexPosition;
}
