#version 300 es
precision highp float;

uniform sampler2D uSrc;
uniform int tileSize;

uniform mat4 w[9];

out vec4 fragColor;
void main() {

    ivec2 xyf = ivec2(gl_FragCoord.xy) % tileSize;
    ivec2 xyi = ivec2(gl_FragCoord.xy) - xyf;

    vec4 r = vec4(0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ivec2 xy = xyf-1+ivec2(i,j);
            if (xy.x<0 || xy.y>=tileSize || xy.y<0 || xy.y>=tileSize) continue;
            mat4 R = w[j*3+i];
            r += R * texelFetch(uSrc, xyi+xy, 0);
        }
    }

    fragColor = r;
    // fragColor = 0.1*texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0);
}
