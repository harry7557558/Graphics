#version 300 es
precision highp float;

uniform sampler2D uSrc;
uniform int tileSize;

uniform mat4 w[16];

out vec4 fragColor;
void main() {

    ivec2 xyf = ivec2(gl_FragCoord.xy) % (2*tileSize);
    ivec2 xyi = (ivec2(gl_FragCoord.xy) - xyf)/2;

    vec4 r = vec4(0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            ivec2 xy = xyf-2+ivec2(i,j);
            if (xy.x%2!=0 || xy.y%2!=0) continue;
            xy /= 2;
            if (xy.x<0 || xy.y>=tileSize || xy.y<0 || xy.y>=tileSize) continue;
            mat4 R = w[j*4+i];
            r += R * texelFetch(uSrc, xyi+xy, 0);
        }
    }

    fragColor = r;
    // fragColor = 0.1*texelFetch(uSrc, ivec2(gl_FragCoord.xy)/2, 0);
}
