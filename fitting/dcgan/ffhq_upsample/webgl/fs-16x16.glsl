#version 300 es
precision highp float;

uniform sampler2D uSrc;
uniform sampler2D uWeights;


int getIRes() {
    ivec2 p = ivec2(gl_FragCoord.xy) % 64;
    int i = p.y * 64 + p.x;
    return i * 4;
}
float getSrc(int i) {
    ivec2 p0 = (ivec2(gl_FragCoord.xy)/64)*64;
    vec4 r = texelFetch(uSrc, p0+ivec2((i/4)%64, (i/4)/64), 0);
    return i%4==0 ? r.x : i%4==1 ? r.y : i%4==2 ? r.z : r.w;
}
float getW(int i) {
    int w = textureSize(uWeights, 0).x;
    vec4 r = texelFetch(uWeights, ivec2((i/4)%w, (i/4)/w), 0);
    return i%4==0? r.x : i%4==1 ? r.y : i%4==2 ? r.z : r.w;
}

out vec4 fragColor;
void main() {

    int i0 = getIRes();
    if (i0 >= 4096) discard;

    float v[4];
    for (int i = i0; i < i0 + 4; i++) {
        int chr = i / 256;  // res channel
        int ri = (i-chr*256) / 16, rj = i % 16;  // res pixel
        float s = 0.0;
        for (int chs = 0; chs < 16; chs++) {  // src channel
            int w0 = (chr * 16 + chs) * 9;
            for (int ci = 0; ci < 3; ci++) for (int cj = 0; cj < 3; cj++) {  // filter
                int si = ri + ci - 1, sj = rj + cj - 1;
                if (si >= 0 && si < 16 && sj >= 0 && sj < 16) {
                    si /= 2, sj /= 2;  // reverse upscale
                    float wv = getW(w0 + ci * 3 + cj);
                    float sv = getSrc(chs * 64 + (si * 8 + sj));
                    s += wv * sv;
                }
            }
        }
        v[i-i0] = max(0.1*s, s);
    }

    fragColor = vec4(v[0], v[1], v[2], v[3]);

}
