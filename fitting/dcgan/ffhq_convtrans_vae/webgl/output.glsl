#version 300 es
precision highp float;

uniform sampler2D uSrc;

out vec4 fragColor;
void main() {
    ivec2 coord = ivec2(gl_FragCoord.xy);
    coord.y = textureSize(uSrc, 0).y-1 - coord.y;
    vec4 c = texelFetch(uSrc, coord, 0);
    c = 1.0 / (1.0+exp(-c));
    fragColor = vec4(c.xyz, 1.0);
}
