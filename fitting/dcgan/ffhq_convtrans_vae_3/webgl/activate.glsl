#version 300 es
precision highp float;

uniform sampler2D uSrc;

uniform float negative_slope;

out vec4 fragColor;
void main() {
    vec4 c = texelFetch(uSrc, ivec2(gl_FragCoord.xy), 0);
    // fragColor = max(c, 0.0);
    fragColor = c / (1.0+exp(-c));
}
