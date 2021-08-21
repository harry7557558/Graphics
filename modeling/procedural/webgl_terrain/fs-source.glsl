precision highp float;

uniform float iRx, iRz, iRy;
uniform float iDist;
uniform vec2 iResolution;

varying vec3 vPos;
varying vec3 vNor;

// https://www.shadertoy.com/view/NsSSRK
vec3 SandyTerrain(float t) {
  float r = .903-.539*t+.319*cos(4.28*t-2.369);
  float g = .481+.071*t+.271*cos(4.704*t-2.322);
  float b = .264-.027*t+.058*cos(5.68*t-2.617);
  return clamp(vec3(r,g,b), 0.0, 1.0);
}


void main() {
    vec3 col = vec3(1.0);

    col = 0.5+0.5*normalize(vNor);
    col = mix(vec3(0.4,0.6,0.8), SandyTerrain(0.6*vPos.z), smoothstep(0.0,1.0,40.0*vPos.z));
    col = (0.2+max(0.8*dot(normalize(vNor),normalize(vec3(0.5,0.0,1.0))),0.0))*col;

    gl_FragColor = vec4(col, 1.0);
}
