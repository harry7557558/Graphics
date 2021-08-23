#version 300 es
precision highp float;

out vec4 fragColor;

in vec3 vRo;
in vec3 vRd;

#define uVisual {%uVisual%}  // visualization mode
uniform float uIso;  // the "completeness" of the object
uniform vec3 uBoxRadius;  // bounding box


// Sample volume

uniform highp sampler3D uSampler3D;

float sampleTexture(vec3 p_) {
    vec3 p = (p_ + uBoxRadius) / (2.0*uBoxRadius);
    if (p.x<0.||p.y<0.||p.z<0. || p.x>1.||p.y>1.||p.z>1.) return 0.0;
    return texture(uSampler3D, 1.0-p).x;
}

vec3 sampleTextureGrad(vec3 p) {
    const float e = 0.01;
    return -vec3(
        sampleTexture(p+vec3(e,0,0)) - sampleTexture(p-vec3(e,0,0)),
        sampleTexture(p+vec3(0,e,0)) - sampleTexture(p-vec3(0,e,0)),
        sampleTexture(p+vec3(0,0,e)) - sampleTexture(p-vec3(0,0,e))
    ) / (2.0*e);
}


// Intersection with bounding cuboid

bool boxIntersection(vec3 ro, vec3 rd, out float tn, out float tf) {
    vec3 inv_rd = 1.0 / rd;
    vec3 n = inv_rd*(ro);
    vec3 k = abs(inv_rd)*uBoxRadius;
    vec3 t1 = -n - k, t2 = -n + k;
    tn = max(max(t1.x, t1.y), t1.z);
    tf = min(min(t2.x, t2.y), t2.z);
    if (tn > tf) return false;
    return true;
}


// Visualization modes

#define STEP 0.005
#define MAX_STEP 250.

// thin slice
vec3 vSlice(in vec3 ro, in vec3 rd, float t0, float t1) {
    vec3 pb = abs(ro) - uBoxRadius;
    if (max(max(pb.x, pb.y), pb.z) > 0.0) return vec3(0.02);
    return vec3(sampleTexture(ro));
}

// maximum intensity projection
vec3 vMip(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float step_size = (t1-t0) / step_count;
    float maxval = 0.0;
    for (float t=t0; t<t1; t+=step_size) {
        float v = sampleTexture(ro+rd*t);
        maxval = max(maxval, v);
    }
    float k = uIso/(1.0-uIso);
    return vec3(clamp(k*maxval, 0.0, 1.0));
}

// beer-lambert law
vec3 vXray(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float step_size = (t1-t0) / step_count;
    float totval = 0.0;
    for (float t=t0; t<t1; t+=step_size) {
        float v = sampleTexture(ro+rd*t);
        totval += v*step_size;
    }
    float k = 2.0*uIso/(1.0-uIso);
    float v = 1.0-exp(-k*totval);
    return vec3(v);
}

// isosurface
vec3 vIsosurf(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float step_size = (t1-t0) / step_count;
    float t = t0;
    float iso = 1.0-uIso;
    float v_old = 0.0;
    for (t=t0; t<t1; t+=step_size) {
        float v = sampleTexture(ro+rd*t);
        if (v > iso) {
            t = t - step_size + (iso-v_old)/(v-v_old)*step_size;
            vec3 n = normalize(sampleTextureGrad(ro+rd*t));
            vec3 col = vec3(0.2+0.1*n.y+0.6*max(dot(n, normalize(vec3(0.5,0.5,1.0))),0.0));
            return col;
        }
        v_old = v;
    }
    return vec3(0.0);
}



void main(void) {
    vec3 ro = vRo, rd = normalize(vRd);

    float t0, t1;
    if (!boxIntersection(ro, rd, t0, t1)) {
        fragColor = vec4(0.05, 0.05, 0.08, 1);
        return;
    }

    vec3 col = vec3(0.0);
    if (uVisual == 0) col = vSlice(ro, rd, t0, t1);
    if (uVisual == 1) col = vMip(ro, rd, t0, t1);
    if (uVisual == 2) col = vXray(ro, rd, t0, t1);
    if (uVisual == 3) col = vIsosurf(ro, rd, t0, t1);

    fragColor = vec4(col, 1.0);
}
