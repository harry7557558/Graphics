#version 300 es
precision highp float;

out vec4 fragColor;

in vec3 vRo;
in vec3 vRd;

#define uVisual {%uVisual%}  // visualization mode
#define uColormap {%uColormap%}  // colormap
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


// Color maps - https://www.shadertoy.com/view/NsSSRK

vec3 cThermometer(float t) {
  float r = .453+.122*t+.385*cos(4.177*t-2.507);
  float g = .284+.142*t+.554*cos(4.181*t-1.918);
  float b = .464+.05*t+.475*cos(3.217*t-.809);
  return vec3(r, g, b);
}

vec3 cGreenPinkTones(float t) {
  float r = .529-.054*t+.55*cos(5.498*t+2.779);
  float g = .21+.512*t+.622*cos(4.817*t-1.552);
  float b = .602-.212*t+.569*cos(5.266*t+2.861);
  return vec3(r, g, b);
}

vec3 cBlueGreenYellow(float t) {
  float r = 2.083+4.676*t+6.451*cos(.818*t+1.879);
  float g = -.467+1.408*t+.504*cos(2.071*t-.424);
  float b = -1.062+1.975*t+1.607*cos(1.481*t+.447);
  return vec3(r, g, b);
}

vec3 cRainbow(float t) {
  float r = 132.228-245.968*t+755.627*cos(.3275*t-1.7461);
  float g = .385-1.397*t+1.319*cos(2.391*t-1.839);
  float b = -142.825+270.693*t+891.307*cos(.3053*t+1.4092);
  return vec3(r, g, b);
}

vec3 cTemperatureMap(float t) {
  float r = .372+.707*t+.265*cos(5.201*t-2.515);
  float g = .888-2.123*t+1.556*cos(2.483*t-1.959);
  float b = 1.182-.943*t+.195*cos(8.032*t+2.875);
  return vec3(r, g, b);
}


// Visualization modes

#define STEP 0.005
#define MAX_STEP 250.

// thin slice
vec3 vSlice(in vec3 ro, in vec3 rd, float t0, float t1) {
    vec3 pb = abs(ro) - uBoxRadius;
    if (max(max(pb.x, pb.y), pb.z) > 0.0) return vec3(0.02);
    float v = sampleTexture(ro);
    return uColormap(v);
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
    //float v = uIso/(1.0-uIso) * maxval;
    float v = pow(maxval, (1.0-uIso)/uIso);
    return uColormap(v);
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
    float k = 4.0*uIso/(1.0-uIso);
    float v = 1.0-exp(-k*totval);
    return uColormap(v);
}

// isosurface
vec3 vIsosurf(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float step_size = (t1-t0) / step_count;
    float iso = 1.0-uIso;
    float t = t0;
    float v_old = 0.0;
    for (t=t0; t<t1; t+=step_size) {
        float v = sampleTexture(ro+rd*t);
        if (v > iso) {
            t = t - step_size + (iso-v_old)/(v-v_old)*step_size;  // linear interpolation
            vec3 n = normalize(sampleTextureGrad(ro+rd*t));
            float col = 0.2+0.1*n.y+0.6*max(dot(n, normalize(vec3(0.5,0.5,1.0))),0.0);
            //return uColormap(col);
            return vec3(col);
        }
        v_old = v;
    }
    return vec3(0.0);
}

// similar to x-ray, integrate color instead of intensity
vec3 vVolumetricXray(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;  // formerly step_size
    vec3 totcol = vec3(0.0);
    for (float t=t0; t<t1; t+=dt) {
        float v = sampleTexture(ro+rd*t);
        vec3 col = v*clamp(uColormap(v), 0.0, 1.0);
        totcol += col*dt;
    }
    float k = 4.0*uIso/(1.0-uIso);
    return 1.0-exp(-k*totcol);
}

vec3 vVolumetricIntegral(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;  // absorption
    float k = 8.0*uIso/(1.0-uIso);
    for (float t=t0; t<t1; t+=dt) {
        float v = sampleTexture(ro+rd*t);
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = k*v;
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
    }
    return totcol;
}

vec3 vVolumetricShadow(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    float k = 4.0*uIso/(1.0-uIso);
    for (float t=t0; t<t1; t+=dt) {
        float v = sampleTexture(ro+rd*t);
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = k*v;
        totabs *= exp(-absorb*dt);
        totcol += col*totabs*dt;  // the only line different from the previous one
    }
    return totcol;
}



void main(void) {
    vec3 ro = vRo, rd = normalize(vRd);
    fragColor = vec4(0, 0, 0, 1);

    float t0, t1;
    if (!boxIntersection(ro, rd, t0, t1)) {
        //fragColor = vec4(0.05, 0.05, 0.08, 1);
        return;
    }

    vec3 col = vec3(0.0);
    if (uVisual == 0) col = vSlice(ro, rd, t0, t1);
    if (uVisual == 1) col = vMip(ro, rd, t0, t1);
    if (uVisual == 2) col = vXray(ro, rd, t0, t1);
    if (uVisual == 3) col = vIsosurf(ro, rd, t0, t1);
    if (uVisual == 4) col = vVolumetricXray(ro, rd, t0, t1);
    if (uVisual == 5) col = vVolumetricIntegral(ro, rd, t0, t1);
    if (uVisual == 6) col = vVolumetricShadow(ro, rd, t0, t1);

    fragColor = vec4(col, 1.0);
}
