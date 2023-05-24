#version 300 es
precision highp float;

out vec4 fragColor;

in vec3 vRo;
in vec3 vRd;

#define uVisual {%uVisual%}  // visualization mode
#define uColormap {%uColormap%}  // colormap
uniform float uIso;  // the "completeness" of the object
uniform vec3 uBoxRadius;  // bounding box

#define ZERO min(uIso, 0.)
#define PI 3.1415926


// Sample volume

uniform highp sampler3D uSampler3D;

float sampleTexture(vec3 p) {
    p = (p + uBoxRadius) / (2.0*uBoxRadius);
    return texture(uSampler3D, 1.0-p).x;
}

float sampleTexture_clamped(vec3 p) {
    p = (p + uBoxRadius) / (2.0*uBoxRadius);
    if (p.x<0.||p.y<0.||p.z<0. || p.x>1.||p.y>1.||p.z>1.) return 0.0;
    return texture(uSampler3D, 1.0-p).x;
}

// central difference gradient, 6 samples, O(e^2)
vec3 sampleTextureGradC(in vec3 p, in float e) {
    return -vec3(
        sampleTexture_clamped(p+vec3(e,0,0)) - sampleTexture_clamped(p-vec3(e,0,0)),
        sampleTexture_clamped(p+vec3(0,e,0)) - sampleTexture_clamped(p-vec3(0,e,0)),
        sampleTexture_clamped(p+vec3(0,0,e)) - sampleTexture_clamped(p-vec3(0,0,e))
    ) / (2.0*e);
}

// tetrahedron gradient, 4 samples, O(e^1)
vec3 sampleTextureGradT(in vec3 p, in float e) {
	float a = sampleTexture(p+vec3(e,e,e));
	float b = sampleTexture(p+vec3(e,-e,-e));
	float c = sampleTexture(p+vec3(-e,e,-e));
	float d = sampleTexture(p+vec3(-e,-e,e));
	return (.25/e)*vec3(a+b-c-d,a-b+c-d,a-b-c+d);
}

// attempt to find analytical gradient
// isn't faster than numerical gradient, not used
vec4 sampleTextureGradVal(vec3 p) {
    p = 1.0 - (p + uBoxRadius) / (2.0*uBoxRadius);
    ivec3 size = textureSize(uSampler3D, 0);
    vec3 xyz = p * vec3(size);
	ivec3 i0 = ivec3(floor(xyz));
    i0 = clamp(i0, ivec3(0), size-ivec3(1));
	float v000 = texelFetch(uSampler3D, i0 + ivec3(0, 0, 0), 0).x;
	float v001 = texelFetch(uSampler3D, i0 + ivec3(0, 0, 1), 0).x;
	float v010 = texelFetch(uSampler3D, i0 + ivec3(0, 1, 0), 0).x;
	float v011 = texelFetch(uSampler3D, i0 + ivec3(0, 1, 1), 0).x;
	float v100 = texelFetch(uSampler3D, i0 + ivec3(1, 0, 0), 0).x;
	float v101 = texelFetch(uSampler3D, i0 + ivec3(1, 0, 1), 0).x;
	float v110 = texelFetch(uSampler3D, i0 + ivec3(1, 1, 0), 0).x;
	float v111 = texelFetch(uSampler3D, i0 + ivec3(1, 1, 1), 0).x;
	vec3 f = xyz - vec3(i0);
	vec4 gradval = v000 * vec4(0, 0, 0, 1) +
		(v100 - v000) * vec4(1.0, 0, 0, f.x) +
		(v010 - v000) * vec4(0, 1.0, 0, f.y) +
		(v001 - v000) * vec4(0, 0, 1.0, f.z) +
		(v000 + v110 - v010 - v100) * vec4(f.y, f.x, 0, f.x*f.y) +
		(v000 + v101 - v001 - v100) * vec4(f.z, 0, f.x, f.x*f.z) +
		(v000 + v011 - v001 - v010) * vec4(0, f.z, f.y, f.y*f.z) +
		(v111 - v011 - v101 - v110 + v100 + v001 + v010 - v000) * vec4(f.y*f.z, f.x*f.z, f.x*f.y, f.x*f.y*f.z)
		;
    return gradval * vec4(size, 1.0);
}



// Intersection with bounding cuboid

bool boxIntersection(vec3 ro, vec3 rd, out float tn, out float tf) {
    vec3 inv_rd = 1.0 / rd;
    vec3 n = inv_rd*(ro);
    vec3 k = abs(inv_rd)*abs(uBoxRadius);
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
    vec3 pb = abs(ro) - abs(uBoxRadius);
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
    float v_old = 0.0, v;
    for (t = t0; t < t1; t += step_size) {
        v = sampleTexture(ro+rd*t);
        if (v > iso) break;
        v_old = v;
    }
    if (v <= iso) return vec3(0.0);
    // raymarching
    for (int s = 0; s < 4; s += 1) {
        v_old = v;
        step_size *= -0.5;
        for (int i = 0; i < 2; i++) {
            t += step_size;
            v = sampleTexture(ro+rd*t);
            if ((v-iso)*(v_old-iso)<0.0) break;
        }
    }
    vec3 n = normalize(sampleTextureGradC(ro+rd*t, 0.01));
    float col = 0.2+0.1*n.y+0.6*max(dot(n, normalize(vec3(0.5,0.5,1.0))),0.0);
    return vec3(col);
}

// not really volumetric
vec3 vVolumetricMip(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float step_size = (t1-t0) / step_count;
    float maxval = 0.0;
    for (float t=t0; t<t1; t+=step_size) {
        float v = sampleTexture(ro+rd*t);
        maxval = max(maxval, v);
    }
    float v = pow(maxval, 0.5*(1.0-uIso)/uIso);
    return v * uColormap(maxval);
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

// the common way of volume rendering
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

// light emission doesn't increase when light absorption increase
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

// opacity is based on the magnitude of gradient
vec3 vVolumetricGradient(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    float k = 0.5*uIso/(1.0-uIso);
    for (float t=t0; t<t1; t+=dt) {
        float v = sampleTexture(ro+rd*t);
        vec3 grad = sampleTextureGradT(ro+rd*t, 0.005);
        //vec3 grad = sampleTextureGradVal(ro+rd*t).xyz;
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = k*length(grad);
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
    }
    return totcol;
}

// faster than volumetric gradient
vec3 vVolumetricDiff(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    float k = 0.5*uIso/(1.0-uIso);
    float v_old = sampleTexture(ro+rd*t0), v;
    for (float t=t0; t<t1; t+=dt) {
        v = sampleTexture(ro+rd*t);
        float grad = abs(v-v_old)/dt;
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = k*grad;
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
        v_old = v;
    }
    return totcol;
}

// 2 layers of high opacity
vec3 vPeriodic2(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    for (float t=t0; t<t1; t+=dt) {
        float v = sampleTexture(ro+rd*t);
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = 0.5+0.5*cos(2.0*6.283*(v-uIso))<0.16 ? 60.0*uIso*v : 0.0;
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
    }
    return totcol;
}

// 3 layers of high opacity
vec3 vPeriodic3(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    for (float t=t0; t<t1; t+=dt) {
        float v = sampleTexture(ro+rd*t);
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = 0.5+0.5*cos(3.0*6.283*(v-uIso))<0.20 ? 40.0*uIso*v : 0.0;
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
    }
    return totcol;
}

// simulate the x-ray image of an isosurface
vec3 vIsosurfXray(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float dt = (t1-t0) / step_count;
    vec3 totcol = vec3(0.0);
    float iso = 1.0-uIso;
    float v_old = sampleTexture(ro+rd*t0), v;
    for (float t=t0; t<t1; t+=dt) {
        v = sampleTexture(ro+rd*t);
        if ((v_old-iso)*(v-iso) < 0.0) {
            float linintp = (iso-v_old)/(v-v_old);
            vec3 p = ro + rd * mix(t-dt, t, linintp);
            vec3 grad = normalize(sampleTextureGradC(p, 0.01));
            //vec3 col = clamp(uColormap(iso), 0.0, 1.0);
            vec3 col = clamp(uColormap((p.z+abs(uBoxRadius.z))/abs(2.0*uBoxRadius.z)), 0.0, 1.0);
            col = col / abs(dot(rd, grad));
            totcol += 0.05*col;
        }
        v_old = v;
    }
    return totcol;
}

// volume rendering + raymarching isosurface
vec3 vSkinBone(in vec3 ro, in vec3 rd, float t0, float t1) {
    float step_count = min(ceil((t1-t0)/STEP), MAX_STEP);
    float t = t0, dt = (t1-t0) / step_count;
    float iso1 = 1.0-uIso;
    float iso2 = 0.5-0.5*uIso;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    float v_old = 0.0, v;
    for (t = t0; t < t1; t += dt) {
        v = sampleTexture(ro+rd*t);
        vec3 col = clamp(uColormap(v), 0.0, 1.0);
        float absorb = abs(v-iso2)<0.2*(1.0-uIso) ? 20.0*uIso : 0.0;
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
        if (v > iso1) break;
        v_old = v;
    }
    if (v < iso1) return totcol;
    for (int s = 0; s < 4; s += 1) {
        v_old = v;
        dt *= -0.5;
        for (int i = 0; i < 2; i++) {
            t += dt;
            v = sampleTexture(ro+rd*t);
            if ((v-iso1)*(v_old-iso1)<0.0) break;
        }
    }
    vec3 n = normalize(sampleTextureGradC(ro+rd*t, 0.01));
    float col = 0.2+0.1*n.y+0.6*max(dot(n, normalize(vec3(0.5,0.5,1.0))),0.0);
    return totcol + col * totabs;
}

// adapted from an aid in creating Shadertoy raymarching shaders
vec3 vSdfVisualizer(in vec3 ro, in vec3 rd, float t0, float t1) {
    float t = t0;
    vec3 totcol = vec3(0.0);
    float totabs = 1.0;
    float v_old = (1.0-uIso)-sampleTexture(ro+rd*t), v;
    float dt = min(STEP, abs(v_old));
    for (float i=ZERO; i<MAX_STEP;) {
        t += dt;
        if (t > t1) return totcol;
        v = (1.0-uIso)-sampleTexture(ro+rd*t);
        if (v*v_old<0.) break;
        vec3 col = uColormap(0.5+0.5*sin(16.0*PI*0.5*(v_old+v)));
        float absorb = 0.3;
        totabs *= exp(-absorb*dt);
        totcol += col*absorb*totabs*dt;
        v_old = v;
        dt = clamp(v, 0.1*STEP, STEP);
        if (dt < 1e-3) break;
        if (++i >= MAX_STEP) return totcol;
    }
    if (v*v_old < 0.) {
        for (int s = int(ZERO); s < 4; s += 1) {
            v_old = v, dt *= -0.5;
            for (int i = int(ZERO); i < 2; i++) {
                t += dt, v = (1.0-uIso)-sampleTexture(ro+rd*t);
                if (v*v_old<0.0) break;
            }
        }
    }
    vec3 grad = normalize(sampleTextureGradC(ro+rd*t, 1e-2));
    vec3 col = vec3(0.2+0.05*grad.y+0.75*max(dot(grad, normalize(vec3(0.5,0.5,1.0))),0.0));
    return totcol + col * totabs;
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
    int visual = uVisual;
    if (visual-- == 0) col = vSlice(ro, rd, t0, t1);
    if (visual-- == 0) col = vMip(ro, rd, t0, t1);
    if (visual-- == 0) col = vXray(ro, rd, t0, t1);
    if (visual-- == 0) col = vIsosurf(ro, rd, t0, t1);
    if (visual-- == 0) col = vVolumetricMip(ro, rd, t0, t1);
    if (visual-- == 0) col = vVolumetricXray(ro, rd, t0, t1);
    if (visual-- == 0) col = vVolumetricIntegral(ro, rd, t0, t1);
    if (visual-- == 0) col = vVolumetricShadow(ro, rd, t0, t1);
    if (visual-- == 0) col = vVolumetricGradient(ro, rd, t0, t1);
    if (visual-- == 0) col = vVolumetricDiff(ro, rd, t0, t1);
    if (visual-- == 0) col = vPeriodic2(ro, rd, t0, t1);
    if (visual-- == 0) col = vPeriodic3(ro, rd, t0, t1);
    if (visual-- == 0) col = vIsosurfXray(ro, rd, t0, t1);
    if (visual-- == 0) col = vSkinBone(ro, rd, t0, t1);
    if (visual-- == 0) col = vSdfVisualizer(ro, rd, t0, t1);

    fragColor = vec4(col, 1.0);
}
