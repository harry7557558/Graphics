#iChannel0 "self"

#iChannel1 "../cubemaps/shadertoy_uffizi_gallery/{}.jpg"
#iChannel1::Type "CubeMap"



uint seed = 0u;
uint randu() { return seed = seed * 1664525u + 1013904223u; }
float rand01() { return float(randu()) * (1./4294967296.); }

#define PI 3.1415926


vec3 light(vec3 rd) {
    vec3 col = texture(iChannel1, rd).xyz;
    vec3 bri = vec3(1.0) + vec3(2.0) * pow(max(dot(rd, normalize(vec3(-0.2, -0.5, 0.5))), 0.), 4.);
    return col * bri;
}


// sphere intersection function
bool intersectSphere(vec3 o, float r, vec3 ro, vec3 rd,
        inout float t, inout vec3 n) {
    ro -= o;
    float b = -dot(ro, rd), c = dot(ro, ro) - r * r;
    float delta = b * b - c;
    if (delta < 0.0) return false;
    delta = sqrt(delta);
    float t1 = b - delta, t2 = b + delta;
    if (t1 > t2) t = t1, t1 = t2, t2 = t;
    if (t1 > t || t2 < 0.) return false;
    t = t1 > 0. ? t1 : t2;
    n = normalize(ro + rd * t);
    return true;
}


float OrenNayarBRDF(float sigma, vec3 n, vec3 wi, vec3 wo) {
    vec3 u = normalize(cross(n, vec3(1.2345, 2.3456, -3.4561)));
    vec3 v = cross(u, n);
    wi = vec3(dot(wi, u), dot(wi, v), dot(wi, n));
    wo = vec3(dot(wo, u), dot(wo, v), dot(wo, n));
    float s2 = sigma * sigma;
    float A = 1.0 - s2 / (2.0*(s2 + 0.33));
    float B = 0.45*s2 / (s2 + 0.09);
    float theta_i = acos(wi.z), phi_i = atan(wi.y, wi.x);
    float theta_o = acos(wo.z), phi_o = atan(wo.y, wo.x);
    return (A + B * max(0.0, cos(phi_i-phi_o))*sin(max(theta_i,theta_o))*tan(min(theta_i,theta_o))) / PI;
}

vec3 sampleCosWeighted(vec3 n) {
    vec3 u = normalize(cross(n, vec3(1.2345, 2.3456, -3.4561)));
    vec3 v = cross(u, n);
    float rn = rand01();
    float an = 2.0*PI*rand01();
    vec2 rh = sqrt(rn) * vec2(cos(an), sin(an));
    float rz = sqrt(1. - rn);
    return rh.x * u + rh.y * v + rz * n;
}

vec3 sampleFresnelDielectric(vec3 rd, vec3 n, float n1, float n2) {
    float eta = n1 / n2;
    float ci = -dot(n, rd);
    if (ci < 0.0) ci = -ci, n = -n;
    float ct = 1.0 - eta * eta * (1.0 - ci * ci);
    if (ct < 0.0) return rd + 2.0*ci*n;
    ct = sqrt(ct);
    float Rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct);
    float Rp = (n1 * ct - n2 * ci) / (n1 * ct + n2 * ci);
    float R = 0.5 * (Rs * Rs + Rp * Rp);
    return rand01() > R ?
        rd * eta + n * (eta * ci - ct)  // refraction
        : rd + 2.0*ci*n;  // reflection
}

vec3 sampleMicrofacet(vec3 wi, vec3 n, float alpha, inout vec3 m_col) {
    vec3 u = normalize(cross(n, vec3(1.2345, 2.3456, -3.4561)));
    vec3 v = cross(u, n);
    wi = vec3(dot(wi, u), dot(wi, v), dot(wi, n));
    vec3 wo, m;
    float D, G, F;
    // GGX
    float su = 2.0*PI*rand01();
    float sv = rand01();
    sv = acos(sqrt((1.0-sv)/((alpha*alpha-1.)*sv+1.)));
    m = vec3(sin(sv)*vec2(cos(su),sin(su)), cos(sv));  // half vector
    wo = -(wi-2.0*dot(wi,m)*m);
    D = wo.z<0. ? 0. : 1.;
    // Geometry
    float tan2_theta_i = (1.0-wi.z*wi.z)/(wi.z*wi.z);
    float tan2_theta_o = (1.0-wo.z*wo.z)/(wo.z*wo.z);
    float lambda_i = 0.5*(sqrt(1.0+alpha*alpha*tan2_theta_i)-1.0);
    float lambda_o = 0.5*(sqrt(1.0+alpha*alpha*tan2_theta_o)-1.0);
    G = 1.0/(1.0+lambda_i+lambda_o);
    // Fresnel
    const float eta_2 = 1.5;
    const float eta_k = 2.0;
    F = ((eta_2-1.0)*(eta_2-1.0)+4.0*eta_2*pow(1.0-dot(wi,m),5.0)+eta_k*eta_k)
        / ((eta_2+1.0)*(eta_2+1.0)+eta_k*eta_k);
    // Put all together
    //float Fr = (D*G*F) / (4.0*wi.z*wo.z);
    //float Fr = (D*G) / (4.0*wi.z*wo.z);
    float Fr = D / (wi.z*wo.z);
    float Fr_cos = Fr * wo.z*wi.z;  // wi.z or wo.z??
    m_col *= Fr_cos;
    return wo.x * u + wo.y * v + wo.z * n;
}



vec3 mainRender(vec3 ro, vec3 rd) {

    const int background = 0;
    const int lambertian = 1;
    const int oren_nayar = 2;
    const int specular = 3;
    const int refractive = 4;
    const int ggx_diffuse = 5;
    const int ggx_glossy = 6;

    vec3 m_col = vec3(1.0), col;
    bool is_inside = false;

    for (int iter = 0; iter < 64; iter++) {
        ro += 1e-4f*rd;
        vec3 n, min_n;
        float t, min_t = 1e12;
        int material = background;

        // plane
        t = -ro.z / rd.z;
        if (t > 0.0) {
            min_t = t, min_n = vec3(0, 0, 1);
            col = vec3(0.9, 0.95, 0.98);
            material = lambertian;
        }

        // objects
        for (float i = 1.; i <= 6.; i++) {
            t = min_t;
            vec3 pos = vec3(2.2*vec2(cos(2.*PI*i/6.), sin(2.*PI*i/6.)), 1.0+1e-4);
            if (intersectSphere(pos, 1.0, ro, rd, t, n)) {
                min_t = t, min_n = n;
                col = vec3(1.0);
                material = int(i);
            }
        }

        // update ray
        if (material == background) {
            // if (iter == 0) return vec3(0.f);
            col = light(rd);
            return m_col * col;
        }
        m_col *= col;
        min_n = dot(rd, min_n) < 0. ? min_n : -min_n;  // ray hits into the surface
        ro = ro + rd * min_t;
        if (material == lambertian) {
            rd = sampleCosWeighted(min_n);
        }
        if (material == oren_nayar) {
            vec3 wi = -rd;
            vec3 wo = sampleCosWeighted(min_n);
            m_col *= PI * OrenNayarBRDF(0.5, min_n, wi, wo);
            rd = wo;
        }
        if (material == specular) {
            rd = rd - 2.0*dot(rd, min_n)*min_n;
        }
        if (material == refractive) {
            vec2 eta = is_inside ? vec2(1.5, 1.0) : vec2(1.0, 1.5);
            rd = sampleFresnelDielectric(rd, min_n, eta.x, eta.y);
        }
        if (material == ggx_diffuse) {
            rd = sampleMicrofacet(-rd, min_n, 0.5, m_col);
        }
        if (material == ggx_glossy) {
            rd = sampleMicrofacet(-rd, min_n, 0.08, m_col);
        }
        if (dot(rd, min_n) < 0.0) {
            is_inside = !is_inside;
        }
        if (m_col == vec3(0.0)) break;
    }
    return m_col;
}



void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    // https://www.shadertoy.com/view/4djSRW, MIT licence
    vec3 p3 = fract(vec3(fragCoord, iFrame) * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    float h = fract((p3.x + p3.y) * p3.z);
    seed = uint(16777216.*h);


    // camera
    float rx = 2.0*(iMouse.y/iResolution.y)-0.5;
    //float rx = 3.14*(iMouse.y/iResolution.y)-1.57;
    float rz = -iMouse.x/iResolution.x*4.0*3.14;
    vec3 w = vec3(cos(rx)*vec2(cos(rz),sin(rz)), sin(rx));
    vec3 u = vec3(-sin(rz),cos(rz),0);
    vec3 v = cross(w,u);
    vec3 ro = 10.0*w + vec3(0, 0, 0.7);
    vec2 uv = 2.0*(fragCoord.xy+vec2(rand01(),rand01())-0.5)/iResolution.xy - vec2(1.0);
    vec3 rd = mat3(u,v,-w)*vec3(uv*iResolution.xy, 1.8*length(iResolution.xy));
    rd = normalize(rd);

    // calculate pixel color
    vec3 col = mainRender(ro, rd);
    vec4 rgbn = texelFetch(iChannel0, ivec2(fragCoord), 0);
    if (iMouse.z>0.) rgbn.w = 0.0;
    fragColor = vec4((rgbn.xyz*rgbn.w + col)/(rgbn.w+1.0), rgbn.w+1.0);
}
