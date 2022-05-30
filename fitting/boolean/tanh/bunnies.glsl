// Comparing different models to fit the Stanford Bunny.
// Note: None of the models is an SDF.

// 0: Polynomial basis
// 1: Trigonometric basis, fixed frequencies
// 2: Trigonometric basis, variable frequencies
// 3: Neural network with sine activation, 1 hidden layer
// 4: Neural network with tanh activation, 1 hidden layer
// 5: Neural network with ReLU activation, 1 hidden layer
// 6: Neural network with sine activation, 2 hidden layers
// 7: Neural network with tanh activation, 2 hidden layers
// 10: Radial basis, varying centers
#define MODEL 6

// Comments:
// Training difficulty: basis < neural < variable basis
// Running speed: ReLU > trigs > sqrt > tanh > pow
// Insensitivity to weights: neural > basis > variable basis
// 

// Interactive rendering of ground truth (128x128x128):
// https://harry7557558.github.io/Graphics/raytracing/webgl_volume/index.html#volume=bunny-sdf&visual=isosurf
// Its ears seem to be thinner than many other bunnies on this site.

// X-data:
// Uniform grid with coordinates between -1 and 1, non-inclusive.

// Y-data:
// The above model, possibly with a reduced resolution.
// -1.0 for inside, 1.0 for outside.

// The training process first minimizes the following function:
// mean[ ( (1-Y)ln(1+exp(2*model(X))) + (1+Y)ln(1+exp(-2*model(X))) )^2 ]
// It is quadratic form-alike when weights are away from a minimum.
// It is less likely to stuck in a local minimum when optimizing this function.

// Then, it minimizes the mean squared error:
// mean[ (tanh(model(X))-Y)^2 ]
// This loss function produces a visually more accurate result.

// Weights are normalized and rounded so they are integers with at most 4 digits.

// Each model has a comment containing the following information:
//   - Number of scalar weights of the model
//   - Training data resolution
//   - MSE calculated using the (unrounded) weights
//   - Size of minimized code in kilibytes
//   - Time to render one frame on my device (800x450, integrated Intel GPU)


// Polynomial function with degree 8
// 165 weights, 64x64x64, MSE=0.0163, 1.06KB, 41ms
// Optimizer: BFGS (SciPy)
int wPoly8[165] = int[165](-4,-7,23,-26,34,-3,72,-84,99,-145,-201,81,-47,-81,1,-123,409,-345,272,-134,587,-1049,489,223,-365,43,691,619,-184,681,-238,921,-67,-217,764,1198,2639,-412,1248,1211,-179,1733,-2606,1526,2623,-1576,267,-127,2080,-668,1533,141,367,1411,-728,1141,-2426,2268,1454,897,1275,1411,531,-969,7438,-1613,323,4225,13,1648,839,-5439,3971,-146,-2497,1089,-1221,1859,-864,219,-2357,821,800,-1663,-2933,-4489,1314,-2972,-1020,462,101,982,2692,6794,3669,-394,-1036,3886,3759,-5001,6771,1505,-7414,2726,4155,657,1770,-3813,-6016,2270,-3189,2451,-4906,2211,-4538,96,-715,-2906,1114,-1589,5336,2020,2794,2784,-2791,-2151,-145,-431,578,-3797,-9999,-551,-4024,-1825,-230,1834,2762,8238,-5052,6496,3561,-1803,734,3123,2012,-3852,4534,-2797,-5124,1016,576,-2005,5140,-3635,3494,3950,-3128,-1168,-839,4656,-2482,3202,348,-304,1995);
#define spow(u,e) (mod(e,2.)==0.?1.:sign(u))*pow(abs(u),e)
float bunnyPoly8(vec3 p) {
    int c = 0; float s=0.;
    for (float d=0.; d<=8.; d++) {
        for (float i=0.; i<=d; i++) {
            for (float j=0.; j<=d-i; j++) {
                s += float(wPoly8[c++]) *
                    spow(p.x,i) * spow(p.y,j) * spow(p.z,d-i-j);
            }
        }
    }
    return s;
}

// Trigonometric functions with fixed frequencies
// 125 weights, 64x64x64, MSE=0.0322, 1.00KB, 11ms
// Optimizer: BFGS (SciPy)
int wTrigC[125] = int[125](9999,-3120,-1738,814,-158,-2273,1753,-4161,799,1446,1487,-2523,-950,-1127,-1120,-4080,792,3957,-243,-2412,-2123,2261,1799,619,1498,-3619,738,-6012,-322,-1431,-928,1154,1623,3167,-314,1624,-3383,2961,1622,3095,1016,-394,2871,-1680,1569,386,-1206,4634,424,2291,-277,-3887,-107,1103,-476,1667,1913,-477,-1951,1207,192,-760,1356,-1006,-789,1555,2554,-5176,-2806,-265,-1724,1675,551,-1053,-715,-1839,2099,4071,518,-1969,341,-894,30,-470,950,-1081,2265,293,1883,2288,304,-946,-3354,-2694,-986,-3470,1012,-2513,-1469,-1772,1313,2359,-646,429,557,1960,86,-1988,1982,-4523,-765,-975,103,3969,304,-2234,-1569,3040,1587,1037,353,-1711,-211,868,204);
float bunnyTrigC(vec3 p) {
    int c=0; float s=0.;
    for (int i=0; i<3; i++) for (int j=0; j<3; j++) for (int k=0; k<3; k++) {
        vec3 q = 2.*vec3(i,j,k)*p;
        for (int di=0; di<2; di++) for (int dj=0; dj<2; dj++) for (int dk=0; dk<2; dk++) {
            if (!((i==0&&di==1)||(j==0&&dj==1)||(k==0&&dk==1)))
                s += float(wTrigC[c++]) *
                    (di==0?cos(q.x):sin(q.x))*
                    (dj==0?cos(q.y):sin(q.y))*
                    (dk==0?cos(q.z):sin(q.z));
        }
    }
    return s;
}

// Trigonometric functions with variable frequencies
// 128 weights, 64x64x64, MSE=0.0288, 0.81KB, 12ms
// Optimizer: Adam + BFGS (Python)
int wTrigF[128] = int[128](5081,699,877,953,2682,1389,2219,-312,3159,-279,827,1488,409,2408,-1782,-888,-1726,1366,2370,2360,1874,2228,722,494,-10031,878,-353,-190,335,-1147,-1166,1202,0,-407,-185,-512,-216,463,-469,730,0,-968,542,-482,311,593,-27,-1132,-194,-1022,117,-59,448,628,581,-406,114,1108,-358,1498,670,527,-30,-367,0,198,906,19,249,-180,-841,817,0,-947,611,-486,-639,-372,744,2,-267,101,353,-218,230,-407,405,-43,92,-59,-914,-444,-1203,10,292,-64,-13,-702,118,-1324,-438,-213,-80,466,0,192,-360,-215,-776,700,88,3113,690,-243,215,224,291,-715,530,-221,144,304,930,81,-581,-8,-114,-180);
float bunnyTrigF(vec3 p) {
    float s=0.;
    for (int i=0; i<32; i++) {
        vec3 q = 0.01*vec3(wTrigF[i+32],wTrigF[i+64],wTrigF[i+96])*p;
        s += float(wTrigF[i])
            * ((i>>0)%2>0?sin(q.x):cos(q.x))
            * ((i>>1)%2>0?sin(q.y):cos(q.y))
            * ((i>>2)%2>0?sin(q.z):cos(q.z));
    }
    return s;
}

// Neural network with 1 hidden layer and sine activation
// 121 weights, 64x64x64, MSE=0.0279, 0.80KB, 5ms
// Optimizer: Adam + BFGS (C++)
int wSine1[121] = int[121](66,-25,-1198,-502,-1303,-29,1198,-66,393,-461,-376,154,534,1407,-1189,816,-66,-684,929,-1297,157,66,-330,-47,-99,305,-782,2290,25,7,510,99,718,2437,-136,-288,-2143,1657,-28,112,99,1071,-857,-536,373,-99,-1403,36,52,-308,1121,-209,-317,2,1223,-52,-1226,-146,341,437,-944,-148,-143,160,-52,51,-1201,-1347,529,52,286,-2,-131,71,-111,-119,-519,-380,-190,131,566,500,418,289,196,-135,-489,-125,131,189,422,208,-351,-131,4,-484,-198,2202,352,1352,1051,-2222,-2524,198,-538,1179,-2950,-2546,-276,277,-3116,3427,198,853,-269,-1922,3859,-198,-514,-2151,2244);
float tSine1[24];
float bunnySine1(vec3 p) {
    for (int i=0; i<24; i++)
        tSine1[i] = sin(.005*(
        dot(vec3(wSine1[i],wSine1[i+24],wSine1[i+48]),p)+float(wSine1[i+72])));
    float s=float(wSine1[120]);
    for (int i=0; i<24; i++)
        s += tSine1[i] * float(wSine1[i+96]);
    return s;
}

// Neural network with 1 hidden layer and tanh activation
// 121 weights, 64x64x64, MSE=0.0296, 0.80KB, 12ms
int wTanh1[121] = int[121](249,-87,-75,-1895,-710,86,-38,-84,-971,-422,-1127,86,672,417,-426,541,839,-17,376,-209,-98,-659,-1031,87,494,-66,8,772,-403,66,1094,-67,1174,657,-231,66,-1705,844,442,-33,776,552,-902,-804,571,-566,-104,66,-1196,-26,555,-1630,618,26,763,-25,320,109,-421,26,1050,-315,-54,-126,196,-957,-229,-1651,192,-1214,103,27,396,1739,-239,-955,-36,-1736,677,1681,833,68,-719,-1738,131,20,86,-307,258,-69,345,297,-151,-30,-351,-1732,-1007,1953,4273,794,1054,-1936,-1569,1615,-1060,-7396,2845,-1954,-549,1955,8365,4612,-1238,1034,5802,-451,7119,799,-1930,-1820,1835);
float tTanh1[24];
float bunnyTanh1(vec3 p) {
    for (int i=0; i<24; i++)
        tTanh1[i] = tanh(.005*(
        dot(vec3(wTanh1[i],wTanh1[i+24],wTanh1[i+48]),p)+float(wTanh1[i+72])));
    float s=float(wTanh1[120]);
    for (int i=0; i<24; i++)
        s += tTanh1[i] * float(wTanh1[i+96]);
    return s;
}

// Neural network with 1 hidden layer and ReLU activation
// 121 weights, 64x64x64, MSE=0.0376, 0.81KB, 4ms
// Fast to train/run; Missing ears
int wRelu1[121] = int[121](204,-9,-45,-310,-1031,-362,-702,-434,-184,690,912,701,-16,-240,-1165,-425,261,-1151,-431,-463,1256,-144,-275,-24,-1197,-16,26,-282,956,-722,377,-365,1514,-449,361,1458,-16,-369,95,442,-1223,-189,-547,1681,238,-684,-265,48,1046,-71,33,-834,94,-1043,-1217,-1054,184,-1379,-969,-19,-73,1262,-121,1136,78,47,-1073,369,391,-1280,-782,74,-190,-158,-185,714,550,18,999,1104,734,-763,-337,297,-249,-266,-552,-193,-841,-158,172,-344,-511,31,675,-246,843,-59,14,-819,680,301,1843,-978,-1022,1331,1020,1032,-178,382,1407,625,1982,-743,337,-1101,822,617,-760,-62,207);
float tRelu1[24];
float bunnyRelu1(vec3 p) {
    for (int i=0; i<24; i++)
        tRelu1[i] = .05*max(0.0,
        dot(vec3(wRelu1[i],wRelu1[i+24],wRelu1[i+48]),p)+float(wRelu1[i+72]));
    float s=float(wRelu1[120]);
    for (int i=0; i<24; i++)
        s += tRelu1[i] * float(wRelu1[i+96]);
    return s;
}

// Neural network with 2 hidden layers and sine activation
// 125 weights, 64x64x64, MSE=0.0174, 0.89KB, 5ms
// Optimizer: Adam + BFGS (C++)
int wSine2[125] = int[125](-300,118,8,69,-242,-131,-3,-55,-394,-97,119,-57,130,349,-169,77,240,118,18,-91,285,-119,-179,-394,-326,221,-65,-123,14,182,44,9,-83,121,-91,1,-11,-47,45,121,-172,2,85,-371,-10,-187,635,67,-618,706,852,209,256,-15,265,-1001,587,-1138,-113,1287,14,144,-586,-22,974,-466,-765,-837,-43,8,-11,45,-54,108,196,-129,3,-79,106,-3,98,-64,-11,63,-151,1,-190,843,-592,1044,77,-809,-91,10,22,-34,35,-8,245,-363,22,27,-108,-48,77,77,70,168,208,234,414,-211,-63,-143,334,-236,3353,3769,1515,-2644,-867,-1068,703,371,2447);
float uSine2[9], vSine2[8];
float bunnySine2(vec3 p) {
    for (int i=0; i<9; i++)
        uSine2[i] = sin(.01*(
        dot(vec3(wSine2[i],wSine2[i+9],wSine2[i+18]),p)+float(wSine2[i+27])));
    for (int i=0; i<8; i++) {
        float s=float(wSine2[108+i]);
        for (int j=0; j<9; j++) s += float(wSine2[8*j+i+36])*uSine2[j];
        vSine2[i]=sin(.01*s);
    }
    float s=float(wSine2[124]);
    for (int i=0; i<8; i++)
        s += vSine2[i] * float(wSine2[i+116]);
    return s;
}

// Neural network with 2 hidden layers and tanh activation
// 125 weights, 64x64x64, MSE=0.0266, 0.90KB, 11ms
int wTanh2[125] = int[125](-206,-11,-106,10,39,-143,336,338,-98,-153,-6,-234,364,371,11,1317,371,47,-61,61,-184,-159,-339,22,209,570,79,-70,-785,78,204,-77,-95,858,-18,7,-97,98,-265,30,157,548,-319,185,-64,-275,-138,-281,18,372,-35,-377,880,-334,40,-240,-152,-295,280,-93,578,-195,-194,1,-67,22,-357,-688,-2,93,-157,71,-1,976,-43,-119,531,-231,455,725,148,-868,-582,-388,137,-89,50,10,59,-607,-201,49,19,77,-25,-22,70,-141,-215,-95,299,-418,-237,-601,30,18,453,26,51,293,160,191,77,-199,10,249,-501,577,-1224,1931,1706,-1106,742,1337,996);
float uTanh2[9], vTanh2[8];
float bunnyTanh2(vec3 p) {
    for (int i=0; i<9; i++)
        uTanh2[i] = tanh(.01*(
        dot(vec3(wTanh2[i],wTanh2[i+9],wTanh2[i+18]),p)+float(wTanh2[i+27])));
    for (int i=0; i<8; i++) {
        float s=float(wTanh2[108+i]);
        for (int j=0; j<9; j++) s += float(wTanh2[8*j+i+36])*uTanh2[j];
        vTanh2[i]=tanh(.01*s);
    }
    float s=float(wTanh2[124]);
    for (int i=0; i<8; i++)
        s += vTanh2[i] * float(wTanh2[i+116]);
    return s;
}

// Radial basis with varying centers
// 121 weights, 64x64x64, MSE=0.0283, 0.93KB, 8ms
// Optimizer: simulated annealing + SciPy BFGS
// Hard to train due to local minima, high precision required for weights
int wRbW[31] = int[31](266511,-179095,118219,-271016,-597465,-403492,122896,-408065,-189574,953887,81225,-162921,-145270,999999,213652,332678,-245761,294460,-265686,-962891,-333099,374601,-23798,-503945,-140531,184730,519218,76838,304345,219342,34537);
int wRbC[90] = int[90](4580,-402,-153531,-1767,-5229,6643,1792,-115939,675085,2806,-5400,2522,-394,-4147,-4475,388341,1065308,-2477,373584,-3057,1626,-5049,-381175,-2413,830591,933,7790,-345,-1590,-1565,-776,3343,346187,-5830,3915,-4481,-3098,3277,282561,-22381,3624,2281,-4632,4511,5085,2455984,-223542,-1077,31846,4884,-10837,1756,-1013828,-409,-1357816,7157,-4981,-6962,5028,-6322,1811,1047,-611137,-2304,-3440,1081,4995,60318,1007427,9213,3255,-4062,4486,-2708,-4307,-374893,378826,4806,59503,-2415,2091,7918,-229580,5561,747136,-2112,1298,5075,-820,-2229);
float bunnyRb(vec3 p) {
    float s = float(wRbW[30]);
    for (int i=0; i<30; i++) {
        s += float(wRbW[i])*length(p-1e-4*vec3(wRbC[i],wRbC[i+30],wRbC[i+60]));
    }
    return s;
}


// Rendering

float map(vec3 p) {
    p = vec3(p.x,p.z,-p.y); // z-up to y-up
    if (MODEL==0) return bunnyPoly8(p);
    if (MODEL==1) return bunnyTrigC(p);
    if (MODEL==2) return bunnyTrigF(p);
    if (MODEL==3) return bunnySine1(p);
    if (MODEL==4) return bunnyTanh1(p);
    if (MODEL==5) return bunnyRelu1(p);
    if (MODEL==6) return bunnySine2(p);
    if (MODEL==7) return bunnyTanh2(p);
    if (MODEL==10) return bunnyRb(p);
}

#define iRes iResolution.xy
void mainImage(out vec4 fragColor, in vec2 fragCoord) {

    // set camera
    float rx = iMouse.z>0.?3.14*(iMouse.y/iRes.y)-1.57:0.2;
    float rz = iMouse.z>0.?-iMouse.x/iRes.x*4.0*3.14:0.5*iTime-2.0;
    rx += 1e-4, rz += 1e-4;  // prevent dividing by zero
    vec3 w = vec3(cos(rx)*vec2(cos(rz),sin(rz)),sin(rx));
    vec3 u = vec3(-sin(rz),cos(rz),0);
    vec3 v = cross(w,u);
    vec2 uv = (2.*fragCoord-iRes.xy)/min(iRes.x,iRes.y);
    vec3 ro = 8.0*w + 1.5*(uv.x*u+uv.y*v), rd = -w;

    // intersection with the unit box
    vec3 ird = 1. / rd;
    vec3 n = ird*(ro), k = abs(ird);
    vec3 d1 = -n - k, d2 = -n + k;
    float t0 = max(max(d1.x, d1.y), d1.z);
    float t1 = min(min(d2.x, d2.y), d2.z);

    // constant step raymarching
    float t = t0, dt = .01;
    float d_old = 0., d;
    for (t = t0; t < t1; t += dt) {
        d = map(ro+rd*t);
        if (d*d_old < 0.) {
            t -= dt * d/(d-d_old);  // linear interpolation
            break;
        }
        if (t>t0) dt=clamp(dt*abs(d/(d-d_old)),.01,.1); // adaptive step size
        d_old = d;
    }
    if (t>=t1) {
        fragColor = vec4(vec3(t0>t1?0.0:0.1), 1);
        return;
    }

    // shading
    vec3 ld = normalize(w+0.5*u+0.1*v), p=ro+rd*t;
    mat3 kd = mat3(p,p,p)-mat3(.01);
    n = normalize(map(p)-vec3(map(kd[0]),map(kd[1]),map(kd[2])));
    vec3 c = vec3(0.2+0.05*n.y+0.6*max(dot(n,ld),0.0));
    vec3 g = abs(sin(31.4*(ro+rd*t))/sqrt(1.-n*n));
    c *= mix(0.8, 1.0, clamp(-0.5+4.0*min(min(g.x,g.y),g.z),0.,1.));
    fragColor = vec4(c, 1.0);
}
