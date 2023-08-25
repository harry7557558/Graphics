#pragma once
#include <math.h>

float funCircle(float x, float y) {
    float v0=x*x, v1=y*y, v2=v0+v1, v3=v2-1.0f;
    return v3;
}

float funA6Heart(float x, float y) {
    float v0=x*x, v1=y*y, v2=v0+v1, v3=v2-1.0f, v4=v3*v3*v3, v5=2.0f*v0, v6=y*y*y, v7=v5*v6, v8=v4-v7;
    return v8;
}

float funRadicalHeart(float x, float y) {
    float v0=x*x, v1=1.3f*y, v2=fabs(x), v3=sqrt(v2), v4=v1-v3, v5=v4*v4, v6=v0+v5, v7=v6-1.0f;
    return v7;
}

float funTooth(float x, float y) {
    float v0=x*x*x, v1=2.0f*v0, v2=x-2.0f, v3=v1*v2, v4=2.0f*x, v5=v3+v4, v6=y*y*y, v7=y-2.0f, v8=v6*v7, v9=v5+v8;
    return v9;
}

float funQuatrefoil(float x, float y) {
    float v0=atan2(y,x), v1=6.0f*v0, v2=sin(v1), v3=4.0f*x, v4=v3*y, v5=v2-v4;
    return v5;
}

float funHyperbolicPlane(float x, float y) {
    float v0=4.0f*x, v1=x*x, v2=y*y, v3=v1+v2, v4=v0/v3, v5=sin(v4), v6=4.0f*y, v7=v6/v3, v8=sin(v7), v9=v5*v8;
    return v9;
}

float funRoundedSquare(float x, float y) {
    float v0=fabs(x), v1=fabs(y), v2=x*x*x, v3=y*y*y, v4=v2-v3, v5=fmax(fmax(v0,v1),v4), v6=v5-1.0f;
    return v6;
}

float funFlower1(float x, float y) {
    float v0=x*x, v1=y*y, v2=v0+v1, v3=sqrt(v2), v4=atan2(y,x), v5=5.0f*v4, v6=sin(v5), v7=asin(v6), v8=0.3f*v7, v9=0.7f+v8, v10=v3-v9;
    return v10;
}

float funFlower2(float x, float y) {
    float v0=5.0f/2.0f, v1=atan2(y,x), v2=v0*v1, v3=sin(v2), v4=v3*v3*v3*v3*v3*v3*v3*v3*v3*v3, v5=3.0f*v4, v6=7.0f-v5, v7=5.0f*v1, v8=sin(v7), v9=v8*v8*v8*v8*v8*v8*v8*v8*v8*v8, v10=5.0f*v9, v11=v6-v10, v12=x*x, v13=y*y, v14=v12+v13, v15=sqrt(v14), v16=6.0f*v15, v17=v11-v16;
    return v17;
}

float funSwirl(float x, float y) {
    float v0=x*x, v1=y*y, v2=v0+v1, v3=pow(v2,1.0f/4.0f), v4=15.0f*v3, v5=sin(v4), v6=x*v5, v7=cos(v4), v8=y*v7, v9=v6+v8, v10=0.75f*v2, v11=v9-v10;
    return v11;
}

float funStar6(float x, float y) {
    float v0=x*x, v1=y*y, v2=v0+v1, v3=atan2(x,y), v4=3.0f*v3, v5=sin(v4), v6=v5*v5, v7=10.0f*v6, v8=1.0f+v7, v9=v2*v8, v10=v9-2.0f;
    return v10;
}

float funEvil13(float x, float y) {
    float v0=13.0f/2.0f, v1=atan2(y,x), v2=v0*v1, v3=x*x, v4=y*y, v5=v3+v4, v6=sqrt(v5), v7=10.0f*v6, v8=sin(v7), v9=v2-v8, v10=sin(v9), v11=fabs(v10), v12=2.0f*v11, v13=5.0f-v12, v14=sqrt(v6), v15=4.0f*v14, v16=v13-v15;
    return v16;
}

float funAbsSpam(float x, float y) {
    float v0=fabs(x), v1=fabs(y), v2=v0+v1, v3=fabs(v2), v4=v0-v1, v5=fabs(v4), v6=2.0f*v5, v7=v3-v6, v8=y-x, v9=fabs(v8), v10=y+x, v11=fabs(v10), v12=v9+v11, v13=v12-0.8f, v14=fabs(v13), v15=v7+v14, v16=v15-0.4f, v17=fabs(v16), v18=v17-0.15f;
    return v18;
}

float funSwirls(float x, float y) {
    float v0=y+4.0f, v1=3.0f*v0, v2=x+3.0f, v3=3.0f*v2, v4=sin(v3), v5=v1*v4, v6=cos(v1), v7=v3*v6, v8=sqrt(v5*v5+v7*v7), v9=atan2(v5,v7), v10=v8-v9, v11=cos(v10);
    return v11;
}

float funPuzzlePieces(float x, float y) {
    float v0=6.0f*x, v1=sin(v0), v2=6.0f*y, v3=sin(v2), v4=v1+v3, v5=12.0f*x, v6=sin(v5), v7=cos(v2), v8=v6+v7, v9=12.0f*y, v10=sin(v9), v11=v8*v10, v12=v4-v11;
    return v12;
}

float funTangent(float x, float y) {
    float v0=2.0f*x, v1=tan(v0), v2=y-v1, v3=tan(y), v4=v2*v3;
    return v4;
}

float funEyes(float x, float y) {
    float v0=y+x, v1=v0+1.0f, v2=3.0f*v1, v3=y-x, v4=v3+1.0f, v5=3.0f*v4, v6=sin(v5), v7=v2*v6, v8=sin(v2), v9=v5*v8, v10=fmin(v7,v9), v11=sin(v10), v12=cos(v5), v13=v2*v12, v14=cos(v2), v15=v5*v14, v16=fmax(v13,v15), v17=cos(v16), v18=v11-v17, v19=2.0f*y, v20=3.0f-v19, v21=v20/9.0f, v22=x*x, v23=2.0f*v22, v24=y*y, v25=v23+v24, v26=v25/6.0f, v27=v26*v26*v26, v28=v21+v27, v29=v18-v28;
    return v29;
}

float funFractalSine(float x, float y) {
    float v0=sin(x), v1=sin(y), v2=v0*v1, v3=2.0f*x, v4=sin(v3), v5=2.0f*y, v6=sin(v5), v7=v4*v6, v8=v7/2.0f, v9=v2+v8, v10=4.0f*x, v11=sin(v10), v12=4.0f*y, v13=sin(v12), v14=v11*v13, v15=v14/3.0f, v16=v9+v15, v17=8.0f*x, v18=sin(v17), v19=8.0f*y, v20=sin(v19), v21=v18*v20, v22=v21/4.0f, v23=v16+v22, v24=17.0f*x, v25=sin(v24), v26=16.0f*y, v27=sin(v26), v28=v25*v27, v29=v28/5.0f, v30=v23+v29, v31=32.0f*x, v32=sin(v31), v33=32.0f*y, v34=sin(v33), v35=v32*v34, v36=v35/6.0f, v37=v30+v36;
    return v37;
}

float funMandelbrotSet(float x, float y) {
    float v0=1.0f/2.0f, v1=x-v0, v2=v1*v1, v3=y*y, v4=v2-v3, v5=v4+v1, v6=v5*v5, v7=2.0f*v1, v8=v7*y, v9=v8+y, v10=v9*v9, v11=v6-v10, v12=v11+v1, v13=v12*v12, v14=2.0f*v5, v15=v14*v9, v16=v15+y, v17=v16*v16, v18=v13-v17, v19=v18+v1, v20=v19*v19, v21=2.0f*v12, v22=v21*v16, v23=v22+y, v24=v23*v23, v25=v20-v24, v26=v25+v1, v27=v26*v26, v28=2.0f*v19, v29=v28*v23, v30=v29+y, v31=v30*v30, v32=v27-v31, v33=v32+v1, v34=v33*v33, v35=2.0f*v26, v36=v35*v30, v37=v36+y, v38=v37*v37, v39=v34-v38, v40=v39+v1, v41=v40*v40, v42=2.0f*v33, v43=v42*v37, v44=v43+y, v45=v44*v44, v46=v41-v45, v47=2.0f*v40, v48=v47*v44, v49=sqrt(v46*v46+v48*v48), v50=v49+1.0f, v51=log(v50)/log(2.0f), v52=log(v51)/log(2.0f), v53=sin(v52);
    return v53;
}