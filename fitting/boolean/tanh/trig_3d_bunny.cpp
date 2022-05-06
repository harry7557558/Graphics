// trig_3d_bunny.py
// See how fast is C++ compared to Python

#if 1
#pragma GCC optimize "Ofast"
#pragma GCC optimize "unroll-loops"
#pragma GCC target "sse,sse2,sse3,sse4,abm,avx,mmx,popcnt"
#else
#pragma GCC optimize "O0"
#endif


#include "bunny_optimizer.h"

#include <chrono>


// regression model

#define N_WAVES 32
#define N_WEIGHTS (N_WAVES*4)

void model(const double* w, vec3 p, double* y, double* y_grad) {
    *y = 0.0;
    for (int i = 0; i < N_WEIGHTS; i++) y_grad[i] = 0.0;
    for (int wi = 0; wi < N_WAVES; wi++) {
        int ti = wi % 8;
        double amp = w[wi];
        double fx = w[wi + N_WAVES], fy = w[wi + 2 * N_WAVES], fz = w[wi + 3 * N_WAVES];
        double bx = (((ti >> 0) & 1) == 0) ? cos(fx * p.x) : sin(fx * p.x);
        double by = (((ti >> 1) & 1) == 0) ? cos(fy * p.y) : sin(fy * p.y);
        double bz = (((ti >> 2) & 1) == 0) ? cos(fz * p.z) : sin(fz * p.z);
        double dbx = p.x * ((((ti >> 0) & 1) == 0) ? -sin(fx * p.x) : cos(fx * p.x));
        double dby = p.y * ((((ti >> 1) & 1) == 0) ? -sin(fy * p.y) : cos(fy * p.y));
        double dbz = p.z * ((((ti >> 2) & 1) == 0) ? -sin(fz * p.z) : cos(fz * p.z));
        *y += amp * bx * by * bz;
        y_grad[wi] += bx * by * bz;
        y_grad[wi + 1 * N_WAVES] += amp * dbx * by * bz;
        y_grad[wi + 2 * N_WAVES] += amp * bx * dby * bz;
        y_grad[wi + 3 * N_WAVES] += amp * bx * by * dbz;
    }
}


// loss functions

void lossLnp(int ndata, const double* w, const vec3* x, const double* y, double* v, double* v_grad) {
    *v = 0.0;
    for (int i = 0; i < N_WEIGHTS; i++) v_grad[i] = 0.0;
    double f;
    double* f_grad = new double[N_WEIGHTS];
    for (int i = 0; i < ndata; i++) {
        model(w, x[i], &f, f_grad);
        double e2f = exp(2.0 * f);
        double e_2f = 1.0 / e2f;
        double v1 = (1.0 - y[i]) * log(1.0 + e2f);
        double v2 = (1.0 + y[i]) * log(1.0 + e_2f);
        double g1 = 2.0 * (1.0 - y[i]) * e2f / (1.0 + e2f);
        double g2 = -2.0 * (1.0 + y[i]) * e_2f / (1.0 + e_2f);
        *v += (v1 + v2) * (v1 + v2);
        double g = 2.0 * (v1 + v2) * (g1 + g2);
        for (int j = 0; j < N_WEIGHTS; j++) v_grad[j] += f_grad[j] * g;
    }
    *v /= double(ndata);
    for (int j = 0; j < N_WEIGHTS; j++) v_grad[j] /= double(ndata);
    delete f_grad;
}

void lossMse(int ndata, const double* w, const vec3* x, const double* y, double* v, double* v_grad) {
    *v = 0.0;
    for (int i = 0; i < N_WEIGHTS; i++) v_grad[i] = 0.0;
    double f;
    double* f_grad = new double[N_WEIGHTS];
    for (int i = 0; i < ndata; i++) {
        model(w, x[i], &f, f_grad);
        double d = tanh(f) - y[i];
        double c = cosh(f);
        *v += d * d;
        double g = 2.0 * d / (c * c);
        for (int j = 0; j < N_WEIGHTS; j++) v_grad[j] += f_grad[j] * g;
    }
    *v /= double(ndata);
    for (int j = 0; j < N_WEIGHTS; j++) v_grad[j] /= double(ndata);
    delete f_grad;
}

// evaluate the loss function with verbose
void evaluateWeights(int ndata, const double* w, vec3* x, double* y) {
    for (int i = 0; i < N_WEIGHTS; i++) printf("%.6lf,", w[i]); printf("\n");
    double loss;
    double* grad = new double[N_WEIGHTS];
    lossMse(ndata, w, x, y, &loss, grad);
    double gradnorm = 0.0;
    for (int i = 0; i < N_WEIGHTS; i++) gradnorm += grad[i] * grad[i];
    gradnorm = sqrt(gradnorm);
    printf("loss=%lf grad=%lf\n", loss, gradnorm);
}

// main function
void optimizeBunny() {

    // weights
    _IDUM = 1u;
    double* w = new double[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) {
        w[i] = -1.0 + 2.0 * randf();
    }

    // data
    std::vector<vec3> x;
    std::vector<double> y;
    int ndata = load_bunny(8, x, y);

    // check the correctness of gradient
    printf("checkgrad=%lf\n", checkGrad(N_WEIGHTS, [=](const double* w, double* val, double* grad) {
        // lossLnp(ndata, w, &x[0], &y[0], val, grad);
        lossMse(ndata, w, &x[0], &y[0], val, grad);
        }, w));

    // optimize
    printf("16x16x16\n");
    ndata = load_bunny(16, x, y);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeAdam(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 1000, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeLossBfgs(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 1000, 1e-4);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    printf("32x32x32\n");
    ndata = load_bunny(32, x, y);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeAdam(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 1000, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeLossBfgs(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1000, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    printf("64x64x64\n");
    ndata = load_bunny(64, x, y);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeLossBfgs(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 2000, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);
}

void optimizeBunnyAdamOnly() {

    // weights
    _IDUM = 0u;
    // double* w = new double[N_WEIGHTS];
    // for (int i = 0; i < N_WEIGHTS; i++) {
    //     w[i] = -1.0 + 2.0 * randf();
    // }
    double w[N_WEIGHTS] = { -10.673769,-1.746616,9.409255,-1.952873,6.022788,0.000000,1.900502,1.643552,7.353547,1.722111,-3.110132,7.553541,4.195577,3.717857,0.000000,0.000000,7.746588,1.687724,-7.258417,-2.299722,-1.676034,-3.531630,0.000997,4.619293,-21.388092,-4.502349,6.192209,-0.000046,1.825197,0.000002,-0.000000,-0.062024,-1.875513,9.130968,0.000000,4.494365,-1.350136,-0.000000,8.283769,6.197621,0.000000,3.473744,6.216631,3.382139,0.000000,-4.139510,-1.889807,0.000000,-0.000771,-4.409266,-2.480820,-4.657740,3.558014,0.969264,-2.017782,-3.405755,-1.654629,4.074049,-4.606696,-0.000046,-7.557604,0.000002,-1.861261,0.000118,-2.458794,-2.176493,3.394316,4.656420,-0.860821,-2.521390,-3.749725,4.244439,0.000000,0.000000,1.851033,2.663542,2.820149,1.313707,0.000000,0.012931,0.000000,-12.065561,-9.147179,6.597035,-4.811986,3.853493,-0.000000,-2.334199,-1.460897,2.730159,4.027670,0.000046,-1.027875,0.893062,-0.000000,-0.000118,5.948641,-0.000000,2.411905,7.657949,-3.148132,0.000016,-10.432479,-6.375811,-0.000000,-0.000000,4.433940,1.263490,-3.833447,3.620085,-0.000000,0.044760,0.000000,-3.482444,-2.801232,-0.000000,8.095999,-5.630478,0.000000,-3.214667,2.179076,4.404299,-2.161572,2.655599,3.823575,-0.000002,0.000462,-0.000118 };

    // data
    std::vector<vec3> x;
    std::vector<double> y;
    int ndata = load_bunny(64, x, y);

    // optimize
    evaluateWeights(ndata, w, &x[0], &y[0]);

    // minimizeAdam(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 256, 0.01, 0.9, 0.999, 20, 1e-2);
    // evaluateWeights(ndata, w, &x[0], &y[0]);

    int nit = 1;
    for (int i = 0; i < nit; i++) {
        double step_size = exp(log(0.01) + i * (log(0.01) - log(0.01)) / nit);
        // printf("Iter %d, ", i);
        minimizeAdam(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1024, step_size, 0.9, 0.999, 100, 1e-5);
    }
    evaluateWeights(ndata, w, &x[0], &y[0]);
}

// debug
void optimizeRosenbrock() {
    double w[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) w[i] = -1.0;
    double val; double grad[N_WEIGHTS];
    rosenbrock(0, w, nullptr, nullptr, &val, grad);
    // printf("%lf\n", val);
    minimizeLossBfgs(N_WEIGHTS, 0, rosenbrock, w, nullptr, nullptr, 1000, 1e-5);
}

// timing
int main(int argc, char* argv[]) {

    auto t0 = std::chrono::high_resolution_clock::now();

    // optimizeBunny();
    optimizeBunnyAdamOnly();
    // optimizeRosenbrock();

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("%.2lf secs elapsed.\n", dt);

    return 0;
}
