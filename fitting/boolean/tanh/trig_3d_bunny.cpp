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

    // load data
    std::vector<vec3> x;
    std::vector<double> y;
    load_bunny(16, x, y);
    int ndata = (int)y.size();

    // weights
    double* w = new double[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) {
        w[i] = -1.0 + 2.0 * randf();
    }
    // double e = checkGrad(N_WEIGHTS, [=](const double* w, double* val, double* grad) {
    //     // rosenbrock(ndata, w, nullptr, nullptr, val, grad);
    //     // model(w, vec3{0., 0., 0.}, val, grad);
    //     // lossLnp(ndata, w, &x[0], &y[0], val, grad);
    //     lossMse(ndata, w, &x[0], &y[0], val, grad);
    //     }, w);
    // printf("checkgrad=%lf\n", e); return 0;
    evaluateWeights(ndata, w, &x[0], &y[0]);

    // optimize
    minimizeAdam(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 400, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);
    minimizeBfgsLoss(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 1000, 1e-3);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    // minimizeAdam(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 400, 1e-5);
    // evaluateWeights(ndata, w, &x[0], &y[0]);

    // print weights
    for (int i = 0; i < N_WEIGHTS; i++) printf("%.4lf,", w[i]);
    printf("\n");

}

// debug
void optimizeRosenbrock() {
    double w[N_WEIGHTS];
    for (int i = 0; i < N_WEIGHTS; i++) w[i] = -1.0;
    double val; double grad[N_WEIGHTS];
    rosenbrock(0, w, nullptr, nullptr, &val, grad);
    printf("%lf\n", val);

    minimizeBfgsLoss(N_WEIGHTS, 0, rosenbrock, w, nullptr, nullptr, 1000, 1e-5);
}

// timing
int main() {

    auto t0 = std::chrono::high_resolution_clock::now();

    optimizeBunny();
    // optimizeRosenbrock();

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("%.2lf secs elapsed.\n", dt);

    return 0;
}
