// Neural network with 1 hidden layer
// Check line 31 for different activation functions.

#pragma GCC optimize "Ofast"
#pragma GCC optimize "unroll-loops"
#pragma GCC target "sse,sse2,sse3,sse4,abm,avx,mmx,popcnt"


#include "bunny_optimizer.h"

#include <chrono>


// regression model

#define N_HIDDEN 24
#define WN_INPUT (4*N_HIDDEN)
#define WN_HIDDEN (N_HIDDEN+1)
#define N_WEIGHTS (WN_INPUT+WN_HIDDEN)


void model(const double* w, vec3 x, double* y, double* y_grad) {
    // layer 1
    double l1[N_HIDDEN];  // activated output of first layer
    double g1[4 * N_HIDDEN];  // gradient of activated output to weights
    for (int i = 0; i < N_HIDDEN; i++) {
        double r1 = w[i + 3 * N_HIDDEN]
            + w[i + 0 * N_HIDDEN] * x.x
            + w[i + 1 * N_HIDDEN] * x.y
            + w[i + 2 * N_HIDDEN] * x.z;  // unactivated output
        // different activation functions
        double fr = sin(r1), dfr = cos(r1);
        // double fr = tanh(r1), dfr = 1.0/(cosh(r1)*cosh(r1));
        // double fr = max(r1, 0.0), dfr = r1 > 0.0 ? 1.0 : 0.0;
        l1[i] = fr;  // activated output
        g1[i + 0 * N_HIDDEN] = x.x * dfr;  // derivative to weights
        g1[i + 1 * N_HIDDEN] = x.y * dfr;
        g1[i + 2 * N_HIDDEN] = x.z * dfr;
        g1[i + 3 * N_HIDDEN] = dfr;  // derivative to bias
    }
    // layer 2
    double r2 = w[WN_INPUT + N_HIDDEN];
    for (int i = 0; i < N_HIDDEN; i++)
        r2 += w[WN_INPUT + i] * l1[i];
    *y = r2;  // output (no activation)
    for (int i = 0; i < 4 * N_HIDDEN; i++)  // gradient to layer 1 weights
        y_grad[i] = w[WN_INPUT + i % N_HIDDEN] * g1[i];
    for (int i = 0; i < N_HIDDEN; i++)
        y_grad[WN_INPUT + i] = l1[i];  // gradient to weights
    y_grad[WN_INPUT + N_HIDDEN] = 1.0;  // gradient to bias
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
        // model(w, x[0], val, grad);
        // lossLnp(ndata, w, &x[0], &y[0], val, grad);
        lossMse(ndata, w, &x[0], &y[0], val, grad);
        }, w, 1e-6, false));

    // optimize
    ndata = load_bunny(64, x, y);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeAdam(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 100, 1e-3);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeAdam(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 400, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    minimizeLossBfgs(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1000, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);
}

// timing
int main(int argc, char* argv[]) {

    auto t0 = std::chrono::high_resolution_clock::now();

    optimizeBunny();

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("%.2lf secs elapsed.\n", dt);

    return 0;
}
