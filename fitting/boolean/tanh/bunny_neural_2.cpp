// Neural network with 2 hidden layers
// Check line 24 for different activation functions.

#pragma GCC optimize "Ofast"
#pragma GCC optimize "unroll-loops"
#pragma GCC target "sse,sse2,sse3,sse4,abm,avx,mmx,popcnt"


#include "bunny_optimizer.h"

#include <chrono>


// regression model

#define N_HIDDEN_1 9
#define N_HIDDEN_2 8
#define WN_INPUT (4*N_HIDDEN_1)
#define WN_HIDDEN_1 ((N_HIDDEN_1+1)*N_HIDDEN_2)
#define WN_HIDDEN_2 (N_HIDDEN_2+1)
#define N_WEIGHTS (WN_INPUT+WN_HIDDEN_1+WN_HIDDEN_2)

// activation functions
#define fr_dfr(r) fr=sin(r), dfr=cos(r)
//#define fr_dfr(r) fr=tanh(r), dfr=1.0/(cosh(r)*cosh(r))


void model(const double* w, vec3 x, double* y, double* y_grad) {
    // layer 1
    double f1[N_HIDDEN_1 + 1];  // activated output of first layer
    double g1[WN_INPUT];  // gradient of activated output to weights
    for (int i = 0; i < N_HIDDEN_1; i++) {
        double r1 = w[i + 3 * N_HIDDEN_1]
            + w[i + 0 * N_HIDDEN_1] * x.x
            + w[i + 1 * N_HIDDEN_1] * x.y
            + w[i + 2 * N_HIDDEN_1] * x.z;  // unactivated output
        double fr_dfr(r1);
        f1[i] = fr;  // activated output
        g1[i + 0 * N_HIDDEN_1] = x.x * dfr;  // gradient to weights
        g1[i + 1 * N_HIDDEN_1] = x.y * dfr;
        g1[i + 2 * N_HIDDEN_1] = x.z * dfr;
        g1[i + 3 * N_HIDDEN_1] = dfr;  // gradient to bias
    }
    f1[N_HIDDEN_1] = 1.0;  // pad 1
    // layer 2
    double f2[N_HIDDEN_2];  // activated output of second layer
    double g2[WN_HIDDEN_1];  // gradient to second layer weights
    for (int i = 0; i < N_HIDDEN_2; i++) {
        double r2 = w[WN_INPUT + N_HIDDEN_1 * N_HIDDEN_2 + i];
        for (int j = 0; j < N_HIDDEN_1; j++)
            r2 += f1[j] * w[WN_INPUT + j * N_HIDDEN_2 + i];  // unactivated output
        double fr_dfr(r2);
        f2[i] = fr;  // activated output
        for (int j = 0; j <= N_HIDDEN_1; j++) {  // gradient to weights and bias
            // g2[j * N_HIDDEN_2 + i] = f1[j] * dfr;
            g2[j * N_HIDDEN_2 + i] = dfr;  // multiply by f1[j] later
        }
    }
    // layer 3
    double f3;
    double g3[WN_HIDDEN_2];
    {
        f3 = w[WN_INPUT + WN_HIDDEN_1 + N_HIDDEN_2];
        for (int i = 0; i < N_HIDDEN_2; i++)  // output (no activation)
            f3 += w[WN_INPUT + WN_HIDDEN_1 + i] * f2[i];
        for (int i = 0; i < N_HIDDEN_2; i++)
            g3[i] = f2[i];  // gradient to weights
        g3[N_HIDDEN_2] = 1.0;  // derivative to bias
    }
    // backpropagation layer 2
    for (int i = 0; i < N_HIDDEN_2; i++) {
        g2[N_HIDDEN_1 * N_HIDDEN_2 + i] *= w[WN_INPUT + WN_HIDDEN_1 + i];  // gradient to bias
        for (int j = 0; j < N_HIDDEN_1; j++)
            g2[j * N_HIDDEN_2 + i] *= w[WN_INPUT + WN_HIDDEN_1 + i];  // gradient to weights
    }
    // backpropagation layer 1
    for (int i = 0; i < N_HIDDEN_1; i++) {
        for (int u = 0; u < 4; u++) { // u<3: weight; u=3: bias
            double s = 0.0;
            for (int j = 0; j < N_HIDDEN_2; j++)
                s += w[WN_INPUT + i * N_HIDDEN_2 + j] * g2[i * N_HIDDEN_2 + j];
            g1[N_HIDDEN_1 * u + i] *= s;
        }
    }
    // output
    *y = f3;  // value
    for (int i = 0; i < WN_INPUT; i++) y_grad[i] = 0.0;
    for (int i = 0; i < WN_INPUT; i++)
        y_grad[i] = g1[i];  // layer 1
    for (int i = 0; i < WN_HIDDEN_1; i++)
        y_grad[WN_INPUT + i] = g2[i] * f1[i / N_HIDDEN_2];  // layer 2
    for (int i = 0; i < WN_HIDDEN_2; i++)
        y_grad[WN_INPUT + WN_HIDDEN_1 + i] = g3[i];  // layer 3
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
    printf("%d weights\n", N_WEIGHTS);

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
    // return;

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
