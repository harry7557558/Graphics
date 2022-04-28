// trig_3d_bunny.py
// See how fast is C++ compared to Python

#pragma GCC optimize "Ofast"
#pragma GCC optimize "unroll-loops"
#pragma GCC target "sse,sse2,sse3,sse4,abm,avx,mmx,popcnt"


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

void evaluateWeights(int ndata, const double *w, vec3 *x, double *y) {
    double loss;
    double *grad = new double[N_WEIGHTS];
    lossMse(ndata, w, x, y, &loss, grad);
    double gradnorm = 0.0;
    for (int i = 0; i < N_WEIGHTS; i++) gradnorm += grad[i] * grad[i];
    gradnorm = sqrt(gradnorm);
    printf("loss=%lf grad=%lf\n", loss, gradnorm);
}

int main() {

    auto t0 = std::chrono::high_resolution_clock::now();

    // load data
    std::vector<vec3> x;
    std::vector<double> y;
    load_bunny(32, x, y);
    int ndata = (int)y.size();

    // weights
    // double* w = new double[N_WEIGHTS];
    // for (int i = 0; i < N_WEIGHTS; i++) {
    //     w[i] = -1.0 + 2.0 * randf();
    // }
    double w[N_WEIGHTS] = {-5.2351,-0.0737,3.1727,0.0101,1.3320,0.5287,1.5776,0.0378,2.1991,-0.0804,0.9363,2.0486,1.1747,1.1929,-0.1597,-1.5171,2.6011,-0.5434,-4.9482,1.1300,-1.7470,-0.4487,0.5385,-0.5625,-5.7769,-0.8215,3.1334,-2.1083,-0.6893,-0.0322,0.8653,-0.6590,-2.0678,0.0105,-1.2487,0.0016,1.5029,0.5848,2.2459,0.0361,0.0158,-0.2410,4.4085,4.9243,-0.7592,1.7660,-2.0783,-3.9629,-0.0000,0.7201,-1.8468,-5.5959,1.1026,0.4416,-1.6238,0.0636,-1.7207,0.0712,-6.1579,-1.4662,-1.2345,0.0890,-0.7839,0.1343,-1.1311,-2.2778,1.9943,0.0002,-3.3509,-0.0000,-2.7261,0.3878,0.0000,-2.9346,-1.5908,3.5336,1.2413,-4.7962,-0.0256,-1.5137,-0.0000,3.3537,-4.5305,2.8845,-1.6286,0.3168,0.3094,0.4009,-2.0512,1.5141,3.7879,2.9739,-0.9523,2.6657,0.8582,-4.4274,4.2945,-1.0409,2.2197,1.1295,-1.3909,0.6412,-3.4199,-0.3823,-0.0000,2.4247,6.9596,0.1848,-2.1125,3.1640,0.0565,3.2177,-0.0000,0.0000,-3.7274,5.6154,5.6841,7.4628,0.2871,-0.8150,2.6085,4.8001,-2.9622,0.5242,0.7536,-0.2827,0.7780,0.2387};
    // double e = checkGrad(N_WEIGHTS, [=](const double* w, double* val, double* grad) {
    //     // rosenbrock(ndata, w, nullptr, nullptr, val, grad);
    //     // model(w, vec3{0., 0., 0.}, val, grad);
    //     // lossLnp(ndata, w, &x[0], &y[0], val, grad);
    //     lossMse(ndata, w, &x[0], &y[0], val, grad);
    //     }, w);
    // printf("checkgrad=%lf\n", e); return 0;
    evaluateWeights(ndata, w, &x[0], &y[0]);

    // optimize
    // optimizeAdam(N_WEIGHTS, ndata, lossLnp, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 400, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);
    
    optimizeAdam(N_WEIGHTS, ndata, lossMse, w, &x[0], &y[0], 1024, 0.01, 0.9, 0.999, 400, 1e-5);
    evaluateWeights(ndata, w, &x[0], &y[0]);

    // print weights
    for (int i = 0; i < N_WEIGHTS; i++) printf("%.4lf,", w[i]);
    printf("\n");

    // timer
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("%.2lf secs elapsed.\n", dt);

    return 0;
}
