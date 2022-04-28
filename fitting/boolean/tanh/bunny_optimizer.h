#pragma once

#include <cstdio>
#include <cmath>
#include <functional>
#include <algorithm> // max/min
#include <cstdint>
#include <cstring> // memcpy
#include <vector>


// PRNG from Numerical Recipes
uint32_t _IDUM = 0u;
uint32_t randu() { return _IDUM = _IDUM * 1664525u + 1013904223u; }
double randf() { return (double)randu() / 4294967296.0; }


// vector3
struct vec3 {
    double x, y, z;
};


// Load the bunny model as data points
void load_bunny(int s, std::vector<vec3>& data_x, std::vector<double>& data_y) {
    // load voxels from raw file
    const int s0 = 128;
    uint8_t* data_raw = new uint8_t[s0 * s0 * s0];
    FILE* fp = fopen("../v_sdfbunny_128x128x128_uint8.raw", "rb");
    fread(data_raw, 1, s0 * s0 * s0, fp);
    fclose(fp);
    // convert to floats
    int vs = s0 / s;
    for (int zi = 0; zi < s; zi++) {
        double z = -1.0 + (zi + 0.5) * 2.0 / s;
        for (int yi = 0; yi < s; yi++) {
            double y = -1.0 + (yi + 0.5) * 2.0 / s;
            for (int xi = 0; xi < s; xi++) {
                double x = -1.0 + (xi + 0.5) * 2.0 / s;
                vec3 p{ y, x, z };
                double val = 0.0;
                for (int dz = 0; dz < vs; dz++) for (int dy = 0; dy < vs; dy++) for (int dx = 0; dx < vs; dx++) {
                    int i = ((xi * vs + dx) * s0 + (yi * vs + dy)) * s0 + (zi * vs + dz);
                    // int i = ((zi * vs + dz) * s0 + (yi * vs + dy)) * s0 + (xi * vs + dx);
                    val += 0.5 * 255 - double(data_raw[i]);
                }
                data_x.push_back(p);
                data_y.push_back(val > 0. ? 1.0 : -1.0);
            }
        }
    }
    delete data_raw;
}


// check if the analytical gradient is correct, returns error
double checkGrad(
    int ndim,
    std::function<void(const double* x, double* val, double* grad)> fun,
    double* x,
    double eps = 1e-6
) {
    double val;
    double* grad0 = new double[ndim];
    double* grad = new double[ndim];
    fun(x, &val, grad0);
    double err2 = 0.0;
    for (int i = 0; i < ndim; i++) {
        double val1, val0;
        double x0 = x[i];
        x[i] = x0 + eps;
        fun(x, &val1, grad);
        x[i] = x0 - eps;
        fun(x, &val0, grad);
        x[i] = x0;
        double dvdx = (val1 - val0) / (2.0 * eps);
        err2 += (dvdx - grad0[i]) * (dvdx - grad0[i]);
    }
    delete grad; delete grad0;
    return sqrt(err2);
}


// test function
void rosenbrock(
    int ndata,
    const double* x, const vec3 *nul_1, const double *nul_2,
    double* val, double* grad
) {
    *val = 0.0;
    for (int i = 0; i < 128; i += 2) {
        *val += pow(1.0 - x[i], 2.0) + 100.0 * pow(x[i + 1] - x[i] * x[i], 2.0);
        grad[i] = -2.0 * (1.0 - x[i]) - 400.0 * x[i] * (x[i + 1] - x[i] * x[i]);
        grad[i + 1] = 200.0 * (x[i + 1] - x[i] * x[i]);
    }
}


// Adam algorithm
void optimizeAdam(
    int ndim, int ndata,
    std::function<void(int ndata, const double* w, const vec3* x, const double* y, double* val, double* grad)> lossfun,
    double* w, const vec3* x_, const double* y_,
    int batch_size, double learning_step,
    double beta_1, double beta_2, int max_epoch, double gtol
) {

    // training data, shuffled at the beginning of each epoch
    vec3* x = new vec3[ndata];
    double* y = new double[ndata];
    std::memcpy(x, x_, ndata * sizeof(vec3));
    std::memcpy(y, y_, ndata * sizeof(double));

    // loss and gradient
    double loss_t, loss = 0.0;
    double* grad_t = new double[ndim];  // gradient in evaluation
    double* grad = new double[ndim];  // smoothed gradient
    double* grad2 = new double[ndim];  // smoothed squared gradient
    for (int i = 0; i < ndim; i++) {
        grad[i] = 0.0;
        grad2[i] = 0.0;
    }

    // epoches
    for (int epoch = 0; epoch < max_epoch; epoch++) {

        // shuffle training data
        for (int i = ndata - 1; i > 0; i--) {
            int j = randu() % (i + 1);
            std::swap(x[i], x[j]);
            std::swap(y[i], y[j]);
        }

        // batches
        for (int batch = 0; batch < ndata; batch += batch_size) {
            // evaluate function
            int n_batch = std::min(ndata - batch, batch_size);
            lossfun(n_batch, w, &x[batch], &y[batch], &loss_t, grad_t);
            // update
            for (int i = 0; i < ndim; i++) {
                loss = beta_1 * loss + (1.0 - beta_1) * loss_t;
                grad[i] = beta_1 * grad[i] + (1.0 - beta_1) * grad_t[i];
                grad2[i] = beta_2 * grad2[i] + (1.0 - beta_2) * grad_t[i] * grad_t[i];
                w[i] -= learning_step * grad[i] / (sqrt(grad2[i]) + 1e-8);
            }
        }

        // check
        double grad_norm = 0.0;
        for (int i = 0; i < ndim; i++) grad_norm += grad[i] * grad[i];
        grad_norm = sqrt(grad_norm);
        printf("Epoch %d, loss=%lf, grad=%lf\n", epoch, loss, grad_norm);
        if (grad_norm < gtol) break;
    }

    delete x; delete y;
    delete grad_t; delete grad; delete grad2;
}
