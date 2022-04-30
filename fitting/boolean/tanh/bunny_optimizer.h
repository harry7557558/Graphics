#pragma once

#include <cstdio>
#include <cmath>
#include <functional>
#include <algorithm> // max/min
#include <cstdint>
#include <cstring> // memcpy
#include <vector>

using std::abs;
using std::max;
using std::min;


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
    const double* x, const vec3* nul_1, const double* nul_2,
    double* val, double* grad
) {
    *val = 0.0;
    // for (int i = 0; i < 128; i += 1) {
    //     *val += x[i] * x[i];
    //     grad[i] = 2.0 * x[i];
    // }
    for (int i = 0; i < 128; i += 2) {
        double _1_xi = 1.0 - x[i];
        double _x_xi2 = x[i + 1] - x[i] * x[i];
        *val += _1_xi * _1_xi + 100.0 * _x_xi2 * _x_xi2;
        grad[i] = -2.0 * _1_xi - 400.0 * x[i] * _x_xi2;
        grad[i + 1] = 200.0 * _x_xi2;
    }
}


// Adam algorithm
void minimizeAdam(
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
            int n_batch = min(ndata - batch, batch_size);
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


// BFGS from SciPy

// Takes twice as much evaluations to converge on the Rosenbrock function compared to SciPy,
// which is very likely a bug but possibly due to floating point error.

// calculates sqrt(xᵀx)
double vecnorm(int ndim, const double* x) {
    double s2 = 0.0;
    for (int i = 0; i < ndim; i++) s2 += x[i] * x[i];
    return sqrt(s2);
}
// calculates uᵀv
double vecdot(int ndim, const double* u, const double* v) {
    double s = 0.0;
    for (int i = 0; i < ndim; i++) s += u[i] * v[i];
    return s;
}
// calculates uᵀmv
double vecmatvecdot(int ndim, const double* u, const double* m, const double* v) {
    double s = 0.0;
    for (int i = 0; i < ndim; i++) for (int j = 0; j < ndim; j++) {
        s += u[i] * m[i * ndim + j] * v[j];
    }
    return s;
}
// calculates res=mv
void matvecdot(int ndim, const double* m, const double* v, double* res) {
    for (int i = 0; i < ndim; i++) {
        res[i] = 0.0;
        for (int j = 0; j < ndim; j++) res[i] += m[i * ndim + j] * v[j];
    }
}


/*

Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


// https://github.com/scipy/scipy/blob/main/scipy/optimize/minpack2/dcstep.f
void dcstep(double& stx, double& fx, double& dx, double& sty, double& fy, double& dy, double& stp, double fp, double dp, bool& brackt, double stpmin, double stpmax) {

    double sgnd = dp * (dx / abs(dx));
    double stpf = stp, stpc, stpq;

    if (fp > fx) {
        double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        double s = max(abs(theta), max(abs(dx), abs(dp)));
        double gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp < stx) gamma = -gamma;
        double p = (gamma - dx) + theta;
        double q = ((gamma - dx) + gamma) + dp;
        double r = p / q;
        stpc = stx + r * (stp - stx);
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
        if (abs(stpc - stx) < abs(stpq - stx)) stpf = stpc;
        else stpf = stpc + (stpq - stpc) / 2.0;
        brackt = true;
    }

    else if (sgnd < 0.0) {
        double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        double s = max(abs(theta), max(abs(dx), abs(dp)));
        double gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp > stx) gamma = -gamma;
        double p = (gamma - dp) + theta;
        double q = ((gamma - dp) + gamma) + dx;
        double r = p / q;
        stpc = stp + r * (stx - stp);
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (abs(stpc - stp) > abs(stpq - stp)) stpf = stpc;
        else stpf = stpq;
        brackt = true;
    }

    else if (abs(dp) < abs(dx)) {
        double theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
        double s = max(abs(theta), max(abs(dx), abs(dp)));
        double gamma = s * sqrt(max(0.0, (theta / s) * (theta / s) - (dx / s) * (dp / s)));
        if (stp > stx) gamma = -gamma;
        double p = (gamma - dp) + theta;
        double q = (gamma + (dx - dp)) + gamma;
        double r = p / q;
        if (r < 0.0 && gamma != 0.0) stpc = stp + r * (stx - stp);
        else if (stp > stx) stpc = stpmax;
        else stpc = stpmin;
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (brackt) {
            if (abs(stpc - stp) < abs(stpq - stp)) stpf = stpc;
            else stpf = stpq;
            if (stp > stx) stpf = min(stp + 0.66 * (sty - stp), stpf);
            else stpf = max(stp + 0.66 * (sty - stp), stpf);
        }
        else {
            if (abs(stpc - stp) > abs(stpq - stp)) stpf = stpc;
            else stpf = stpq;
            stpf = max(stpmin, min(stpmax, stpf));
        }
    }

    else {
        if (brackt) {
            double theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
            double s = max(abs(theta), max(abs(dy), abs(dp)));
            double gamma = s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
            if (stp > sty) gamma = -gamma;
            double p = (gamma - dp) + theta;
            double q = ((gamma - dp) + gamma) + dy;
            double r = p / q;
            stpc = stp + r * (sty - stp);
            stpf = stpc;
        }
        else if (stp > stx) stpf = stpmax;
        else stpf = stpmin;
    }

    if (fp > fx) {
        sty = stp, fy = fp, dy = dp;
    }
    else {
        if (sgnd < 0.0) {
            sty = stx, fy = fx, dy = dx;
        }
        stx = stp, fx = fp, dx = dp;
    }
    stp = stpf;
}

// https://github.com/scipy/scipy/blob/main/scipy/optimize/minpack2/dcsrch.f
vec3 scalarSearchWolfe1(
    std::function<void(double x, double* val, double* grad)> fun, double f0, double old_f0, double g0,
    double ftol = 1e-4, double gtol = 0.9,
    double stpmax = 1e100, double stpmin = 1e-100, double xtol = 1e-14
) {
    double stp = 1.0;
    if (g0 != 0.0) {
        stp = min(1.0, 1.01 * 2.0 * (f0 - old_f0) / g0);
        if (stp <= 0.) stp = 1.0;
    }

    double f = f0, g = g0;

    if (stp < stpmin || stp > stpmax || g >= 0.0) {
        // bad start
        stp = -1.0;
        return vec3{ stp, f, f0 };
    }

    bool brackt = false;
    int stage = 1;
    double gtest = ftol * g0;
    double width = stpmax - stpmin;
    double width1 = 2.0 * width;

    const double xtrapl = 1.1, xtrapu = 4.0;
    double stx = 0.0, sty = 0.0;
    double fx = f0, gx = g0, fy = f0, gy = g0;
    double stmin = 0.0;
    double stmax = stp + xtrapu * stp;

    fun(stp, &f, &g);

    const int maxiter = 100;
    for (int i = 0; i < maxiter; i++) {

        double ftest = f0 + stp * gtest;
        if (stage == 1 && f <= ftest && g >= 0.0)
            stage = 2;

        if ((brackt && (stp <= stmin || stp >= stmax)) ||
            (brackt && stmax - stmin <= xtol * stmax) ||
            (stp == stpmax && f <= ftest && g <= gtest) ||
            (stp == stpmin && (f > ftest || g >= gtest))) {
            // warning
            return vec3{ -1.0, f, f0 };
        }

        if (f <= ftest && abs(g) <= gtol * (-g0)) {
            // success
            return vec3{ stp, f, f0 };
        }

        if (stage == 1 && f <= fx && f >= ftest) {

            double fm = f - stp * gtest;
            double fxm = fx - stx * gtest;
            double fym = fy - sty * gtest;
            double gm = g - gtest;
            double gxm = gx - gtest;
            double gym = gy - gtest;

            dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);

            fx = fxm + stx * gtest;
            fy = fym + sty * gtest;
            gx = gxm + gtest;
            gy = gym + gtest;
        }
        else {
            dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
        }
        // printf("%lf %lf\n", stx, sty);
        // exit(0);

        if (brackt) {
            if (abs(sty - stx) >= 0.66 * width1)
                stp = stx + 0.5 * (sty - stx);
            width1 = width;
            width = abs(sty - stx);
        }

        if (brackt) {
            stmin = min(stx, sty);
            stmax = max(stx, sty);
        }
        else {
            stmin = stp + xtrapl * (stp - stx);
            stmax = stp + xtrapu * (stp - stx);
        }

        stp = min(max(stp, stpmin), stpmax);

        if (brackt && (stp <= stmin || stp >= stmax) || (brackt && stmax - stmin <= xtol * stmax))
            stp = stx;

        fun(stp, &f, &g);
    }

    // maxiter exceeded
    return vec3{ -1.0, f, f0 };
}

double lineSearchWolfe1(
    const int ndim,
    std::function<void(const double* x, double* val, double* grad)> fun,
    const double* xk, const double* pk, const double* gfk,
    double* fval, double* old_fval, double* gval
) {
    // printf("xk/pk %.12lf %.12lf\n", vecnorm(ndim, xk), vecnorm(ndim, pk));
    std::memcpy(gval, gfk, sizeof(double) * ndim);

    double* x_param = new double[ndim];
    auto phi = [&](double s, double* val, double* grad) {
        for (int i = 0; i < ndim; i++) x_param[i] = xk[i] + s * pk[i];
        // printf("s/x_param/gval %.12lf %.12lf %.12lf\n", s, vecnorm(ndim, x_param), vecnorm(ndim, gval));
        fun(x_param, val, gval);
        // printf("gval/grad %.12lf\n", vecnorm(ndim, gval));
        *grad = vecdot(ndim, gval, pk);
    };

    double derphi0 = vecdot(ndim, gfk, pk);

    vec3 res = scalarSearchWolfe1(phi, *fval, *old_fval, derphi0);
    delete x_param;
    *fval = res.y;
    *old_fval = res.z;
    double stp = res.x;
    return stp;
}

// https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm#Algorithm
void minimizeBfgs(
    int ndim,
    std::function<void(const double* x, double* y, double* grad)> fun,
    double* x, double gtol = 1e-5, int maxiter = -1
) {
    if (maxiter <= 0)
        maxiter = ndim * 200;

    double old_fval;
    double* gfk = new double[ndim];
    fun(x, &old_fval, gfk);

    int k = 0;
    double* Hk = new double[ndim * ndim];
    for (int i = 0; i < ndim; i++) for (int j = 0; j < ndim; j++)
        Hk[i * ndim + j] = (i == j) ? 1.0 : 0.0;

    // Sets the initial step guess to dx ~ 1
    double old_old_fval = old_fval + vecnorm(ndim, gfk) / 2.0;

    double* xk = new double[ndim];
    std::memcpy(xk, x, sizeof(double) * ndim);

    double* pk = new double[ndim];
    double* gfkp1 = new double[ndim];
    double* xkp1 = new double[ndim];
    double* sk = new double[ndim];
    double* yk = new double[ndim];
    double* m1 = new double[ndim * ndim];
    double* m2 = new double[ndim * ndim];
    double* hkyk = new double[ndim];

    int warnflag = 0;
    double gnorm = vecnorm(ndim, gfk);
    while ((gnorm > gtol) && (k < maxiter)) {
        // printf("Hk %lf\n", vecnorm(ndim*ndim, Hk));

        for (int i = 0; i < ndim; i++) {
            pk[i] = 0.0;
            for (int j = 0; j < ndim; j++) pk[i] -= Hk[i * ndim + j] * gfk[j];
        }

        // line search
        double alpha_k = lineSearchWolfe1(ndim, fun, xk, pk, gfk, &old_fval, &old_old_fval, gfkp1);
        if (alpha_k < 0.0) {
            warnflag = 2;
            break;
        }
        if (std::isnan(old_fval)) {
            warnflag = 2;
            break;
        }
        for (int i = 0; i < ndim; i++) xkp1[i] = xk[i] + alpha_k * pk[i];
        for (int i = 0; i < ndim; i++) sk[i] = xkp1[i] - xk[i];
        std::memcpy(xk, xkp1, sizeof(double) * ndim);

        for (int i = 0; i < ndim; i++) yk[i] = gfkp1[i] - gfk[i];
        std::memcpy(gfk, gfkp1, sizeof(double) * ndim);
        k += 1;
        double gnorm = vecnorm(ndim, gfk);
        if (gnorm <= gtol) break;

        if (1) {
            printf("BFGS %d %lf %lf\n", k, old_fval, gnorm);
        }

        // update matrix
        double skyk = vecdot(ndim, sk, yk);
        if (skyk == 0.0) skyk = 1e-3;
        double w1 = (vecdot(ndim, sk, yk) + vecmatvecdot(ndim, yk, Hk, yk)) / (skyk * skyk);
        for (int i = 0; i < ndim; i++) for (int j = 0; j < ndim; j++)
            m1[i * ndim + j] = sk[i] * sk[j];
        double w2 = vecdot(ndim, sk, yk);
        w2 = (w2 == 0.0 ? -1e3 : -1.0 / w2);
        matvecdot(ndim, Hk, yk, hkyk);
        for (int i = 0; i < ndim; i++) for (int j = 0; j < ndim; j++)
            m2[i * ndim + j] = hkyk[i] * sk[j] + sk[i] * hkyk[j];
        for (int i = 0; i < ndim * ndim; i++)
            Hk[i] += m1[i] * w1 + m2[i] * w2;
    }

    double fval = old_fval;

    if (warnflag == 2) printf("Precision loss\n");
    else if (k >= maxiter) printf("Maximum number of iterations exceeded\n");
    else if (std::isnan(gnorm) || std::isnan(fval) || std::isnan(vecnorm(ndim, xk))) printf("NAN encountered\n");
    else printf("BFGS optimization success\n");

    std::memcpy(x, xk, sizeof(double) * ndim);

    delete Hk;
    delete xk; delete pk;
    delete gfkp1; delete xkp1;
    delete sk; delete yk;
    delete m1; delete m2; delete hkyk;
}

void minimizeBfgsLoss(
    int ndim, int ndata,
    std::function<void(int ndata, const double* w, const vec3* x, const double* y, double* val, double* grad)> lossfun,
    double* w, const vec3* x_, const double* y_,
    int maxiter, double gtol
) {
    int nfev = 0;
    auto lossfun_wrap = [&](const double* x, double* val, double* grad) {
        nfev += 1;
        lossfun(ndata, x, x_, y_, val, grad);
    };
    // printf("checkGrad %lf\n", checkGrad(ndim, lossfun_wrap, w, 1e-6));
    minimizeBfgs(ndim, lossfun_wrap, w, gtol, maxiter);
    printf("nfev: %d\n", nfev);
}