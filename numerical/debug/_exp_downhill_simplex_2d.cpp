#include <stdio.h>
#include <algorithm>
#include "numerical/geometry.h"

// visualization
void img_init(const char*);
void img_renderFrame(vec2, vec2, vec2, int);
void img_finish();

// function to minimize
int evalCount = 0;
double fun(vec2 p) {
	evalCount++;
	double x = p.x, y = p.y;
#define sq(x) ((x)*(x))
	//return x * x + y * y - 1.;  // 94 evals
	return sqrt(x*x + y * y) - 1.;  // 154 evals
	//return x * x*x*x + y * y*y*y - 1.;  // 63 evals
	//return 100 * sq(y - x * x) + sq(1 - x) - 1.;  // 195 evals (valley)
	//return sqrt(100 * sq(y - x * x) + sq(1 - x)) - 1.;
	//return pow(100 * sq(.5*y - .25*x * x) + sq(1 - .5*x), .25);  // 365 evals
	//return log(100 * sq(.5*y - .25*x * x) + sq(1 - .5*x) + 1.);  // 192 evals
	//return 100 * abs(y - x * x) + abs(1 - x);  // **FAIL**
	//return sq(x*x + y - 11) + sq(y*y + x - 7);  // 102 evals, local minimum
	//return x * x*x*x - 4 * x*x*y + y * y*y*y;  // 93 evals
	//return 100 * sq(sqrt(x*x + y * y) - 2.) - (x - y);  // 257 evals (valley)
	//return pow(100 * sq(sqrt(x*x + y * y) - 2.) - (x - y) + 2.833427, .25);  // 355 evals
	//return sq(1.5 - x + x * y) + sq(2.25 - x + x * y*y) + sq(2.625 - x + x * y*y*y) + 1e6 * max(-min(x, y), 0.);  // 102 evals
	//return sin(y)*exp(sq(1 - cos(x))) + cos(x)*exp(sq(1 - sin(y))) + sq(x - y) + sq(x + y);  // 101 evals, local minimum
	//return abs(x) + abs(y);  // 152 evals
	//return sq(length(p - vec2(1, 1)) + length(p + vec2(1, 1))) - 8.;  // 74 evals (flat)
	//return sq(length(p - vec2(1, 1)) + length(p + vec2(1, 1)) + length(p + vec2(-1, 1))) - 5.;  // 94 evals
	//return sq(max(abs(x), abs(y)) - 1.) + x + y;  // 137 evals
	//return abs(x - 1) + abs(x + 1) + abs(y - 2) + abs(y + 2);  // 24 evals (flat)
	//return max(log(abs(x) + abs(y)) + abs(y), -100.);  // 722 evals
	//return sqrt(abs(x - 0.1)) + log(abs(y) + 1);  // 129 evals
	//return sq(x - 1)*(x - 1) - y + 1 < 0. && x + y - 2 < 0. ? sq(1 - x) + 100 * sq(y - x * x) : NAN;  // 211 evals, local minimum
	//return min(x*x + y * y, 5.0001);  // 94 evals
	//return 5.0 - sqrt(6 - x * x - y * y);  // 86 evals
	//return sqrt(10. - sq(x*x + y * y - 5.));  // 45 evals, almost success??
	//return sq(pow(x + y, x - y) - 1.) + abs(x - y);  // **FAIL**
	//return 1.0 - 2.* cos(x)*cos(y)*exp(-x * x - y * y);  // 67 evals, **very poor local minimum**
	//return 0.001*abs(x - y) + 100 * (1. - pow(log(1 + exp(-sq(x*y - 1.)))*(1 + tanh(100.*min(x, y))), 10.));  // 1616 evals (valley)
	//return 100 * sqrt(abs(y - 0.01*x*x)) + 0.01*abs(x + 10);  // 116 evals, **FAIL**
	//return x * x - abs(x)*y + y * y - 1;  // 102 evals
	//return sqrt(x*x - abs(x)*y + y * y) - 1;  // 166 evals
	//return sin(x)*cos(y);  // 82 evals
	//return x * x + x * y + y * y - 0.5*cos(10.*x)*cos(10.*y);  // 95 evals, local minimum
	//return sq(x - 3) + sq(y + 3) + (x - 1) * (y + 1) + 6;  // 94 evals
	//return sqrt(sq(x - 3) + sq(y + 3) + (x - 1) * (y + 1) + 8.);  // 150 evals
	//return sqrt(sq(x - 3) + sq(y + 3) + (x - 1) * (y + 1) + 7.9);  // 211 evals
	//return sqrt(x*x + y * y) - exp(atan2(y, x));  // 107 evals, almost success??
	//return x * x + y * y - atan2(y, x);  // 137 evals, almost success??
	//return max(log(x*x + y * y) - atan2(y, x), -100.);  // 497 evals
	//return sq(y - x * tan(.5*log(x*x + y * y))) + 0.01*abs(x - y);  // 136 evals, unintended global minimum
	//return 100. * sq(tanh(y - x * tan(sqrt(x*x + y * y)))) + (x*x + y * y);  // 583 evals
	//return 100. * sq(tanh(y - x * tan(sqrt(x*x + y * y)))) + (sq(x - 2.) + y * y);  // 425 evals, local minimum
	//return sin(x) + sin(2.*x) / 2. + sin(4.*x) / 4. + sin(8.*x) / 8. + sin(16.*x) / 16. + sin(32.*x) / 32. + sin(64.*x) / 64. + .1*cos(y);  // 88 evals, Wow!
#undef sq
}


// optimization
struct sample {
	vec2 p;
	double val;
};
sample downhillSimplex(sample S[3]) {
	evalCount = 0;

	auto evaluate = [](vec2 x) {
		double val = fun(x);
		return isnan(val) ? INFINITY : val;
	};
	for (int i = 0; i < 3; i++) {
		S[i].val = evaluate(S[i].p);
	}

	double old_minval = INFINITY;
	int noimporv_count = 0;

	for (int iter = 0; iter < 1000; iter++) {

		std::sort(S, S + 3, [](sample a, sample b) { return a.val < b.val; });
		img_renderFrame(S[0].p, S[1].p, S[2].p, iter);

		// termination condition
		if (S[0].val < old_minval - 1e-8) {
			noimporv_count = 0;
			old_minval = S[0].val;
		}
		else if (++noimporv_count > 10) {
			printf("%d iters\n", iter);
			return S[0];
		}

		// reflection
		sample refl;
		//vec2 center = (S[0].p + S[1].p + S[2].p) / 3.;
		vec2 center = (S[0].p + S[1].p) / 2.;
		refl.p = center * 2. - S[2].p;
		refl.val = evaluate(refl.p);
		if (refl.val >= S[0].val && refl.val < S[1].val) {
			S[2] = refl;
			continue;
		}

		// expansion
		if (refl.val < S[0].val) {
			sample expd;
			expd.p = center + (center - S[2].p)*2.;
			expd.val = evaluate(expd.p);
			if (expd.val < refl.val)
				S[2] = expd;
			else
				S[2] = refl;
			continue;
		}

		// contraction
		sample ctrct;
		ctrct.p = center + .5*(S[2].p - center);
		ctrct.val = evaluate(ctrct.p);
		if (ctrct.val < S[2].val) {
			S[2] = ctrct;
			continue;
		}

		// compression
		S[1].p = S[0].p + (S[1].p - S[0].p)*.5;
		S[2].p = S[0].p + (S[2].p - S[0].p)*.5;

	}

	printf("Iteration limit exceed.\n");
	return S[0].val < S[1].val && S[0].val < S[2].val ? S[0]
		: S[1].val < S[2].val ? S[1] : S[2];
}


int main(int argc, char** argv) {
	img_init(argv[1]);
	sample S[3] = {
		sample{ vec2(-2,2) },
		sample{ vec2(-1,2) },
		sample{ vec2(-2,1) }
	};
	sample t = downhillSimplex(S);
	printf("(%lf,%lf,%lf)\n", t.p.x, t.p.y, t.val);
	printf("%d evals\n", evalCount);
	return 0;
}



// visualization

#define GIF_FLIP_VERT
#include ".libraries/gif.h"
GifWriter GIF;

#define W 600
double buffer[W][W];
uint32_t Imgbase[W][W], Img[W][W];
const double SC = 50;

#include "ui/debug_font_rendering.h"

void img_init(const char* filename) {
	double maxval = 0;
	for (int j = 0; j < W; j++) for (int i = 0; i < W; i++) {
		double val = fun(vec2(i - 0.5*W, j - 0.5*W) / SC);
		if (val > maxval) maxval = val;
		buffer[j][i] = val;
	}
	for (int i = 0; i < W; i++) for (int j = 0; j < W; j++) {
		double val = buffer[j][i];
		buffer[j][i] = maxval > 40. ? sin(PI*.5*log10(val*val + 1.)) : sin(PI*val)*tanh(PI*val);
	}
	for (int i = 1; i < W - 1; i++) for (int j = 1; j < W - 1; j++) {
		double val = buffer[i][j];
		double dfdx = (buffer[i + 1][j] - buffer[i - 1][j]) / 2.;
		double dfdy = (buffer[i][j + 1] - buffer[i][j - 1]) / 2.;
		val = abs(val) / sqrt(dfdx*dfdx + dfdy * dfdy);
		//val = min(val, min(abs(i - .5*W), abs(j - .5*W)));
		val = clamp(val, 0., 1.);
		unsigned u = (unsigned)(255.*val);
		Imgbase[i][j] = 0xFF000000 | (u << 16) | (u << 8) | u;
	}
	GifBegin(&GIF, filename, W, W, 40);
}
void img_renderFrame(vec2 a, vec2 b, vec2 c, int iter) {
	for (int i = 0; i < W; i++) for (int j = 0; j < W; j++) Img[i][j] = Imgbase[i][j];
	vec2 p0 = a * SC + vec2(.5*W), p1 = b * SC + vec2(.5*W), p2 = c * SC + vec2(.5*W);
	// just lazy
	// https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
	vec2 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;
	vec2 minp = pMin(p0, pMin(p1, p2)), maxp = pMax(p0, pMax(p1, p2));
	int i0 = max(int(minp.x), 0), i1 = min(int(maxp.x + 1.), W);
	int j0 = max(int(minp.y), 0), j1 = min(int(maxp.y + 1.), W);
	for (int i = i0; i < i1; i++) for (int j = j0; j < j1; j++) {
		vec2 p = vec2(i, j);
		vec2 v0 = p - p0, v1 = p - p1, v2 = p - p2;
		vec2 pq0 = v0 - e0 * clamp(dot(v0, e0) / e0.sqr(), 0.0, 1.0);
		vec2 pq1 = v1 - e1 * clamp(dot(v1, e1) / e1.sqr(), 0.0, 1.0);
		vec2 pq2 = v2 - e2 * clamp(dot(v2, e2) / e2.sqr(), 0.0, 1.0);
		double d = min(min(pq0.sqr(), pq1.sqr()), pq2.sqr());
		if (sqrt(d) < 1.) Img[j][i] = 0xFF0000FF;
	}
	// write words
	char s[256];
	int L = sprintf(s, "#%d eval=%d", iter, evalCount);
	for (int d = 0; d < L; d++) {
		char c = s[d]; const double FontSize = 30;
		vec2 p = vec2(10 + .5*d*FontSize, W - 10 - FontSize);
		int x0 = max((int)p.x, 0), x1 = min((int)(p.x + FontSize + 1.), W);
		int y0 = max((int)p.y, 0), y1 = min((int)(p.y + FontSize + 1.), W);
		for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) {
			double d = sdFont(c, (x - x0) / FontSize, 1.0 - (y - y0) / FontSize);
			if (d < 0.0) Img[y][x] = 0xFF800000;
		}
	}
	GifWriteFrame(&GIF, (uint8_t*)&Img[0][0], W, W, 40);
}
void img_finish() {
	GifEnd(&GIF);
}

