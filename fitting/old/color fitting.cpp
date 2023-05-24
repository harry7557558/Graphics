#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626

// http://nothings.org/stb
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Regression, it is not recommend for N > 8
void fitPolynomial(double *x, double *y, unsigned n, double *c, unsigned N) {
	unsigned L = N + 1;
	double *sumxn = new double[2 * N + 1], *sumxny = new double[L];
	for (unsigned i = 0; i <= 2 * N; i++) sumxn[i] = 0;
	for (unsigned i = 0; i < L; i++) sumxny[i] = 0;
	for (unsigned i = 0, j; i < n; i++) {
		double t;
		for (j = 0, t = 1.0; j <= 2 * N; j++) sumxn[j] += t, t *= x[i];
		for (j = 0, t = 1.0; j <= N; j++) sumxny[j] += t * y[i], t *= x[i];
	}
	double *Mat = new double[L*L];
	for (unsigned i = 0; i < L; i++) {
		for (unsigned j = 0; j < L; j++) {
			Mat[i*L + j] = sumxn[2 * N - i - j];
		}
	}
	double *X = new double[L];
	for (unsigned i = 0; i < L; i++) X[i] = sumxny[N - i];
	delete sumxn, sumxny;

#define M(i,j) Mat[(i)*L+(j)]
	double d;
	for (unsigned i = 0; i < L; i++) {
		for (unsigned j = i + 1; j < L; j++) {
			d = -(M(j, i) / M(i, i));
			for (unsigned k = i; k < L; k++) M(j, k) += M(i, k) * d;
			X[j] += X[i] * d;
		}
	}
	for (int i = L - 1; i >= 0; i--) {
		for (int j = i - 1; j >= 0; j--) {
			d = -(M(j, i) / M(i, i));
			for (int k = L - 1; k >= i; k--) M(j, k) += M(i, k) * d;
			X[j] += X[i] * d;
		}
	}
	for (unsigned i = 0; i < L; i++) X[i] /= M(i, i);
#undef M

	for (unsigned i = 0; i < L; i++) c[i] = X[N - i];
	delete X;
	delete Mat;
}

double eval(const double *c, int n, double t) {
	double r = c[n];
	for (int i = n; i > 0; i--) {
		r = r * t + c[i - 1];
	}
	return r;
}


typedef unsigned char byte;
typedef struct {
	byte r, g, b;
} rgb;

#define clamp(x,x0,x1) ((x)<(x0)?(x0):(x)>(x1)?(x1):(x))


void visualize_origional(rgb* canvas, int w, int h, const double *cr, const double *cg, const double *cb, int n) {
	auto getVal = [](const double *a, int n, double t) ->double {	// cubic interpolation
		double d = t * n; int m = (int)floor(d); t = d - m;
		double v1 = m == 0 ? a[0] : a[m - 1], v2 = a[m], v3 = m + 1 < n ? a[m + 1] : a[n - 1], v4 = m + 2 < n ? a[m + 2] : a[n - 1];
		double v = (((-0.5*v1 + 1.5*v2 - 1.5*v3 + 0.5*v4)*t + (v1 - 2.5*v2 + 2.0*v3 - 0.5*v4))*t + (-0.5*v1 + 0.5*v3))*t + v2;
		return v * 255.0;
	};
	for (int i = 0; i < w; i++) {
		double t = double(i) / w;
		double r = getVal(cr, n, t), g = getVal(cg, n, t), b = getVal(cb, n, t);
		rgb col;
		col.r = (byte)r, col.g = (byte)g, col.b = (byte)b;
		for (int j = 0; j < h; j++) canvas[j*w + i] = col;
		int R = r * h / 255.0, G = g * h / 255.0, B = b * h / 255.0;
		unsigned Col;
		if (R >= 0 && R < h) Col = 0x000000FF, canvas[(h - R - 1)*w + i] = *(rgb*)&Col;
		if (G >= 0 && G < h) Col = 0x00007F00, canvas[(h - G - 1)*w + i] = *(rgb*)&Col;
		if (B >= 0 && B < h) Col = 0x00FF0000, canvas[(h - B - 1)*w + i] = *(rgb*)&Col;
	}
}

void visualize(rgb* canvas, int w, int h, const double *cr, unsigned dr, const double* cg, unsigned dg, const double *cb, unsigned db) {
	for (int i = 0; i < w; i++) {
		double t = double(i) / w;
		rgb col;
		double r = 255.0*eval(cr, dr, t);
		double g = 255.0*eval(cg, dg, t);
		double b = 255.0*eval(cb, db, t);
		col.r = (byte)clamp(r, 0.0, 255.99), col.g = (byte)clamp(g, 0.0, 255.99), col.b = (byte)clamp(b, 0.0, 255.99);
		for (int j = 0; j < h; j++) canvas[j*w + i] = col;
		int R = r * h / 255.0, G = g * h / 255.0, B = b * h / 255.0;
		unsigned Col;
		if (R >= 0 && R < h) Col = 0x000000FF, canvas[(h - R - 1)*w + i] = *(rgb*)&Col;
		if (G >= 0 && G < h) Col = 0x00007F00, canvas[(h - G - 1)*w + i] = *(rgb*)&Col;
		if (B >= 0 && B < h) Col = 0x00FF0000, canvas[(h - B - 1)*w + i] = *(rgb*)&Col;
	}
}


#define OUTPUT_W 768
#define OUTPUT_H 128


int main() {
	cout << "+------------------------------+\n|        Color Fitting         |\n+------------------------------+\n";
	cout << "Find a RGB coloring function based on a bitmap image.\nImage decoder from http://nothings.org/stb, support most common image formats.\n\n";

	auto prompt = [](string s) {
		cout << s;
		for (unsigned i = 0; i < s.size(); i++) s[i] = '\b';
		cout << s;
	};

	string filename;
	cout << "Select Image: ";
	prompt("[filename/path]");
	//getline(cin, filename);
	filename = "d:\\col.png";
	while (filename[0] == ' ' && !filename.empty()) filename.erase(0, 1);
	int w, h, bpp;
	rgb* img_orig = (rgb*)stbi_load(&filename[0], &w, &h, &bpp, 3);
	if (!img_orig) {
		cout << "Error loading image. Press Enter to exit.\n";
		getchar(); return 0;
	}
	cout << endl;

	double *r = new double[w], *g = new double[w], *b = new double[w], *t = new double[w];
	for (int i = 0; i < w; i++) {
		r[i] = g[i] = b[i] = 0.0;
		for (int j = 0; j < h; j++) {
			rgb c = img_orig[j*w + i];
			r[i] += c.r, g[i] += c.g, b[i] += c.b;
		}
		r[i] /= 255.0*h, g[i] /= 255.0*h, b[i] /= 255.0*h;
		t[i] = double(i) / w;
	}

	rgb* img_orig_process = new rgb[OUTPUT_W*OUTPUT_H];
	visualize_origional(img_orig_process, OUTPUT_W, OUTPUT_H, r, g, b, w);
	stbi_write_png("D:\\col_orig.png", OUTPUT_W, OUTPUT_H, 3, img_orig_process, OUTPUT_W * 3);
	delete img_orig_process;
	//return 0;

	cout << "Image loading succeeds. (" << w << "x" << h << ")\n";


	auto output = [](string s, double *coe, int n) {
		cout << s;
		//cout << setprecision(12);
		for (int i = 1; i < n; i++) cout << "(";
		for (int i = n; i >= 0; i--) {
			if (i == 0) cout << showpos << coe[i] << endl;
			else if (i == n) cout << noshowpos << coe[i] << "*t";
			else cout << showpos << coe[i] << ")*t";
		}
	};

#define MAX_DEG 8

	auto fit = [](double *t, double *c, int n, double* &coe, unsigned &deg, double prec) {
		for (deg = 1; deg <= MAX_DEG; deg++) {
			coe = new double[deg + 1];
			fitPolynomial(t, c, n, coe, deg);
			double s = 0;
			for (int i = 0; i < n; i++) s += abs(eval(coe, deg, t[i]) - c[i]);
			s /= n;
			if (s < prec) break;
			if (deg != MAX_DEG) delete coe;
		}
	};

	double *coer, *coeg, *coeb; unsigned degr, degg, degb;
	double prec = 0.02;
	fit(t, r, w, coer, degr, prec);
	output("r(t)=", coer, degr);
	fit(t, g, w, coeg, degg, prec);
	output("g(t)=", coeg, degg);
	fit(t, b, w, coeb, degb, prec);
	output("b(t)=", coeb, degb);

	rgb *img = new rgb[OUTPUT_W*OUTPUT_H];
	visualize(img, OUTPUT_W, OUTPUT_H, coer, degr, coeg, degg, coeb, degb);

	stbi_write_png_compression_level = 100;
	filename = "D:\\col_fitted.png";
	stbi_write_png(&filename[0], OUTPUT_W, OUTPUT_H, 3, img, OUTPUT_W * 3);

	delete img;


	free(img_orig);

	return 0;
}

