#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// solving systems of linear equations using elimination, designed for polynomial fitting
void solve_q(double* Mat, double *X, int l) {
#define M(i,j) Mat[(i)*l+(j)]
	double d;
	for (int i = 0; i < l; i++) {
		for (int j = i + 1; j < l; j++) {
			d = -(M(j, i) / M(i, i));
			for (int k = i; k < l; k++) M(j, k) += M(i, k) * d;
			X[j] += X[i] * d;
		}
	}
	for (int i = l - 1; i >= 0; i--) {
		for (int j = i - 1; j >= 0; j--) {
			d = -(M(j, i) / M(i, i));
			for (int k = l - 1; k >= i; k--) M(j, k) += M(i, k) * d;
			X[j] += X[i] * d;
		}
	}
	for (int i = 0; i < l; i++) X[i] /= M(i, i);
#undef M
}

// It is not recommand for N > 8
void fitPolynomial(double *x, double *y, unsigned n, double *c, unsigned N) {
	unsigned L = N + 1;
	double *sumxn = new double[2 * N + 1], *sumxny = new double[L];
	for (unsigned i = 0; i <= 2 * N; i++) sumxn[i] = 0;
	for (unsigned i = 0; i < L; i++) sumxny[i] = 0;
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j <= 2 * N; j++) sumxn[j] += powl(x[i], j);
		for (unsigned j = 0; j < L; j++) sumxny[j] += powl(x[i], j) * y[i];
	}

	double *M = new double[L*L];
	for (unsigned i = 0; i < L; i++) {
		for (unsigned j = 0; j < L; j++) {
			M[i*L + j] = sumxn[2 * N - i - j];
		}
	}
	double *X = new double[L];
	for (unsigned i = 0; i < L; i++) X[i] = sumxny[N - i];
	solve_q(M, X, L);
	for (unsigned i = 0; i < L; i++) c[i] = X[N - i];
	delete X;
	delete M;
}

double eval(double *c, int n, double t) {
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


int main() {
	cout << "+------------------------------+\n|        Color Fitting         |\n+------------------------------+\n";
	cout << "Find a RGB coloring function based on a bitmap image.\nImage decoder from http://nothings.org/stb, support most common image formats.\n\n";

	auto prompt = [](string s) {
		cout << s;
		for (int i = 0; i < s.size(); i++) s[i] = '\b';
		cout << s;
	};

	string filename;
	cout << "Select Image: ";
	prompt("[filename/path]");
	getline(cin, filename);
	//filename = "d:\\col.png";
	while (filename[0] == ' ' && !filename.empty()) filename.erase(0, 1);
	int w, h, bpp;
	rgb* img = (rgb*)stbi_load(&filename[0], &w, &h, &bpp, 3);
	if (!img) {
		cout << "Error loading image. Press Enter to exit.\n";
		getchar(); return 0;
	}
	cout << endl;

	double *r = new double[w], *g = new double[w], *b = new double[w], *t = new double[w];
	for (int i = 0; i < w; i++) {
		r[i] = g[i] = b[i] = 0.0;
		for (int j = 0; j < h; j++) {
			rgb c = img[j*w + i];
			r[i] += c.r, g[i] += c.g, b[i] += c.b;
		}
		r[i] /= 255.0*h, g[i] /= 255.0*h, b[i] /= 255.0*h;
		t[i] = double(i) / w;
	}

	cout << "Loading image succeed. (" << w << "x" << h << ")\n";


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

	for (int i = 0; i < w; i++) {
		double t = double(i) / w;
		rgb col;
		col.r = (byte)(255.0*eval(coer, degr, t));
		col.g = (byte)(255.0*eval(coeg, degg, t));
		col.b = (byte)(255.0*eval(coeb, degb, t));
		for (int j = 0; j < h; j++) {
			img[j*w + i] = col;
		}
	}

	filename = "D:\\col_fitted.png";
	stbi_write_png(&filename[0], w, h, 3, img, w * 3);

	return 0;
}

