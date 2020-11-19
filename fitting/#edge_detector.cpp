#include <iostream>
#include <chrono>

#define PI 3.1415926535897932

// image format library, https://github.com/nothings/stb
#define STB_IMAGE_IMPLEMENTATION
#include ".libraries/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include ".libraries/stb_image_write.h"

#define byte unsigned char
typedef struct {
	byte r, g, b;
} rgb;
typedef struct {
	double r, g, b;
} frgb;

frgb fcol(double u) {
	frgb c; c.r = c.g = c.b = u; return c;
}
frgb fcol(double r, double g, double b) {
	frgb c; c.r = r, c.g = g, c.b = b; return c;
}

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
frgb cmin(const frgb &a, const frgb &b) {
	return fcol(min(a.r, b.r), min(a.g, b.g), min(a.b, b.b));
}
frgb cmax(const frgb &a, const frgb &b) {
	return fcol(max(a.r, b.r), max(a.g, b.g), max(a.b, b.b));
}

frgb operator * (const frgb &c, double k) {
	return fcol(k*c.r, k*c.g, k*c.b);
}
frgb operator + (const frgb &c, const frgb &d) {
	return fcol(c.r + d.r, c.g + d.g, c.b + d.b);
}
frgb operator - (const frgb &c) {
	return fcol(-c.r, -c.g, -c.b);
}
frgb operator - (const frgb &c, const frgb &d) {
	return fcol(c.r - d.r, c.g - d.g, c.b - d.b);
}
frgb operator * (const frgb &c, const frgb &d) {
	return fcol(c.r * d.r, c.g * d.g, c.b * d.b);
}
void operator += (frgb &c, const frgb &d) {
	c.r += d.r, c.g += d.g, c.b += d.b;
}
void operator *= (frgb &c, const double &d) {
	c.r *= d, c.g *= d, c.b *= d;
}
void operator *= (frgb &c, const frgb &d) {
	c.r *= d.r, c.g *= d.g, c.b *= d.b;
}
frgb abs(const frgb &c) {
	return fcol(abs(c.r), abs(c.g), abs(c.b));
}
frgb sqrt(const frgb &c) {
	return fcol(sqrt(c.r), sqrt(c.g), sqrt(c.b));
}
frgb atan2(const frgb &c, const frgb &d) {
	return fcol(atan2(c.r, d.r), atan2(c.g, d.g), atan2(c.b, d.b));
}
double dot(const frgb &c, const frgb &d) {
	return c.r*d.r + c.g*d.g + c.b*d.b;
}
double length(const frgb &c) {
	return sqrt(c.r*c.r + c.g*c.g + c.b*c.b);
}

void to_rgb(rgb &c, const frgb &fc) {
	c.r = fc.r < 0.0 ? 0 : fc.r > 1.0 ? 255 : (byte)(255.0 * fc.r);
	c.g = fc.g < 0.0 ? 0 : fc.g > 1.0 ? 255 : (byte)(255.0 * fc.g);
	c.b = fc.b < 0.0 ? 0 : fc.b > 1.0 ? 255 : (byte)(255.0 * fc.b);
}
void to_frgb(frgb &fc, const rgb &c) {
	const double s = 1. / 255.;
	fc.r = s * c.r, fc.g = s * c.g, fc.b = s * c.b;
}

template<typename T> class image {
	int w, h;
	T *data;
public:
	image() : w(0), h(0), data(0) {}
	image(int w, int h) :w(w), h(h) {
		data = new T[w*h];
	}
	image(std::string path) {
		int bpp;
		byte* img = stbi_load(&path[0], &w, &h, &bpp, 3);
		if (img == 0) {
			w = h = 0; data = 0; return;
		}
		data = (rgb*)img;
	}
	image<T>& operator = (const image<T> &other) {
		if (data) delete data;
		w = other.w, h = other.h;
		int l = w * h;
		if (l) data = new T[l];
		for (int i = 0; i < l; i++) data[i] = other.data[i];
		return *this;
	}
	image(const image<T> &other) {
		data = 0;
		*this = other;
	}
	~image() {
		if (data) delete data;
		data = 0; w = h = 0;
	}
	int width() const { return w; }
	int height() const { return h; }
	bool fail() const {
		return w == 0 || h == 0 || data == 0;
	}
	const T* operator [] (const int &n) const {
		return &data[n*w];
	}
	T* operator [] (const int &n) {
		return &data[n*w];
	}
	void resize(int w, int h) {
		if (data) delete data;
		this->w = w, this->h = h;
		data = new T[w*h];
	}
	friend bool save(const image<rgb> &img, std::string path);
	friend bool to_rgb(image<rgb> &img, const image<frgb> &fimg);
	friend bool to_frgb(image<frgb> &fimg, const image<rgb> &img);
};

bool save(const image<rgb> &img, std::string path) {
	if (img.fail()) return false;
	return stbi_write_png(&path[0], img.w, img.h, 3, (char*)img.data, img.w * 3);
}
bool to_rgb(image<rgb> &img, const image<frgb> &fimg) {
	if (fimg.fail()) return false;
	if (img.data) delete img.data;
	img.w = fimg.w, img.h = fimg.h;
	int l = img.w*img.h;
	img.data = new rgb[l];
	for (int i = 0; i < l; i++) {
		to_rgb(img.data[i], fimg.data[i]);
	}
	return true;
}
bool to_frgb(image<frgb> &fimg, const image<rgb> &img) {
	if (img.fail()) return false;
	if (fimg.data) delete fimg.data;
	fimg.w = img.w, fimg.h = img.h;
	int l = fimg.w*fimg.h;
	fimg.data = new frgb[l];
	for (int i = 0; i < l; i++) {
		to_frgb(fimg.data[i], img.data[i]);
	}
	return true;
}



class filter {
	int w;
	double *data;
public:
	filter() :w(0), data(0) {}
	filter(int w) :w(w) {
		data = new double[w*w];
	}
	filter& operator = (const filter &other) {
		if (data) delete data;
		w = other.w;
		int l = w * w;
		if (l) data = new double[l];
		for (int i = 0; i < l; i++) data[i] = other.data[i];
		return *this;
	}
	filter(const filter &other) {
		data = 0;
		*this = other;
	}
	~filter() {
		if (data) delete data;
		w = 0; data = 0;
	}
	const double* operator [] (const int &n) const {
		return &data[n*w];
	}
	double* operator [] (const int &n) {
		return &data[n*w];
	}
	int width() const { return w; }
};


void scale2(const image<frgb> &src, image<frgb> &tgt) {
	int w = src.width() / 2, h = src.height() / 2;
	tgt.resize(w, h);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			tgt[j][i] = fcol(0.0);
			for (int v = 0; v < 2; v++) for (int u = 0; u < 2; u++) {
				tgt[j][i] += src[2 * j + v][2 * i + u];
			}
			tgt[j][i] *= 0.25;
		}
	}
}


bool convolute(const image<frgb> &src, image<frgb> &tgt, const filter &c) {
	if (src.fail()) return false;
	int s = c.width(), w = src.width() - s, h = src.height() - s;
	if (w <= 0 || h <= 0) return false;
	tgt.resize(w, h);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			frgb d = fcol(0.0);
			for (int v = 0; v < s; v++) {
				for (int u = 0; u < s; u++) {
					d += src[j + v][i + u] * c[v][u];
				}
			}
			tgt[j][i] = d;
		}
	}
	return true;
}

bool smooth(const image<frgb> &src, image<frgb> &tgt, int n, double sigma) {
	filter M(2 * n + 1);
	sigma *= sigma;
	double k = 1. / (2 * PI * sigma), s = 0;
	for (int i = -n; i <= n; i++) {
		for (int j = -n; j <= n; j++) {
			M[i + n][j + n] = k * exp((i*i + j * j) / (-2 * sigma));
			s += M[i + n][j + n];
		}
	}
	for (int i = 0; i <= 2 * n; i++) {
		for (int j = 0; j <= 2 * n; j++) {
			M[i][j] /= s;
		}
	}
	s = 0;
	return convolute(src, tgt, M);
}

bool stroke(const image<frgb> &src, image<frgb> &tgt) {
	if (src.fail()) return false;
	int w = src.width() - 3, h = src.height() - 3;
	if (w <= 0 || h <= 0) return false;
	tgt.resize(w, h);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			frgb dx = (src[j + 2][i + 2] + src[j + 1][i + 2] * 2.0 + src[j][i + 2]) - (src[j + 2][i] + src[j + 1][i] * 2.0 + src[j][i]);
			frgb dy = (src[j + 2][i + 2] + src[j + 2][i + 1] * 2.0 + src[j + 2][i]) - (src[j][i + 2] + src[j][i + 1] * 2.0 + src[j][i]);
			tgt[j][i] = cmax(abs(dx), abs(dy));
			tgt[j][i] = fcol(max(max(tgt[j][i].r, tgt[j][i].b), tgt[j][i].b));
		}
	}
	return true;
}

typedef struct {
	double m; byte d;
} grad;
bool imgGrad(const image<frgb> &src, image<frgb> &tgt) {
	if (src.fail()) return false;
	int w = src.width(), h = src.height();
	image<grad> grd(w - 2, h - 2);
	for (int j = 1; j < h - 1; j++) {
		for (int i = 1; i < w - 1; i++) {
			frgb dx = (src[j + 1][i + 1] + src[j][i + 1] * 2.0 + src[j - 1][i + 1]) - (src[j + 1][i - 1] + src[j][i - 1] * 2.0 + src[j - 1][i - 1]);
			frgb dy = (src[j + 1][i + 1] + src[j + 1][i] * 2.0 + src[j + 1][i - 1]) - (src[j - 1][i + 1] + src[j - 1][i] * 2.0 + src[j - 1][i - 1]);
			frgb M = sqrt(dx * dx + dy * dy) * (1. / 6); double m, a;
			if (M.r >= M.g && M.r >= M.b) m = M.r, a = atan2(dy.r, dx.r);
			else if (M.g >= M.b) m = M.g, a = atan2(dy.g, dx.g);
			else m = M.b, a = atan2(dy.b, dx.b);
			grd[j - 1][i - 1].m = m, grd[j - 1][i - 1].d = (byte)(a * (4 / PI) + 4.5) % 4;
		}
	}
	w -= 2, h -= 2;
	double T = 8 / 256.0, t = 4 / 256.0;  // these may be chosen adaptively
	image<byte> dth(w - 2, h - 2);
	for (int j = 1; j < h - 1; j++) {
		for (int i = 1; i < w - 1; i++) {
			dth[j - 1][i - 1] = 0;
			if (grd[j][i].m > t) {
				if (((grd[j][i].d) == 0 && (grd[j][i].m >= grd[j][i - 1].m && grd[j][i].m >= grd[j][i + 1].m)) ||
					((grd[j][i].d) == 2 && (grd[j][i].m >= grd[j - 1][i].m && grd[j][i].m >= grd[j + 1][i].m)) ||
					((grd[j][i].d) == 1 && (grd[j][i].m >= grd[j - 1][i - 1].m && grd[j][i].m >= grd[j + 1][i + 1].m)) ||
					((grd[j][i].d) == 3 && (grd[j][i].m >= grd[j + 1][i - 1].m && grd[j][i].m >= grd[j - 1][i + 1].m))) {
					dth[j - 1][i - 1] = grd[j - 1][i - 1].m > T ? 2 : 1;
				}
			}
		}
	}
	w -= 2, h -= 2;
	tgt.resize(w, h);
	for (int j = 0; j < h; j++) {
		for (int i = 0; i < w; i++) {
			tgt[j][i] = fcol(1.0 - dth[j][i] / 2.0);
			//double m = grd[j + 1][i + 1].m*3.0; byte k = grd[j + 1][i + 1].d;
			//tgt[j][i] = fcol(m);
			//tgt[j][i] = (k == 0 ? fcol(1, 0, 0) : k == 1 ? fcol(1, 1, 0) : k == 2 ? fcol(0, 1, 0) : fcol(0, 0, 1))*m;
		}
	}
	return true;
}





#include <filesystem>




#define Try(act) \
	if (!act) { \
		printf("Error %d\n", __LINE__); \
		continue; \
	}

int main(int argc, char* argv[]) {
	// load images
	std::vector<std::string> in_paths, out_paths;
	for (const auto &entry : std::filesystem::directory_iterator(argv[1])) {
		std::string path(&entry.path().string()[0]);
		if (path.find("_edge.png") == -1)
			in_paths.push_back(path),
			out_paths.push_back(path + "_edge.png");
	}
	for (auto i : out_paths)
		printf("%s\n", &i[0]);
	for (int T = 0, TL = out_paths.size(); T < TL; T++) {
		image<rgb> img(in_paths[T]);
		image<frgb> src, tgt;
		Try(to_frgb(src, img));
#if 0
		Try(smooth(src, tgt, 2, 1.4));
		src = tgt;
		Try(stroke(src, tgt));
#else
		Try(smooth(src, tgt, 2, 1.4));
		src = tgt;
		image<grad> grad;
		//Try(imgGrad(src, grad));
		Try(imgGrad(src, tgt));
#endif
		Try(to_rgb(img, tgt));
		std::string s = out_paths[T];
		Try(save(img, s));
		printf("%s\n", &s[0]);
	}
	return 0;
}

