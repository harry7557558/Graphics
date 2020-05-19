// most of code are for rendering not simulating

#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626

// https://github.com/charlietangora/gif-h
#include "libraries\gif.h"
#pragma warning(disable:4996)

typedef unsigned char byte;
typedef unsigned int abgr;

class vec3 {
public:
	double x, y, z;
	vec3() {}
	vec3(double a) :x(a), y(a), z(a) {}
	vec3(double x, double y, double z) :x(x), y(y), z(z) {}
	vec3 operator - () const { return vec3(-x, -y, -z); }
	vec3 operator + (const vec3 &v) const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator - (const vec3 &v) const { return vec3(x - v.x, y - v.y, z - v.z); }
	vec3 operator * (const double &k) const { return vec3(k * x, k * y, k * z); }
	double sqr() { return x * x + y * y + z * z; }
	friend double length(vec3 v) { return sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
	friend vec3 normalize(vec3 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y + v.z*v.z)); }
	friend double dot(vec3 u, vec3 v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
	friend vec3 cross(vec3 u, vec3 v) { return vec3(u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x); }
};

const double g = 9.8;
const double r = 1.0;

// a disgusting function
void render3d(abgr* img, int w, vec3 CM, vec3 pos, vec3 dir, double su) {
	const vec3 light = normalize(vec3(0.2, -0.5, 1.0));
	auto intSphere = [](double r, vec3 p, vec3 d, double &t, vec3 &n) {
		//if (dot(p, d) >= 0.0) return false;
		double rd2 = cross(p, d).sqr(); if (rd2 >= r * r) return false;
		t = sqrt(p.sqr() - rd2) - sqrt(r*r - rd2); if (t < 1e-6) return false;
		n = (p + d * t) * (1. / r); return true;
	};
	dir = normalize(dir); double rz = atan2(dir.x, -dir.y), rx = atan2(hypot(dir.x, dir.y), dir.z);
	double M[3][3] = { {cos(rz), -cos(rx)*sin(rz), sin(rx)*sin(rz)}, {sin(rz), cos(rx)*cos(rz), -sin(rx)*cos(rz)}, {0, sin(rx), cos(rx)} };

	// I HATE this axis for polluting the simplicity and efficiency of my source (see how simple it is in the 2d rendering function)
	const vec3 cyl_d = normalize(CM); const double cyl_h = r, cyl_r = 0.03, cyl_r2 = cyl_r * cyl_r;
	rz = atan2(cyl_d.x, -cyl_d.y), rx = atan2(hypot(cyl_d.x, cyl_d.y), cyl_d.z);
	auto cos2 = [](double t) {return cos(t)*cos(t); }; auto sin2 = [](double t) {return sin(t)*sin(t); };
	double c_x2 = cos2(rz) + cos2(rx)*sin2(rz), c_y2 = sin2(rz) + cos2(rx)*cos2(rz), c_z2 = sin2(rx), c_xy = 2 * sin2(rx)*sin(rz)*cos(rz), c_xz = -2 * sin(rx)*cos(rx)*sin(rz), c_yz = 2 * sin(rx)*cos(rx)*cos(rz);
	double px2 = pos.x*pos.x, py2 = pos.y*pos.y, pz2 = pos.z*pos.z, pxy = pos.x*pos.y, pxz = pos.x*pos.z, pyz = pos.y*pos.z;
	auto intCylinder = [&](vec3 d, double &t, vec3 &n) {
		double a = 2.0 * (c_x2 * d.x*d.x + c_y2 * d.y*d.y + c_z2 * d.z*d.z + c_xy * d.x*d.y + c_xz * d.x*d.z + c_yz * d.y*d.z);
		double b = 2.0 * (c_x2 * d.x*pos.x + c_y2 * d.y*pos.y + c_z2 * d.z*pos.z) + c_xy * (d.x*pos.y + d.y*pos.x) + c_xz * (d.x*pos.z + d.z*pos.x) + c_yz * (d.y*pos.z + d.z*pos.y);
		double delta = b * b - 2.0 * a * (c_x2 * px2 + c_y2 * py2 + c_z2 * pz2 + c_xy * pxy + c_xz * pxz + c_yz * pyz - cyl_r2); if (delta <= 0) return false;
		delta = sqrt(delta); delta /= a, b /= -a; a = b - delta, b += delta; if (a > b) delta = a, a = b, b = delta; if (b < 1e-6) return false;
		if (a > 1e-6) { t = a, n = pos + d * t, a = dot(n, cyl_d); if (a > 0.0 && a < cyl_h) { n = (n - cyl_d * a) * (1. / cyl_r); return true; } }
		t = b, n = pos + d * t, b = dot(n, cyl_d); if (b < 0.0 || b > cyl_h) return false;
		n = (n - cyl_d * b) * (1. / cyl_r); return true;
	};

	for (int x = 0; x < w; x++) {
		for (int y = 0; y < w; y++) {
			vec3 col(0.0);
			//for (int u = 0; u < 2; u++) for (int v = 0; v < 2; v++) {
			//vec3 d = normalize(vec3(0.5*w - (x + 0.5*u), -0.5*w + (y + 0.5*v), su*w));
			vec3 d = normalize(vec3(0.5*w - x, -0.5*w + y, su*w));
			d = vec3(M[0][0] * d.x + M[0][1] * d.y + M[0][2] * d.z, M[1][0] * d.x + M[1][1] * d.y + M[1][2] * d.z, M[2][0] * d.x + M[2][1] * d.y + M[2][2] * d.z);
			vec3 ecol = vec3(0.0);
			double t, mt = 1e+12; vec3 n, mn; bool r = false;
			if (intSphere(0.1, pos, d, t, n) && t < mt) r = true, mt = t, mn = n, ecol = vec3(0.9, 0.5, 0.5);
			if (intSphere(0.2, pos - CM, d, t, n) && t < mt) r = true, mt = t, mn = n, ecol = vec3(0.9, 0.8, 0.8);
			if (intCylinder(d, t, n) && t < mt) r = true, mt = t, mn = n, ecol = vec3(0.9, 0.9, 0.4);
			if (r) {
				double spc = dot(d - mn * (2.0*dot(mn, d)), light); if (spc < 0.0) spc = 0.0; spc *= spc, spc *= spc;
				double dif = dot(mn, light); dif = dif < 0.0 ? 0.0 : dif > 1.0 ? 1.0 : dif;
				col = col + ecol * (0.7 * dif + 0.2 * spc) + vec3(0.05, 0.1, 0.15);
			}
			//}
			//col = col * (0.25 * 255.0);
			col = col * 255.0;
			img[(w - y - 1)*w + x] = (0xFF000000 | (abgr)(col.x) | ((abgr)(col.y) << 8) | ((abgr)(col.z) << 16));
		}
	}
}


int main() {
	const double dt = 0.001;
	const int w = 400;
	abgr* img = new abgr[w*w];
	GifWriter gif;
	GifBegin(&gif, "D:\\pendulum3d.gif", w, w, 4);

	double t = 0;
	vec3 v(0.0, 1.0, 2.0), p(2.0, 2.0, -1.0); p = normalize(p)*r;
	double E = 0.5*dot(v, v) + g * p.z;
	for (int i = 0; i < 10000; i++) {
		double r1 = length(p);
		vec3 u = p * (1. / r1), a;
		/*if (r1 < r) {	// free fall
			a = vec3(0.0, 0.0, -g);
		}
		else*/ {	// slide down
			p = u * r; r1 = r;
			double m = sqrt(2.0*(E - g * p.z));
			double u1 = sqrt(1.0 - u.z*u.z);
			vec3 in(-u.z*u.x / u1, -u.z*u.y / u1, u1);
			//v = (u1 * v.z > sqrt(v.x*v.x + v.y*v.y) * u.z ? in : -in) * m;
			v = normalize(v) * m;	// debug
			vec3 ag = in * (-g * u1 / (r1*r1));
			vec3 ac = u * (-dot(v, v) / r1);
			a = ag + ac;
			//a = vec2(0.0, 0.0, -g);
		}
		t += dt, v = v + a * dt, p = p + v * dt;
		if (i % 40 == 0) {
			render3d(img, w, p, vec3(3.0, -5.0, 1.0), vec3(-3.0, 5.0, -1.0), 2.0);
			GifWriteFrame(&gif, (uint8_t*)img, w, w, 4);
		}
	}

	GifEnd(&gif);
	delete img;
	return 0;
}

