// most of code are for rendering not simulating

#include <cmath>
using namespace std;

#define PI 3.1415926535897932384626
#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

// https://github.com/charlietangora/gif-h
#include "libraries\gif.h"
#pragma warning(disable:4996)

typedef unsigned char byte;
typedef unsigned int abgr;

class vec2 {
public:
	double x, y;
	vec2() {}
	vec2(double a) :x(a), y(a) {}
	vec2(double x, double y) :x(x), y(y) {}
	vec2 operator - () const { return vec2(-x, -y); }
	vec2 operator + (vec2 v) const { return vec2(x + v.x, y + v.y); }
	vec2 operator - (vec2 v) const { return vec2(x - v.x, y - v.y); }
	vec2 operator * (double a) const { return vec2(x*a, y*a); }
	double sqr() const { return x * x + y * y; }
	friend double length(vec2 v) { return sqrt(v.x*v.x + v.y*v.y); }
	friend vec2 normalize(vec2 v) { return v * (1. / sqrt(v.x*v.x + v.y*v.y)); }
	friend double dot(vec2 u, vec2 v) { return u.x*v.x + u.y*v.y; }
	friend double det(vec2 u, vec2 v) { return u.x*v.y - u.y*v.x; }
};



const double g = 9.8;
const double r = 0.015;
const double E = 0.7, e = 0.99; // coefficient of restitution
const vec2 box(2.0, 3.0);	// [-x, x], [0, y]

double obstacle(vec2 p) {
	p = p - vec2(0.3, 0.0);
	double d = length(p) - 0.5;
	return max(d, p.x);
}

const int N = 1000;
vec2 *p, *v;

void init() {
	const double offset = 0.01*r;
	p[0] = vec2(r - box.x, r), v[0] = vec2(0.0);
	for (int i = 1; i < N; i++) {
		p[i] = p[i - 1] + vec2(0.0, 2.0*r + offset);
		if (p[i].y > box.y - (r + offset)) p[i].y = r, p[i].x += 2.0*r + offset;
		v[i] = vec2(sin(436237.47*i), cos(236892.24*i))*0.001;
	}
}

void render(abgr* img, int w) {
	const double SC = 1.5 * length(box) / w;
	for (int j = 0; j < w; j++) {
		for (int i = 0; i < w; i++) {
			vec2 P = vec2(i - 0.5*w, 0.5*w - j) * SC;
			if ((abs(P.x) < box.x && P.y >= -0.5*box.y && P.y < 0.5*box.y) && obstacle(P + vec2(0.0, 0.5*box.y)) > 0.0) img[j*w + i] = 0xFFFFFFFF;
			else img[j*w + i] = 0xFF7F7F7F;
		}
	}
	double R = r / SC;
	for (int d = 0; d < N; d++) {
		vec2 c = (p[d] - vec2(0.0, 0.5*box.y)) * (1.0 / SC); c.x += 0.5*w, c.y = 0.5*w - c.y;
		int i0 = max(0, (int)floor(c.x - R - 1)), i1 = min(w - 1, (int)ceil(c.x + R + 1));
		int j0 = max(0, (int)floor(c.y - R - 1)), j1 = min(w - 1, (int)ceil(c.y + R + 1));
		for (int i = i0; i < i1; i++) {
			for (int j = j0; j < j1; j++) {
				if (length(vec2(i, j) - c) < R) img[j*w + i] = 0xFF7F4F2F;
			}
		}
	}
}



// collision detection in O(n·log(n))

struct node {
	vec2 c, r;
	vec2 *p = 0;
	node* ch[4] = { 0, 0, 0, 0 };	// topleft, topright, bottomleft, bottomright
};
double sdBox(const node* b, const vec2 *p) {
	vec2 d = *p - b->c;
	d = vec2(abs(d.x), abs(d.y)) - b->r;
	//return d.x > 0. ? (d.y > 0. ? length(d) : d.x) : (d.y > 0. ? d.y : max(d.x, d.y));
	return max(d.x, d.y);
}

node* createTree(vec2 C, vec2 R, vec2 **P, int N) {
	if (N == 0) return 0;
	node* parent = new node;
	parent->c = C, parent->r = R;
	if (N == 1) {
		parent->p = P[0];
		return parent;
	}
	vec2 **p[4]; int n[4];
	for (int i = 0; i < 4; i++) p[i] = new vec2*[N], n[i] = 0;
	vec2 c[4], r = R * 0.5;
	c[0] = C + vec2(-r.x, r.y), c[1] = C + r, c[2] = C - r, c[3] = C + vec2(r.x, -r.y);
	for (int i = 0; i < N; i++) {
		vec2 d = *P[i] - C;
		bool x = d.x > 0.0, y = d.y > 0.0;
		if (!x && y) p[0][n[0]] = P[i], n[0]++;
		else if (x && y) p[1][n[1]] = P[i], n[1]++;
		else if (!x && !y) p[2][n[2]] = P[i], n[2]++;
		else p[3][n[3]] = P[i], n[3]++;
	}
	for (int i = 0; i < 4; i++) {
		parent->ch[i] = createTree(c[i], r, p[i], n[i]);
	}
	return parent;
}
void deleteTree(node *T) {
	for (int i = 0; i < 4; i++) {
		if (T->ch[i] != 0) deleteTree(T->ch[i]);
	}
	delete T;
}

auto reflect = [](vec2 &v, vec2 &p, vec2 n, double d, double e) {		// n·p - d = 0
	v = v - n * (2.0*dot(v, n));
	v = v * e;
	p = p - n * (2.0*(dot(n, p) - d));
};
auto collide = [](vec2 &p1, vec2 &p2, vec2 &v1, vec2 &v2, double e) {
	vec2 u1 = v1, u2 = v2;
	vec2 n = normalize(p1 - p2);
	v1 = n * dot(u2 - u1, n) + u1;
	v2 = u2 - v1 + u1;
	v1 = v1 * e, v2 = v2 * e;
	double d = 2.0 * r - length(p1 - p2);
	p1 = p1 + n * d, p2 = p2 - n * d;
};

void detectCollision(vec2 *P, node *T) {
	if (T->p) {
		if (T->p != P && length(*T->p - *P) < 2.0*r) collide(*P, *T->p, v[P - p], v[T->p - p], e);
		return;
	}
	for (int i = 0; i < 4; i++) {
		if (T->ch[i] && sdBox(T->ch[i], P) < 2.0*r) detectCollision(P, T->ch[i]);
	}
}

void collisionDetectionReaction() {
	vec2 **pf = new vec2*[N];
	for (int i = 0; i < N; i++) pf[i] = &p[i];
	node *T = createTree(vec2(0, 0.5*box.y), vec2(box.x, 0.5*box.y), pf, N);
	for (int c = 0; c < N; c++) {
		if (p[c].y < r) reflect(v[c], p[c], vec2(0, 1), r, E);
		if (p[c].x > box.x - r) reflect(v[c], p[c], vec2(1, 0), box.x - r, E);
		if (p[c].y > box.y - r) reflect(v[c], p[c], vec2(0, 1), box.y - r, E);
		if (p[c].x < -box.x + r) reflect(v[c], p[c], vec2(1, 0), -box.x + r, E);
		if (obstacle(p[c]) < r) {
			double e = 1e-4;
			vec2 n = normalize(vec2(
				obstacle(vec2(p[c].x + e, p[c].y)) - obstacle(vec2(p[c].x - e, p[c].y)),
				obstacle(vec2(p[c].x, p[c].y + e)) - obstacle(vec2(p[c].x, p[c].y - e))
			) * (0.5 / e));
			p[c] = p[c] - n * (2.0 * (obstacle(p[c]) - r));
			v[c] = v[c] - n * (2.0 * dot(v[c], n));
			v[c] = v[c] * E;
		}
		detectCollision(&p[c], T);
	}
	delete pf;
	deleteTree(T);
}





#include <chrono>

int main() {
	const double dt = 0.001;
	const int w = 1000;
	abgr* img = new abgr[w*w];
	GifWriter gif;
	GifBegin(&gif, "D:\\balls.gif", w, w, 4);

	p = new vec2[N], v = new vec2[N];
	init();

	// time recording
	double tot_t = 0.0, sim_t = 0.0, rnd_t = 0.0, enc_t = 0.0;
	auto t0 = chrono::high_resolution_clock::now();

	// simulation and rendering
	double t = 0;
	const vec2 a(0, -g);
	for (int i = 0; i < 10000; i++) {
		auto s0 = chrono::high_resolution_clock::now();
		for (int d = 0; d < N; d++) {
			v[d] = v[d] + a * dt, p[d] = p[d] + v[d] * dt;
		}
		collisionDetectionReaction();
		t += dt;
		auto s1 = chrono::high_resolution_clock::now();
		sim_t += chrono::duration<double>(s1 - s0).count();
		if (i % 40 == 0) {
			// rendering
			s0 = chrono::high_resolution_clock::now();
			render(img, w);
			s1 = chrono::high_resolution_clock::now();
			rnd_t += chrono::duration<double>(s1 - s0).count();
			// encoding
			GifWriteFrame(&gif, (uint8_t*)img, w, w, 4);
			s0 = chrono::high_resolution_clock::now();
			enc_t += chrono::duration<double>(s0 - s1).count();
		}
	}

	auto t1 = chrono::high_resolution_clock::now();
	tot_t += chrono::duration<double>(t1 - t0).count();
	printf("tot_t: %lf\nsim_t: %lf\nrnd_t: %lf\nenc_t: %lf\n", tot_t, sim_t, rnd_t, enc_t);

	delete p; delete v;

	GifEnd(&gif);
	delete img;
	return 0;
}

