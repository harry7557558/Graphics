// try simulated annealing for combinatorial optimization
// algorithm introduced in Numerical Recipes

// standard traveling salesman problem, the cost is the sum of Euclidean distances

// the code can be optimized a lot in time without changing the result,
// however, since most of the time used by the program is writing GIF,
// I won't spend much human time doing that.


#include <stdio.h>
#include <algorithm>
#include "numerical/geometry.h"
#include "numerical/random.h"

// cities as global variable
#define NCity 100
vec2 City[NCity];  // cities
int Path[NCity];  // cities to visit in order
void initCities() {
	_SRAND(0);
	for (int i = 0; i < NCity; i++) {
		City[i] = vec2(randf(0, 1), randf(0, 1));
		Path[i] = i;
	}
}

// visualization
#define GIF_FLIP_VERT
#include ".libraries/gif.h"
typedef unsigned char byte;
typedef unsigned int RGBA;  // seems to be little endian
#define IMG_W 600
RGBA IMG[IMG_W][IMG_W];
GifWriter GIF;
void writeFrame();  // at the end of this source



// calculate the "energy" of the path
double calcE(vec2* city) {
	double sum = 0;
	for (int i = 0; i < NCity; i++) {
		sum += length(city[i] - city[(i + 1) % NCity]);
	}
	return sum;
	// in practice, this can be done in O(1) time with pre-computing
}

// rearrangement: reverse the order of a path section
void reverseSection(vec2* city, int a, int b) {  // a<b inclusive
	for (int i = a, j = b; i < j; i++, j--) {
		std::swap(city[i%NCity], city[j%NCity]);
	}
}
// rearrangement: move a path section to another section
void shiftSection(vec2* city, int d0, int N, int d1) {  // d0+N<d1
	vec2 T[NCity];
	for (int i = 0; i < N; i++) T[i] = city[(d0 + i) % NCity];
	for (int i = d0; i + N < d1; i++) city[i%NCity] = city[(i + N) % NCity];
	for (int i = 0; i < N; i++) city[(d1 - N + i + NCity) % NCity] = T[i];
}


// use to compare simulated annealing
// only reverseSection: 8.587563
// only shiftSection: 8.738819
// both of the above: 8.746257
// change with different random number seeds; average ≈8.5
void downhill_only() {
	_SRAND(0); srand(1);
	vec2 City_t[NCity];
	double E0 = calcE(City);
	const int MAXJ = 0x10000; int maxJ = 0;
	for (int i = 0; i < 600; i++) {
		double E = E0;
		int j = 0;
		if (maxJ < MAXJ) {
			for (j = 0; j < MAXJ; j++) {
				for (int i = 0; i < NCity; i++) City_t[i] = City[i];
				int choice = rand() % 2;  // 0, 1, rand()%2
				if (choice == 0) {
					int a, b; do { a = randi(0, NCity), b = randi(0, NCity); b = b < a ? b + NCity : b; } while (b - a < 3);
					reverseSection(City_t, a, b);
				}
				if (choice == 1) {
					int d0, d1, N;
					do { d0 = randi(0, NCity), d1 = randi(0, NCity), N = randi(0, NCity); d1 = d1 < d0 ? d1 + NCity : d1; } while (d0 + N >= d1);
					shiftSection(City_t, d0, N, d1);
				}
				E = calcE(City_t);
				if (E < E0) break;
			}
		}
		if (E < E0) {
			E0 = E;
			for (int i = 0; i < NCity; i++) City[i] = City_t[i];
		}
		maxJ = max(maxJ, j);

		if (i % 10 == 0) writeFrame();
		printf("%d %d \t%lf\n", i, j, E0);
	}
}


// 8.013108
// _SRAND(3): 7.981511
// _SRAND(5): 7.949294
// _SRAND(7): 7.959723
// average ≈8.0
void simulated_annealing() {
	_SRAND(0); srand(1);
	vec2 City_t[NCity];

	const int nOver = 100 * NCity;  // maximum # of paths to try
	const int nLimit = 10 * NCity;  // maximum # of successful path changes before continuing

	double T = 0.5;
	double E0 = calcE(City);

	for (int i = 0; i < 100; i++) {
		int succ_count = 0;
		for (int k = 1; k < nOver; k++) {
			// find the starting and ending segment points
			int a, b;
			do { a = randi(0, NCity), b = randi(0, NCity); b += NCity * (b < a); } while (b - a < 2);
			for (int i = 0; i < NCity; i++) City_t[i] = City[i];
			// segment reversal/transport
			if (rand() & 1) {  // reversal
				reverseSection(City_t, a, b);
			}
			else {  // transport
				int N = randi(b - a + 1, NCity);
				shiftSection(City_t, a, b - a, a + N);
			}
			// calculate cost
			double E = calcE(City_t);  // note that this can be done in constant time
			double dE = E - E0;
			if (dE < 0. || (randf(0., 1.) < exp(-dE / T))) {
				succ_count++;
				E0 = E;
				for (int i = 0; i < NCity; i++) City[i] = City_t[i];
			}
			if (succ_count > nLimit) break;
		}
		writeFrame();
		T *= 0.9;
	}
}



int main(int argc, char** argv) {
	// output format: GIF
	GifBegin(&GIF, argv[1], IMG_W, IMG_W, 4);

	initCities();
	writeFrame();

	simulated_annealing();

	GifEnd(&GIF);
	return 0;
}





// rendering function
#include "_debug_font_rendering.h"
void writeFrame() {
	// clear image
	for (int i = 0; i < IMG_W*IMG_W; i++) IMG[0][i] = 0xFFFFFF;
	// draw cities
	for (int i = 0; i < NCity; i++) {
		vec2 c = City[i] * IMG_W;
		const double r = 3.0;
		int x0 = max((int)floor(c.x - r), 0), x1 = min((int)ceil(c.x + r), IMG_W - 1);
		int y0 = max((int)floor(c.y - r), 0), y1 = min((int)ceil(c.y + r), IMG_W - 1);
		for (int x = x0; x <= x1; x++) for (int y = y0; y <= y1; y++) {
			if ((vec2(x, y) - c).sqr() < r*r) IMG[y][x] = 0x000000;
		}
	}
	// draw lines
	for (int i = 0; i < NCity; i++) {
		vec2 p = City[i] * IMG_W, q = City[(i + 1) % NCity] * IMG_W;
		vec2 d = q - p;
		double slope = d.y / d.x;
		if (abs(slope) <= 1.0) {
			if (p.x > q.x) std::swap(p, q);
			int x0 = max(0, int(p.x)), x1 = min(IMG_W - 1, int(q.x)), y;
			double yf = slope * x0 + (p.y - slope * p.x);
			for (int x = x0; x <= x1; x++) {
				y = (int)yf;
				if (y >= 0 && y < IMG_W) IMG[y][x] = 0x000000;
				yf += slope;
			}
		}
		else {
			slope = d.x / d.y;
			if (p.y > q.y) std::swap(p, q);
			int y0 = max(0, int(p.y)), y1 = min(IMG_W - 1, int(q.y)), x;
			double xf = slope * y0 + (p.x - slope * p.y);
			for (int y = y0; y <= y1; y++) {
				x = (int)xf;
				if (x >= 0 && x < IMG_W) IMG[y][x] = 0x000000;
				xf += slope;
			}
		}
	}
	// font
	const double FontSize = 30;
	char s[256]; int L = sprintf(s, "%lf", calcE(City));
	for (int d = 0; d < L; d++) {
		char c = s[d];
		vec2 p = vec2(.5*d*FontSize, IMG_W - FontSize);
		int x0 = max((int)p.x, 0), y0 = max((int)p.y, 0);
		int x1 = min((int)(p.x + FontSize), IMG_W - 1), y1 = min((int)(p.y + FontSize), IMG_W - 1);
		for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
			double d = sdFont(c, (x - x0) / FontSize, 1.0 - (y - y0) / FontSize);
			if (d < 0.0) IMG[y][x] = 0x000000;
		}
	}
	// output
	GifWriteFrame(&GIF, (uint8_t*)&IMG[0][0], IMG_W, IMG_W, 4);
}

