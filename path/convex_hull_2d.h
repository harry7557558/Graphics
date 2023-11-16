// find the convex hull of points on a 2d plane
// Graham scan - O(NlogN)

#if 0
#include "numerical/geometry.h"  // vec2
#endif

#include <vector>


// return the points on the convex hull in order
template<typename vec_2>
std::vector<vec_2> convexHull_2d(std::vector<vec_2> P) {
	std::sort(P.begin(), P.end(), [](vec_2 p, vec_2 q) {
		return p.x == q.x ? p.y < q.y : p.x < q.x;
	});
	std::vector<vec_2> C;
	C.push_back(P[0]);
	for (int i = 1; i < (int)P.size();) {
		int Cn = C.size();
		if (Cn == 1) C.push_back(P[i]);
		else {
			if (det(C[Cn - 1] - C[Cn - 2], P[i] - C[Cn - 2]) <= 0) {
				C[Cn - 1] = P[i];
				while (Cn > 2 && det(C[Cn - 2] - C[Cn - 3], C[Cn - 1] - C[Cn - 3]) <= 0)
					C.pop_back(), Cn--, C[Cn - 1] = P[i];
			}
			else C.push_back(P[i]);
		}
		do { i++; } while (i < (int)P.size() && P[i].x == P[i - 1].x);
	}
	for (int i = P.size() - 1; i >= 0;) {
		int Cn = C.size();
		if (i == P.size() - 1) {
			if (!(C[Cn - 1] == P[i])) C.push_back(P[i]);
		}
		else {
			if (det(C[Cn - 1] - C[Cn - 2], P[i] - C[Cn - 2]) < 0) {
				C[Cn - 1] = P[i];
				while (det(C[Cn - 2] - C[Cn - 3], C[Cn - 1] - C[Cn - 3]) < 0)
					C.pop_back(), Cn--, C[Cn - 1] = P[i];
			}
			else C.push_back(P[i]);
		}
		do { i--; } while (i >= 0 && P[i].x == P[i + 1].x);
	}
	if (C.back() == C[0]) C.pop_back();
	return C;
}


// exactly the same function except it uses static arrays
// C should be allocated the same size of memory as P
// warn that P will be sorted after the call
template<typename vec_2>
void convexHull_2d(vec_2 *P, int Pn, vec_2 *C, int &Cn) {
	std::sort(P, P + Pn, [](vec_2 p, vec_2 q) {
		return p.x == q.x ? p.y < q.y : p.x < q.x;
	});
	Cn = 0; C[Cn++] = P[0];
	for (int i = 1; i < Pn;) {
		if (Cn == 1) C[Cn++] = P[i];
		else {
			if (det(C[Cn - 1] - C[Cn - 2], P[i] - C[Cn - 2]) <= 0) {
				C[Cn - 1] = P[i];
				while (Cn > 2 && det(C[Cn - 2] - C[Cn - 3], C[Cn - 1] - C[Cn - 3]) <= 0) Cn--, C[Cn - 1] = P[i];
			}
			else C[Cn++] = P[i];
		}
		do { i++; } while (i < Pn && P[i].x == P[i - 1].x);
	}
	for (int i = Pn - 1; i >= 0;) {
		if (i == Pn - 1) {
			if (!(C[Cn - 1] == P[i])) C[Cn++] = P[i];
		}
		else {
			if (det(C[Cn - 1] - C[Cn - 2], P[i] - C[Cn - 2]) < 0) {
				C[Cn - 1] = P[i];
				while (det(C[Cn - 2] - C[Cn - 3], C[Cn - 1] - C[Cn - 3]) < 0) Cn--, C[Cn - 1] = P[i];
			}
			else C[Cn++] = P[i];
		}
		do { i--; } while (i >= 0 && P[i].x == P[i + 1].x);
	}
	if (C[Cn - 1] == C[0]) Cn--;
}


// To-do: https://en.wikipedia.org/wiki/Convex_hull_of_a_simple_polygon

