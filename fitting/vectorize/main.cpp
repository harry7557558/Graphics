#include <cstdlib>
#include <cstdio>
#include <vector>

#include "rootsolve.h"



void printCurve(mat2x4 c) {
	printf("(%f,%f)+(%f,%f)t+(%f,%f)t^2+(%f,%f)t^3\n",
		c[0][0], c[1][0], c[0][1], c[1][1], c[0][2], c[1][2], c[0][3], c[1][3]);
}


mat2x4 fitCubicCurveLinear(
	const std::vector<vec2> &seg,
	const std::vector<float> &ts
) {
	int n = (int)seg.size();
	mat4 eTeTT(0.0);
	mat2x4 eTseg(0.0);
	for (int i = 0; i < n; i++) {
		float t = ts[i];
		vec4 eT(1.0f, t, t*t, t*t*t);
		eTeTT += outerProduct(eT, eT);
		eTseg += outerProduct(eT, seg[i]);
	}
	return inverse(eTeTT) * eTseg;
}


mat2x4 fitCubicCurve(
	const std::vector<vec2> &seg,
	std::vector<float> &ts,
	std::vector<float> &errs
) {
	int n = (int)seg.size();

	// initial guess
	if (ts.empty()) {
		ts.resize(n);
		for (int i = 0; i < n; i++)
			ts[i] = (float)i / (float)(n-1);
	}
	mat2x4 c1 = fitCubicCurveLinear(seg, ts);

	// iterative linear fit
	mat2x4 c2 = c1;
	std::vector<float> dts(n, 0.0f);
	errs = std::vector<float>(n, 0.0f);
	for (int iter = 0; iter < 3; iter++) {
		// move to closest point to curve, slightly over
		mat4x2 c2T = transpose(c2);
		for (int i = 0; i < n; i++) {
			vec4 r = cubicCurveDistanceSquared<vec2>(
				(vec2*)&c2T[0], seg[i]);
			dts[i] = r.x - ts[i];
			ts[i] += 1.5f*dts[i];
			errs[i] = r.y;
		}
		// normalize t
		float tmin = ts[0], tmax = ts.back();
		for (int i = 0; i < n; i++)
			ts[i] = (ts[i]-tmin) / (tmax-tmin);
		// linear fit
		c2 = fitCubicCurveLinear(seg, ts);
		// termination
		float maxdt = 0.0f;
		float maxerr = 0.0f;
		for (int i = 0; i < n; i++) {
			maxdt = fmax(maxdt, fabs(dts[i]));
			maxerr = fmax(maxerr, errs[i]);
		}
		maxerr = sqrt(maxerr);
		const float tol = 0.1f;
		if (maxdt < 1e-3 ||
			(iter >= 2 && maxerr > 1.5f*tol) ||
			(maxerr < 0.5f*tol)
			) break;
	}

	return c2;
}


template<typename vec>
void gaussianFilter1d(
	std::vector<vec> &v,
	float sigma, int step
) {
	int n = (int)v.size();
	float s = sigma*n;
	int w = (int)(sqrt(12.0f*s*s/(float)step+1.0f)/2.0f);
	auto getI = [&](int i) {
		int m = 2*n-2;
		int j = i % m;
		j = (j+m) % m;
		return j < n ? j : m-j;
	};
	std::vector<vec> v1 = v;
	for (int iter = 0; iter < step; iter++) {
		vec s = vec(0.0);
		for (int i = -w; i <= w; i++)
			s += v[getI(i)];
		for (int i = 0; i < n; i++) {
			v1[i] = s / (float)(2*w+1);
			s = s - v[getI(i-w)] + v[getI(i+w+1)];
		}
		v = v1;
	}
}


std::vector<mat2x4> fitCubicSpline(
	std::vector<vec2> seg
) {
	// fit curve
	int n = (int)seg.size();
	std::vector<float> errs, ts;
	mat2x4 c = fitCubicCurve(seg, ts, errs);
	if (n < 8)
		return std::vector<mat2x4>(1, c);

	// cut at maximum error between valleys
	gaussianFilter1d<float>(errs, 1.0f/n, 3);
	std::vector<int> mins;
	for (int i = 4; i < n-5; i++) {
		if (errs[i]<errs[i-1] && errs[i]<errs[i+1])
			mins.push_back(i);
	}
	int maxi = n / 2;
	if (mins.size() > 2) {
		for (int i = mins[0]; i < mins.back(); i++)
			if (errs[i] > errs[maxi])
				maxi = i;
	}
	float maxe = errs[maxi];

	// divide recursively
	float tol = 0.01f;
	if (maxe < tol*tol)
		return std::vector<mat2x4>(1, c);
	std::vector<mat2x4> sp1 = fitCubicSpline(
		std::vector<vec2>(seg.begin(), seg.begin()+maxi+1));
	std::vector<mat2x4> sp2 = fitCubicSpline(
		std::vector<vec2>(seg.begin()+maxi, seg.end()));
	sp1.insert(sp1.end(), sp2.begin(), sp2.end());
	return sp1;
}

int main() {
	const int n = 1000;
	std::vector<vec2> seg;
	for (int i = 0; i <= n; i++) {
		float t = (float)i / (float)n;
		// vec2 p = vec2(cos(7.0f*PIf*t),sin(3.0f*PIf*t));
		vec2 p = vec2(cos(6.0f*t),sin(6.0f*t))*(float)(t+asin(sin(14.0f*t)));
		seg.push_back(p);
	}
	// std::vector<float> ts, errs;
	// mat2x4 c = fitCubicCurve(seg, ts, errs);
	// printCurve(c);
	
	std::vector<mat2x4> cs = fitCubicSpline(seg);
	for (mat2x4 c : cs)
		printCurve(c);
	return 0;
}
