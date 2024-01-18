#include <cstdlib>
#include <cstdio>
#include <vector>
#include <functional>

#include "rootsolve.h"



void printCurve(mat2x4 c) {
	printf("(%f,%f)+(%f,%f)t+(%f,%f)t^2+(%f,%f)t^3\n",
		c[0][0], c[1][0], c[0][1], c[1][1], c[0][2], c[1][2], c[0][3], c[1][3]);
}


mat2x4 fitCubicCurveLinear(
	const std::vector<vec2> &seg,
	const std::vector<float> &ts,
	const std::vector<float> &ws
) {
	int n = (int)seg.size();
	mat4 eTeTT(0.0);
	mat2x4 eTseg(0.0);
	for (int i = 0; i < n; i++) {
		float t = ts[i];
		vec4 eT(1.0f, t, t*t, t*t*t);
		// eTeTT += ws[i] * outerProduct(eT, eT);
		// eTseg += ws[i] * outerProduct(eT, seg[i]);
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
	std::vector<float> lengths(n, 0.0f);
	float totLength = 0.0f;
	for (int i = 0; i < n-1; i++) {
		float l = length(seg[i+1]-seg[i]);
		lengths[i] = l;
		totLength += l;
	}
	if (ts.empty()) {
		ts.resize(n);
		float l = 0.0;
		for (int i = 0; i < n; i++) {
			ts[i] = l / totLength;
			l += lengths[i];
			ts[i] = i / (float)(n-1);
		}
	}
	std::vector<float> ws(n, 0.0f);
	ws[0] = lengths[0]*2.0f;
	ws[n-1] = lengths[n-2]*2.0f;
	for (int i = 1; i < n-1; i++) {
		ws[i] = lengths[i-1] + lengths[i];
	}
	mat2x4 c1 = fitCubicCurveLinear(seg, ts, ws);

	// iterative linear fit
	mat2x4 c2 = c1;
	std::vector<float> dts(n, 0.0f);
	errs = std::vector<float>(n, 0.0f);
	for (int iter = 0; iter < 3; iter++) {
		// move to closest point to curve
		mat4x2 c2T = transpose(c2);
		for (int i = 0; i < n; i++) {
			vec4 r = cubicCurveDistanceSquared<vec2>(
				(vec2*)&c2T[0], seg[i]);
			dts[i] = r.x - ts[i];
			// ts[i] += dts[i];
			ts[i] += 1.5f*dts[i];
			errs[i] = r.y;
		}
		// normalize t
		float tmin = ts[0], tmax = ts.back();
		for (int i = 0; i < n; i++)
			ts[i] = (ts[i]-tmin) / (tmax-tmin);
		// linear fit
		c2 = fitCubicCurveLinear(seg, ts, ws);
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
	const int radius = 4;

	// fit curve
	int n = (int)seg.size();
	std::vector<float> errs, ts;
	mat2x4 c = fitCubicCurve(seg, ts, errs);
	if (n < 2*radius)
		return std::vector<mat2x4>(1, c);

	// error??
	std::vector<float> ks(n, 0.0f);
	for (int i = radius; i+1 < n-radius; i++) {
		#if 1
		vec2 d1 = seg[i-1]-seg[i-2];
		vec2 d2 = seg[i+2]-seg[i+1];
		float da = acos(dot(d1,d2)/sqrt(dot(d1,d1)*dot(d2,d2)));
		float dl = length(d1)+length(d2)+length(seg[i]-seg[i-1])+length(seg[i+1]-seg[i]);
		ks[i] = da / dl;
		if (std::isnan(ks[i])) ks[i] = 0.0f;
		#else
		mat3 A(0.0); vec3 b(0.0);
		for (int j = i-radius; j <= i+radius; j++) {
			vec2 p(seg[j]);
			vec3 e(dot(p,p), p);
			A += outerProduct(e, e);
			b += e;
		}
		vec3 c = inverse(A) * b;
		vec2 p = vec2(c.y,c.z)/(2.0f*c.x);
		float r2 = 1.0f/c.x + dot(p,p);
		ks[i] = -fmax(r2, 0.0f);
		#endif
	}

	// cut at maximum error between valleys
	// gaussianFilter1d<float>(angles, 1.0f/n, 3);
	std::vector<int> mins;
	for (int i = radius; i+1 < n-radius; i++) {
		if (ks[i]<ks[i-1] && ks[i]<ks[i+1])
			mins.push_back(i);
	}
	int maxi = n / 2;
	if (mins.size() > 2) {
		for (int i = mins[0]; i < mins.back(); i++)
			if (ks[i] > ks[maxi])
				maxi = i;
	}
	float meank = 0.0f;
	for (float k : ks)
		meank += k;
	meank /= n;
	if (ks[maxi] < 2.0f*meank)
		maxi = n / 2;

	// divide recursively
	float tol = 0.01f;
	float err = 0.0f;
	for (float e : errs)
		err = fmax(err, e);
	err = sqrt(err);
	// for (float e : errs)
	// 	err += e;
	// err = 2.0f * sqrt(err/n);
	if (err < tol)
		return std::vector<mat2x4>(1, c);
	if (err < 2.0*tol)
		maxi = n / 2;
	std::vector<mat2x4> sp1 = fitCubicSpline(
		std::vector<vec2>(seg.begin(), seg.begin()+maxi+1));
	std::vector<mat2x4> sp2 = fitCubicSpline(
		std::vector<vec2>(seg.begin()+maxi, seg.end()));
	sp1.insert(sp1.end(), sp2.begin(), sp2.end());
	return sp1;
}


std::vector<vec2> discretizeParametricCurve(
	std::function<vec2(float)> curve,
	float etol, float ltol, float stol
) {
    auto segmentSdf = [](vec2 p, vec2 a, vec2 b) {
        vec2 ba = b - a, pa = p - a;
        float h = dot(pa, ba) / dot(ba, ba);
        vec2 dp = pa - clamp(h, 0.0f, 1.0f) * ba;
        return length(dp);
    };
    auto errorBetween = [&](float s1, float s2) {
        vec2 p1 = curve(s1);
        vec2 p2 = curve(s2);
        int ns = (int)((s2-s1)/(0.25f*stol)+0.51f);
		ns = 20;
        float ds = (s2-s1) / ns;
        float err = 0.0;
        for (int i = 0; i < ns; i++) {
            float s = mix(s1, s2, (i+0.5f)/ns);
            vec2 p = curve(s);
            float sdf = segmentSdf(p, p1, p2);
            // err += sdf * ds;
            err = fmax(err, sdf);
        }
        // return err / (s2-s1);
        return err;
    };
    std::vector<vec2> stack;
    stack.push_back(vec2(0.0f, 1.0f));
    stack.push_back(vec2(0.0f, 0.5f));  // prevent instant termination
    std::vector<vec2> output;
    while (!stack.empty()) {
        vec2 s = stack.back();
        float err = errorBetween(s.x, s.y);
        float l = length(curve(s.y) - curve(s.x));
        float ds = s.y - s.x;
        if (err > 9.0f * etol || l > 3.0f * ltol || ds > 3.0f * stol) {
            stack.push_back(vec2(s.x, 0.5f*(s.x+s.y)));
            continue;
        }
        output.push_back(curve(s.x));
        if (err > 4.0f * etol || l > 2.0f * ltol || ds > 2.0f * stol) {
            output.push_back(curve(mix(s.x, s.y, 1.0f/3.0f)));
            output.push_back(curve(mix(s.x, s.y, 2.0f/3.0f)));
        }
        else if (err > etol || l > ltol || ds > stol) {
            output.push_back(curve(0.5f*(s.x+s.y)));
        }
        while (!stack.empty() && s.y == stack.back().y)
            stack.pop_back();
        if (!stack.empty())
            stack.push_back(vec2(s.y, stack.back().y));
    }
	output.push_back(curve(1.0f));
	return output;
}


int main() {
	auto curve = [](float t) -> vec2 {
		// return vec2(cos(7.0f*PIf*t),sin(3.0f*PIf*t));
		return vec2(cos(6.0f*t),sin(6.0f*t))*(float)(t+asin(sin(14.0f*t)));
		return vec2(cos(10.0f*t),sin(10.0f*t))*(float)(exp(2.0f*(t-1.0f))*(1.0f+0.5f*sin(100.0f*t)));
	};
	// const int n = 4000;
	// std::vector<vec2> seg;
	// for (int i = 0; i <= n; i++) {
	// 	float t = (float)i / (float)n;
	// 	seg.push_back(curve(t));
	// }
	std::vector<vec2> seg =
		discretizeParametricCurve(
			curve, 0.0001f, 0.01f, 1.0f/256.0f);
	int n = (int)seg.size();
	
	printf("[");
	for (int i = 0; i < n; i++)
		printf("(%f,%f)%s", seg[i].x, seg[i].y, i+1==n?"]\n":",");

	// std::vector<float> ts, errs;
	// mat2x4 c = fitCubicCurve(seg, ts, errs);
	// printCurve(c);
	
	std::vector<mat2x4> cs = fitCubicSpline(seg);
	printf("%d\n", (int)cs.size());
	for (mat2x4 c : cs)
		printCurve(c);
	return 0;
}
