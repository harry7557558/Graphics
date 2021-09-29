// modified from bvh.h

#include "UI/3d_reader.h"
#include <vector>


struct BVH_Triangle {
	vec3f n;  // cross(A,B)
	vec3f p, a, b;  // P+uA+vB
};

#define MAX_TRIG 16
struct BVH {
	BVH_Triangle *obj[MAX_TRIG];
	vec3f c, r;  // bounding box
	BVH *b1 = 0, *b2 = 0;  // children
};



void constructBVH(BVH* &R, std::vector<BVH_Triangle*> &T, vec3f &Min, vec3f &Max) {
	// R should not be null and T should not be empty, calculates box range

	int N = (int)T.size();
	Min = vec3f(INFINITY), Max = vec3f(-INFINITY);

	if (N <= MAX_TRIG) {
		for (int i = 0; i < N; i++) R->obj[i] = T[i];
		if (N < MAX_TRIG) R->obj[N] = 0;
		for (int i = 0; i < N; i++) {
			vec3f a = T[i]->p, b = T[i]->p + T[i]->a, c = T[i]->p + T[i]->b;
			Min = pMin(pMin(Min, a), pMin(b, c));
			Max = pMax(pMax(Max, a), pMax(b, c));
		}
		R->c = 0.5f*(Max + Min), R->r = 0.5f*(Max - Min);
		return;
	}
	else R->obj[0] = NULL;

	// profiling shows this is the most time-consuming part of this function
	for (int i = 0; i < N; i++) {
		vec3f c = T[i]->p + 0.33333333f * (T[i]->a + T[i]->b);
		Min = pMin(Min, c), Max = pMax(Max, c);
	}
	vec3f dP = Max - Min;

	std::vector<BVH_Triangle*> c1, c2;
	if (dP.x >= dP.y && dP.x >= dP.z) {
		double x = 0.5f*(Min.x + Max.x); for (int i = 0; i < N; i++) {
			if (T[i]->p.x + 0.33333333f * (T[i]->a.x + T[i]->b.x) < x) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}
	else if (dP.y >= dP.x && dP.y >= dP.z) {
		double y = 0.5f*(Min.y + Max.y); for (int i = 0; i < N; i++) {
			if (T[i]->p.y + 0.33333333f * (T[i]->a.y + T[i]->b.y) < y) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}
	else {
		double z = 0.5f*(Min.z + Max.z); for (int i = 0; i < N; i++) {
			if (T[i]->p.z + 0.33333333f * (T[i]->a.z + T[i]->b.z) < z) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}

	if (c1.empty() || c2.empty()) {
		// faster in neither construction nor intersection
		// I keep it because...
		if (dP.x >= dP.y && dP.x >= dP.z) std::sort(T.begin(), T.end(), [](BVH_Triangle *a, BVH_Triangle *b) { return 3.f*a->p.x + a->a.x + a->b.x < 3.f*b->p.x + b->a.x + b->b.x; });
		else if (dP.y >= dP.x && dP.y >= dP.z) std::sort(T.begin(), T.end(), [](BVH_Triangle *a, BVH_Triangle *b) { return 3.f*a->p.y + a->a.y + a->b.y < 3.f*b->p.y + b->a.y + b->b.y; });
		else std::sort(T.begin(), T.end(), [](BVH_Triangle *a, BVH_Triangle *b) { return 3.f*a->p.z + a->a.z + a->b.z < 3.f*b->p.z + b->a.z + b->b.z; });
		int d = N / 2;
		c1 = std::vector<BVH_Triangle*>(T.begin(), T.begin() + d);
		c2 = std::vector<BVH_Triangle*>(T.begin() + d, T.end());
	}

	vec3f b0, b1;
	R->b1 = new BVH; constructBVH(R->b1, c1, Min, Max);
	R->b2 = new BVH; constructBVH(R->b2, c2, b0, b1);
	Min = pMin(Min, b0); Max = pMax(Max, b1);
	R->c = 0.5f*(Max + Min), R->r = 0.5f*(Max - Min);
}




// optimized intersection functions
inline float intTriangle_r(const vec3f &p, const vec3f &a, const vec3f &b,
	const vec3f &n, const vec3f &ro, const vec3f &rd) {  // relative with precomputer normal cross(a,b)
	vec3f rp = ro - p;
	vec3f q = cross(rp, rd);
	float d = 1.0f / dot(rd, n);
	float u = -d * dot(q, b); if (u<0.f || u>1.f) return (float)NAN;
	float v = d * dot(q, a); if (v<0.f || (u + v)>1.f) return (float)NAN;
	return -d * dot(n, rp);
}
inline float intBoxC(const vec3f &R, const vec3f &ro, const vec3f &inv_rd) {  // inv_rd = 1/rd
	vec3f p = -inv_rd * ro;
	vec3f k = abs(inv_rd)*R;
	vec3f t1 = p - k, t2 = p + k;
	float tN = max(max(t1.x, t1.y), t1.z);
	float tF = min(min(t2.x, t2.y), t2.z);
	if (tN > tF || tF < 0.0f) return NAN;
	return tN;
}


void rayIntersectBVH(const BVH* R,
	const vec3f &ro, const vec3f &rd, const vec3f &inv_rd, float &mt,
	BVH_Triangle* &obj) {  // assume ray already intersects current BVH
	if (R->obj[0]) {
		for (int i = 0; i < MAX_TRIG; i++) {
			BVH_Triangle *T = R->obj[i];
			if (!T) break;
			double t = intTriangle_r(T->p, T->a, T->b, T->n, ro, rd);
			if (t > 0. && t < mt) {
				mt = t, obj = T;
			}
		}
	}
	else {
		double t1 = intBoxC(R->b1->r, ro - R->b1->c, inv_rd);
		double t2 = intBoxC(R->b2->r, ro - R->b2->c, inv_rd);
#if 0
		if (t1 < mt) rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
		if (t2 < mt) rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
#else
		// test intersection for the closer box first
		// there is a significant performance increase
		if (t1 < mt && t2 < mt) {
			if (t1 < t2) {
				rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
				if (t2 < mt) rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
			}
			else {
				rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
				if (t1 < mt) rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
			}
		}
		else {
			if (t1 < mt) rayIntersectBVH(R->b1, ro, rd, inv_rd, mt, obj);
			if (t2 < mt) rayIntersectBVH(R->b2, ro, rd, inv_rd, mt, obj);
		}
#endif
	}
}

bool intersectBVH(BVH* model, vec3f ro, vec3f rd, float &mt, vec3f &n) {
	float t = intBoxC(model->r, ro - model->c, vec3f(1.0f) / rd);
	if (!(t < mt)) return false;
	if (!isnan(t)) {
		BVH_Triangle* obj = 0;
		rayIntersectBVH(model, ro, rd, vec3f(1.0f) / rd, mt, obj);
		if (obj) {
			n = normalize(obj->n);
			return true;
		}
	}
	return false;
}





// load model from file

BVH* loadModel(const char* filename) {
	FILE* fp = fopen(filename, "rb");
	vec3f *vs; ply_triangle *fs; int VN, FN; COLORREF *col;
	read3DFile(fp, vs, fs, VN, FN, col, col);
	fclose(fp);
	std::vector<BVH_Triangle*> trigs; trigs.reserve(FN);
	for (int i = 0; i < FN; i++) {
		vec3f a = vs[fs[i][0]], b = vs[fs[i][1]], c = vs[fs[i][2]];
		trigs.push_back(new BVH_Triangle{ cross(b - a, c - a), a, b - a, c - a });
	}
	BVH *R = new BVH;
	vec3f Min, Max;
	constructBVH(R, trigs, Min, Max);
	return R;
}
