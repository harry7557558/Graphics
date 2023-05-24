// acceleration structure for ray tracing
// require std::vector and vec3


struct BVH_Triangle {
	vec3 n;  // cross(A,B)
	vec3 P, A, B;  // P+uA+vB
};

#define MAX_TRIG 16
struct BVH {
	BVH_Triangle *Obj[MAX_TRIG];
	vec3 C, R;  // bounding box
	BVH *b1 = 0, *b2 = 0;  // children
};



void constructBVH(BVH* &R, std::vector<BVH_Triangle*> &T, vec3 &Min, vec3 &Max) {
	// R should not be null and T should not be empty, calculates box range

	int N = (int)T.size();
	Min = vec3(INFINITY), Max = vec3(-INFINITY);

	if (N <= MAX_TRIG) {
		for (int i = 0; i < N; i++) R->Obj[i] = T[i];
		if (N < MAX_TRIG) R->Obj[N] = 0;
		for (int i = 0; i < N; i++) {
			vec3 A = T[i]->P, B = T[i]->P + T[i]->A, C = T[i]->P + T[i]->B;
			Min = pMin(pMin(Min, A), pMin(B, C));
			Max = pMax(pMax(Max, A), pMax(B, C));
		}
		R->C = 0.5*(Max + Min), R->R = 0.5*(Max - Min);
		return;
	}
	else R->Obj[0] = NULL;

	// Analysis shows this is the most time-consuming part of this function
	const double _3 = 1. / 3;
	for (int i = 0; i < N; i++) {
		vec3 C = T[i]->P + _3 * (T[i]->A + T[i]->B);
		Min = pMin(Min, C), Max = pMax(Max, C);
	}
	vec3 dP = Max - Min;

	std::vector<BVH_Triangle*> c1, c2;
	if (dP.x >= dP.y && dP.x >= dP.z) {
		double x = 0.5*(Min.x + Max.x); for (int i = 0; i < N; i++) {
			if (T[i]->P.x + _3 * (T[i]->A.x + T[i]->B.x) < x) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}
	else if (dP.y >= dP.x && dP.y >= dP.z) {
		double y = 0.5*(Min.y + Max.y); for (int i = 0; i < N; i++) {
			if (T[i]->P.y + _3 * (T[i]->A.y + T[i]->B.y) < y) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}
	else {
		double z = 0.5*(Min.z + Max.z); for (int i = 0; i < N; i++) {
			if (T[i]->P.z + _3 * (T[i]->A.z + T[i]->B.z) < z) c1.push_back(T[i]);
			else c2.push_back(T[i]);
		}
	}

	if (c1.empty() || c2.empty()) {
		// faster in neither construction nor intersection
		// I keep it because...
		if (dP.x >= dP.y && dP.x >= dP.z) std::sort(T.begin(), T.end(), [](BVH_Triangle *a, BVH_Triangle *b) { return 3.*a->P.x + a->A.x + a->B.x < 3.*b->P.x + b->A.x + b->B.x; });
		else if (dP.y >= dP.x && dP.y >= dP.z) std::sort(T.begin(), T.end(), [](BVH_Triangle *a, BVH_Triangle *b) { return 3.*a->P.y + a->A.y + a->B.y < 3.*b->P.y + b->A.y + b->B.y; });
		else std::sort(T.begin(), T.end(), [](BVH_Triangle *a, BVH_Triangle *b) { return 3.*a->P.z + a->A.z + a->B.z < 3.*b->P.z + b->A.z + b->B.z; });
		int d = N / 2;
		c1 = std::vector<BVH_Triangle*>(T.begin(), T.begin() + d);
		c2 = std::vector<BVH_Triangle*>(T.begin() + d, T.end());
	}
	// A paper I haven't read yet: https://graphicsinterface.org/wp-content/uploads/gi1989-22.pdf

	vec3 b0, b1;
	R->b1 = new BVH; constructBVH(R->b1, c1, Min, Max);
	R->b2 = new BVH; constructBVH(R->b2, c2, b0, b1);
	Min = pMin(Min, b0); Max = pMax(Max, b1);
	R->C = 0.5*(Max + Min), R->R = 0.5*(Max - Min);
}




// optimized intersection functions
#define invec3 const vec3&
inline double intTriangle_r(invec3 P, invec3 a, invec3 b, invec3 n, invec3 ro, invec3 rd) {  // relative with precomputer normal cross(a,b)
	vec3 rp = ro - P;
	vec3 q = cross(rp, rd);
	double d = 1.0 / dot(rd, n);
	double u = -d * dot(q, b); if (u<0. || u>1.) return NAN;
	double v = d * dot(q, a); if (v<0. || (u + v)>1.) return NAN;
	return -d * dot(n, rp);
}
inline double intBoxC(invec3 R, invec3 ro, invec3 inv_rd) {  // inv_rd = vec3(1.0)/rd
	vec3 p = -inv_rd * ro;
	vec3 k = abs(inv_rd)*R;
	vec3 t1 = p - k, t2 = p + k;
	double tN = max(max(t1.x, t1.y), t1.z);
	double tF = min(min(t2.x, t2.y), t2.z);
	if (tN > tF || tF < 0.0) return NAN;
	return tN;
}


void rayIntersectBVH(const BVH* R, invec3 ro, invec3 rd, invec3 inv_rd, double &mt, BVH_Triangle* &obj) {  // assume ray already intersects current BVH
	if (R->Obj[0]) {
		for (int i = 0; i < MAX_TRIG; i++) {
			BVH_Triangle *T = R->Obj[i];
			if (!T) break;
			double t = intTriangle_r(T->P, T->A, T->B, T->n, ro, rd);
			if (t > 0. && t < mt) mt = t, obj = T;
		}
	}
	else {
		double t1 = intBoxC(R->b1->R, ro - R->b1->C, inv_rd);
		double t2 = intBoxC(R->b2->R, ro - R->b2->C, inv_rd);
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

bool intersectScene(BVH* Scene, vec3 ro, vec3 rd, double &t, vec3 &n) {
	t = intBoxC(Scene->R, ro - Scene->C, vec3(1.0) / rd);
	if (!isnan(t)) {
		BVH_Triangle* obj = 0;
		t = INFINITY;
		rayIntersectBVH(Scene, ro, rd, vec3(1.0) / rd, t, obj);
		if (obj) {
			n = normalize(obj->n);
			return true;
		}
	}
	return false;
}






#ifndef _INC_STDIO
#include <stdio.h>
#endif

// load object from file

void readBinarySTL(const char* filename, BVH_Triangle* &STL, int &N) {
	FILE* fp = fopen(filename, "rb");
	fseek(fp, 80, SEEK_SET);
	fread(&N, sizeof(int), 1, fp);
	STL = new BVH_Triangle[N];
	//printf("%d\n", STL_N);
	auto readf = [&](double &x) {
		float t; fread(&t, sizeof(float), 1, fp);
		x = (double)t;
	};
	auto readTrig = [&](BVH_Triangle &T) {
		readf(T.n.x); readf(T.n.y); readf(T.n.z);
		readf(T.P.x); readf(T.P.y); readf(T.P.z);
		readf(T.A.x); readf(T.A.y); readf(T.A.z); T.A -= T.P;
		readf(T.B.x); readf(T.B.y); readf(T.B.z); T.B -= T.P;
		short c; fread(&c, 2, 1, fp);
		T.n = cross(T.A, T.B);
	};
	for (int i = 0; i < N; i++) {
		readTrig(STL[i]);
	}
	fclose(fp);
}

void BVH_BoundingBox(BVH_Triangle *P, int N, vec3 &p0, vec3 &p1) {
	p0 = vec3(INFINITY), p1 = vec3(-INFINITY);
	for (int i = 0; i < N; i++) {
		p0 = pMin(pMin(p0, P[i].P), P[i].P + pMin(P[i].A, P[i].B));
		p1 = pMax(pMax(p1, P[i].P), P[i].P + pMax(P[i].A, P[i].B));
	}
}

