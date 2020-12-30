// test parametric surface modeling and generate "surfaces_info.h"

#include <vector>
#include <stdio.h>
#include <algorithm>

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"

#include "surfaces.h"
#include "triangulate/parametric_surface_adaptive_dist.h"

#include "simulation/balance/balance_3d_nms.h"


std::vector<triangle_3d> comps;


// calculate some informations of each shape in "surfaces.h" and write them to "surfaces_info.h"
void shapeInfos() {
	// default number of triangles, name, surface or solid
	// axes-aligned bounding box
	// area/volume, center of mass, unit inertia tensor calculated at center of mass
	// minimum gravitational potential energy and value
	struct shapeInfo {
		char name[32];
		int Trig_N; bool isSolid;
		vec3 AABB_min, AABB_max;
		double SA_or_V; vec3 CoM; mat3 InertiaTensor_u;
		vec3 minGravPotential_vec; double minGravPotential_u;
	};

	// start writing file
	FILE* fp = fopen("surfaces_info.h", "w");
	fprintf(fp, "\
// included by \"surfaces.h\"\n\
\n\
struct shapeInfo {\n\
	char name[32];\n\
	int Trig_N; bool isSolid;\n\
	vec3 AABB_min, AABB_max;\n\
	double SA_or_V; vec3 CoM; mat3 InertiaTensor_u;\n\
	vec3 minGravPotential_vec; double minGravPotential_u;\n\
};\n\
\n\
const std::vector<shapeInfo> info({\n\
");


	for (unsigned i = 0; i < ParamSurfaces.size(); i++) {
		auto S = ParamSurfaces[i];
		shapeInfo info;

		// triangulation
		std::vector<vec3> Ps;
		S.param2points(Ps);
		int PN = Ps.size();
		std::vector<triangle_3d> T;
		int TN = S.points2trigs(&Ps[0], T);

		strcpy(info.name, S.name);
		info.Trig_N = TN; info.isSolid = S.isGaussianSurface;

		// calculate axis-aligned bounding box
		info.AABB_min = vec3(INFINITY); info.AABB_max = vec3(-INFINITY);
		for (int i = 0; i < PN; i++) {
			info.AABB_min = pMin(Ps[i], info.AABB_min);
			info.AABB_max = pMax(Ps[i], info.AABB_max);
		}

		// calculate area/volume, center of mass, and the ratio of moment of inertia to mass
		// assuming uniform unit density
		if (info.isSolid) {
			double V = 0.; vec3 C(0.); mat3 I(0.);
			for (int i = 0; i < TN; i++) {
				vec3 a = T[i][1], b = T[i][0], c = T[i][2];  // volume must be positive
				double dV = det(a, b, c) / 6.;
				V += dV;
				C += dV * (a + b + c) / 4.;
				I += dV * 0.1*(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
					(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
			}
			info.SA_or_V = V;
			info.CoM = (C = C / V);
			info.InertiaTensor_u = I * (1. / V) - (mat3(dot(C, C)) - tensor(C, C));
		}
		else {
			double A = 0.; vec3 C(0.); mat3 I(0.);
			for (int i = 0; i < TN; i++) {
				vec3 a = T[i][0], b = T[i][1], c = T[i][2];
				double dA = 0.5*length(cross(b - a, c - a));
				A += dA;
				C += dA / 3. * (a + b + c);
				I += dA / 6. *(mat3(dot(a, a) + dot(b, b) + dot(c, c) + dot(a, b) + dot(a, c) + dot(b, c)) -
					(tensor(a, a) + tensor(b, b) + tensor(c, c) + 0.5*(tensor(a, b) + tensor(a, c) + tensor(b, a) + tensor(b, c) + tensor(c, a) + tensor(c, b))));
			}
			info.SA_or_V = A;
			info.CoM = (C = C / A);
			info.InertiaTensor_u = I * (1. / A) - (mat3(dot(C, C)) - tensor(C, C));
		}
		// prevent numbers like "xxxxe-15"
		double epsilon = 1e-8;
		for (int i = 0; i < 3; i++) if (abs(((double*)&info.CoM)[i]) < epsilon) ((double*)&info.CoM)[i] = 0.;
		for (int i = 0; i < 9; i++) if (abs(((double*)&info.InertiaTensor_u)[i]) < epsilon) ((double*)&info.InertiaTensor_u)[i] = 0.;

		// find an orientation that will be "still" when placed
		for (int i = 0; i < PN; i++) Ps[i] -= info.CoM;
		vec3 minn = vec3(0, 1e-6, -1);
		info.minGravPotential_vec = balance_3d_NMS(&Ps[0], Ps.size(), minn, 1e-8*pow(determinant(info.InertiaTensor_u), 1. / 6.), &info.minGravPotential_u);

		// output
		if (1) {
			printf("[%d] - %s\n", i, S.name);
			printf("AABB: (%.8lg,%.8lg,%.8lg), (%.8lg,%.8lg,%.8lg)\n",
				info.AABB_min.x, info.AABB_min.y, info.AABB_min.z, info.AABB_max.x, info.AABB_max.y, info.AABB_max.z);
			printf("%s: %.8lg\n", info.isSolid ? "Volume" : "Area", info.SA_or_V);
			printf("Center of mass: (%.8lg,%.8lg,%.8lg)\n", info.CoM.x, info.CoM.y, info.CoM.z);
			printf("Unit inertia tensor: (\n\t%.8lg,%.8lg,%.8lg,\n\t%.8lg,%.8lg,%.8lg,\n\t%.8lg,%.8lg,%.8lg)\n",
				info.InertiaTensor_u.v[0][0], info.InertiaTensor_u.v[0][1], info.InertiaTensor_u.v[0][2], info.InertiaTensor_u.v[1][0], info.InertiaTensor_u.v[1][1], info.InertiaTensor_u.v[1][2], info.InertiaTensor_u.v[2][0], info.InertiaTensor_u.v[2][1], info.InertiaTensor_u.v[2][2]);
			printf("Still position: (%.8lf,%.8lf,%.8lf) => %.8lg\n",
				info.minGravPotential_vec.x, info.minGravPotential_vec.y, info.minGravPotential_vec.z, info.minGravPotential_u);
			printf("\n");
		}
		if (1) {
			fprintf(fp, "\tshapeInfo{\n");
			fprintf(fp, "\t\t\"%s\", %d, %s,\n", S.name, TN, info.isSolid ? "true" : "false");
			fprintf(fp, "\t\tvec3(%.8lg,%.8lg,%.8lg), vec3(%.8lg,%.8lg,%.8lg),\n",
				info.AABB_min.x, info.AABB_min.y, info.AABB_min.z, info.AABB_max.x, info.AABB_max.y, info.AABB_max.z);
			fprintf(fp, "\t\t%.8lg, vec3(%.8lg,%.8lg,%.8lg),\n",
				info.SA_or_V, info.CoM.x, info.CoM.y, info.CoM.z);
			fprintf(fp, "\t\tmat3(%.8lg,%.8lg,%.8lg,\n\t\t     %.8lg,%.8lg,%.8lg,\n\t\t     %.8lg,%.8lg,%.8lg),\n",
				info.InertiaTensor_u.v[0][0], info.InertiaTensor_u.v[0][1], info.InertiaTensor_u.v[0][2], info.InertiaTensor_u.v[1][0], info.InertiaTensor_u.v[1][1], info.InertiaTensor_u.v[1][2], info.InertiaTensor_u.v[2][0], info.InertiaTensor_u.v[2][1], info.InertiaTensor_u.v[2][2]);
			fprintf(fp, "\t\tvec3(%.8lf,%.8lf,%.8lf), %.8lg\n",
				info.minGravPotential_vec.x, info.minGravPotential_vec.y, info.minGravPotential_vec.z, info.minGravPotential_u);
			fprintf(fp, "\t},\n");

		}

		comps.insert(comps.end(), T.begin(), T.end());
	}

	fprintf(fp, "\
});\n\
\n");
	fclose(fp);
}




int main(int argc, char* argv[]) {
	//shapeInfos(); return 0;

	FILE* fp = fopen(argv[1], "wb");

#if 1
	for (unsigned i = 0; i < ParamSurfaces.size(); i++) {
		std::vector<triangle_3d> temp;
		auto S = ParamSurfaces[i];
		auto info = ParamSurfaceInfo::info[i];
		printf("%d - %s\n", i, S.name);
		S.param2trigs(temp);
		//temp = AdaptiveParametricSurfaceTriangulator_dist(S.P).triangulate_adaptive(S.u0, S.u1, S.v0, S.v1, 7, 7, 16, 0.01*pow(determinant(info.InertiaTensor_u), 1. / 6.), false, false);
		int TN = temp.size();
		if (0) {
			// separated files
			char s[64];
			int L = sprintf(s, "%03d_%s.stl", i, S.name);
			std::replace(&s[0], &s[L], ' ', '_');
			writeSTL(s, &temp[0], TN, "", STL_CCW);
			continue;
		}
		translateToCOM_shell(&temp[0], TN);
		scaleGyrationRadiusTo_shell(&temp[0], TN, 0.2);
		if (0) {
			// balanced position
			mat3 M = axis_angle(cross(info.minGravPotential_vec, vec3(0, 1e-8, -1)), acos(-info.minGravPotential_vec.z));
			for (int i = 0; i < TN; i++) temp[i].applyMatrix(M);
			translateShape_onPlane(&temp[0], TN);
		}
		translateShape(&temp[0], TN, vec3(i / 8, i % 8, 0.));
		comps.insert(comps.end(), temp.begin(), temp.end());
	}
#else
	auto S = ParamSurfaces[47];
	S.param2trigs(comps);
#endif

	writeSTL(fp, &comps[0], comps.size(), nullptr, STL_CCW);
	//writeSTL_recolor_normal(fp, &comps[0], comps.size(), nullptr, [](vec3 n) { return 0.5*n + vec3(.5); });
	fclose(fp);
	return 0;
}

