// test this header
#include "parametric_surface_adaptive_dist.h"

#include <chrono>

#include "modeling/generators/parametric/surfaces.h"
#include "ui/stl_encoder.h"

int main(int argc, char* argv[]) {
	auto S = ParamSurfaces[12];

	auto t0 = std::chrono::high_resolution_clock::now();
	std::vector<triangle> T = AdaptiveParametricSurfaceTriangulator_dist(S.P).triangulate_adaptive(S.u0, S.u1, S.v0, S.v1, 5, 5, 12, 0.01, true, false);
	double time_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
	printf("%lfms\n", 1000.*time_elapsed);

	writeSTL(argv[1], &T[0], T.size(), "", "cba");
	return 0;
}
