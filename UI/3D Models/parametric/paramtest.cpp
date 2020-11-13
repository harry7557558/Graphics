// parametric surface test

#include <vector>
#include <stdio.h>

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"

#include "surfaces.h"



std::vector<triangle> comps;



int main(int argc, char* argv[]) {
	FILE* fp = fopen(argv[1], "wb");

#if 0
	for (int i = 0; i < 20; i++)
		ParamSurfaces[i].param2trigs(comps, 10.*vec3(i / 4, i % 4, 0.));
#else
	ParamSurfaces[19].param2trigs(comps);
#endif

	writeSTL(fp, &comps[0], comps.size(), nullptr, "bac");
	fclose(fp);
	return 0;
}

