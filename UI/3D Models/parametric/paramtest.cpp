// parametric surface test

#include <vector>
#include <stdio.h>

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"

#include "surfaces.h"



std::vector<triangle> comps;



int main() {
	FILE* fp = fopen("D:\\test.stl", "wb");

	for (int i = 0; i < 16; i++)
		ParamSurfaces[i].param2trigs(comps, 10.*vec3(i / 4, i % 4, 0.));

	writeSTL(fp, &comps[0], comps.size(), "bac");
	fclose(fp);
	return 0;
}

