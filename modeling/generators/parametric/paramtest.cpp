// parametric surface test

#include <vector>
#include <stdio.h>

#include "numerical/geometry.h"
#include "UI/stl_encoder.h"

#include "surfaces.h"



std::vector<triangle> comps;



int main(int argc, char* argv[]) {
	FILE* fp = fopen(argv[1], "wb");

#if 1
	for (unsigned i = 0; i < ParamSurfaces.size(); i++) {
		std::vector<triangle> temp;
		ParamSurfaces[i].param2trigs(temp);
		int TN = temp.size();
		translateToCOM_shell(&temp[0], TN);
		scaleGyrationRadiusTo_shell(&temp[0], TN, 0.2);
		translateShape(&temp[0], TN, vec3(i / 4, i % 4, 0.));
		comps.insert(comps.end(), temp.begin(), temp.end());
	}
#else
	ParamSurfaces[30].param2trigs(comps);
#endif

	writeSTL(fp, &comps[0], comps.size(), nullptr, "bac");
	fclose(fp);
	return 0;
}

