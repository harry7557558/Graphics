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
	for (int i = 0; i < 20; i++) {
		std::vector<triangle> temp;
		ParamSurfaces[i].param2trigs(temp);
		int TN = temp.size();
		translateToCOM_shell(&temp[0], TN);
		scaleGyrationRadiusTo_shell(&temp[0], TN, 0.2);
		for (int u = 0; u < TN; u++)
			temp[u].translate(vec3(i / 4, i % 4, 0.));
		comps.insert(comps.end(), temp.begin(), temp.end());
	}
#else
	ParamSurfaces[19].param2trigs(comps);
#endif

	writeSTL(fp, &comps[0], comps.size(), nullptr, "bac");
	fclose(fp);
	return 0;
}

