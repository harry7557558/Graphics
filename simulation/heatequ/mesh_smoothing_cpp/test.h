// included by main.cpp

#include <random>
float random() {
	return float(rand()) / float(RAND_MAX);
}

// Built-in states for testing
namespace BuiltInStates {

	State _genParametricPatch(std::function<vec3(float, float)> fun, int un, int vn) {
		// vertices
		std::vector<vec3> vertices((un+1)*(vn+1), vec3(0.0));
		for (int ui = 0; ui < un+1; ui++)
			for (int vi = 0; vi < vn+1; vi++) {
				float u = float(ui)/float(un);
				float v = float(vi)/float(vn);
				vertices[ui*(vn+1)+vi] = fun(u, v);
			}
		// faces
		std::vector<ivec3> faces;
		for (int ui = 0; ui < un; ui++)
			for (int vi = 0; vi < vn; vi++) {
				int i00 = ui*(vn+1)+vi;
				int i01 = ui*(vn+1)+(vi+1);
				int i10 = (ui+1)*(vn+1)+vi;
				int i11 = (ui+1)*(vn+1)+(vi+1);
				faces.push_back(ivec3(i00, i01, i11));
				faces.push_back(ivec3(i00, i11, i10));
			}
		return State(vertices, faces);
	}

	State unitCube() {
		return State(
			std::vector<vec3>({
				vec3(-1, -1, -1),
				vec3(-1, -1, 1),
				vec3(-1, 1, -1),
				vec3(-1, 1, 1),
				vec3(1, -1, -1),
				vec3(1, -1, 1),
				vec3(1, 1, -1),
				vec3(1, 1, 1)
				}),
			std::vector<ivec3>({
				ivec3(0, 1, 3), ivec3(0, 3, 2),
				ivec3(4, 5, 7), ivec3(4, 7, 6),
				ivec3(0, 4, 5), ivec3(0, 5, 1),
				ivec3(1, 5, 7), ivec3(1, 7, 3),
				ivec3(2, 6, 7), ivec3(2, 7, 3),
				ivec3(0, 4, 6), ivec3(0, 6, 2)
				})
		);
	}

	State plane(vec2 xyr, int xn, int yn, float noise) {
		auto fun = [=](float u, float v) {
			vec2 xy = xyr * (2.0f*vec2(u, v)-1.0f);
			float z = noise*(2.0f*random()-1.0f);
			return vec3(xy, z);
		};
		return _genParametricPatch(fun, xn, yn);
	}

	State states[] = {
		unitCube(),
		plane(vec2(1.5f, 1.0f), 15, 10, 0.2f)
	};

}