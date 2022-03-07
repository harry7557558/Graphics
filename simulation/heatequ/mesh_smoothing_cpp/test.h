// included by main.cpp

#include <random>
float random() {
	return float(rand()) / float(RAND_MAX);
}

// Built-in states for testing
namespace BuiltInStates {

	// open parametric surface
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

	// half-open parametric surface
	State _genParametricCylinder(std::function<vec3(float, float)> fun, int un, int vn) {
		// vertices
		std::vector<vec3> vertices(un*(vn+1), vec3(0.0));
		for (int ui = 0; ui < un; ui++)
			for (int vi = 0; vi < vn+1; vi++) {
				float u = float(ui)/float(un-1);
				float v = float(vi)/float(vn);
				vertices[ui*(vn+1)+vi] = fun(u, v);
			}
		// faces
		std::vector<ivec3> faces;
		for (int ui = 0; ui < un; ui++)
			for (int vi = 0; vi < vn; vi++) {
				int i00 = ui*(vn+1)+vi;
				int i01 = ui*(vn+1)+(vi+1);
				int i10 = ((ui+1)%un)*(vn+1)+vi;
				int i11 = ((ui+1)%un)*(vn+1)+(vi+1);
				faces.push_back(ivec3(i00, i01, i11));
				faces.push_back(ivec3(i00, i11, i10));
			}
		return State(vertices, faces);
	}

	State _genParametricTorus(std::function<vec3(float, float)> fun, int un, int vn) {
		// vertices
		std::vector<vec3> vertices(un*vn, vec3(0.0));
		for (int ui = 0; ui < un; ui++)
			for (int vi = 0; vi < vn; vi++) {
				float u = float(ui)/float(un-1);
				float v = float(vi)/float(vn-1);
				vertices[ui*vn+vi] = fun(u, v);
			}
		// faces
		std::vector<ivec3> faces;
		for (int ui = 0; ui < un; ui++)
			for (int vi = 0; vi < vn; vi++) {
				int i00 = ui*vn+vi;
				int i01 = ui*vn+(vi+1)%vn;
				int i10 = ((ui+1)%un)*vn+vi;
				int i11 = ((ui+1)%un)*vn+(vi+1)%vn;
				faces.push_back(ivec3(i00, i01, i11));
				faces.push_back(ivec3(i00, i11, i10));
			}
		return State(vertices, faces);
	}

	// triangulated unit cube, normals not necessary CCW
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

	// plane with noise along z
	State plane(vec2 xyr, int xn, int yn, float noise) {
		auto fun = [=](float u, float v) {
			vec2 xy = xyr * (2.0f*vec2(u, v)-1.0f);
			float z = noise*(2.0f*random()-1.0f);
			return vec3(xy, z);
		};
		return _genParametricPatch(fun, xn, yn);
	}

	// elliptical cylinder with noise along normal
	State cylinder(float a, float b, float h, int un, int vn, bool seamed, float noise) {
		auto fun = [=](float u, float v) {
			float noi = noise * (2.0f*random()-1.0f) / sqrt(a*a+b*b);
			float x = (a+noi*b) * cos(2.0f*PI*u);
			float y = (b+noi*a) * sin(2.0f*PI*u);
			float z = h * (2.0f*v-1.0f);
			return vec3(x, y, z);
		};
		if (seamed) return _genParametricPatch(fun, un, vn);
		else return _genParametricCylinder(fun, un, vn);
	}

	// torus with noise along normal
	State torus(float r0, float r1, int un, int vn, bool u_seamed, bool v_seamed, float noise) {
		auto fun = [=](float u, float v) {
			float noi = noise*(2.0f*random()-1.0f);
			vec2 xy = (r0+r1*cos(2.0f*PI*u)) * vec2(cos(2.0f*PI*v), sin(2.0f*PI*v));
			float z = r1*sin(2.0f*PI*u);
			vec2 xyn = cos(2.0f*PI*u) * vec2(cos(2.0f*PI*v), sin(2.0f*PI*v));
			float zn = sin(2.0f*PI*u);
			return vec3(xy, z) + noi*vec3(xyn, zn);
		};
		if (u_seamed) {
			if (v_seamed) return _genParametricPatch(fun, un, vn);
			else return _genParametricCylinder([=](float u, float v) {return fun(v, u); }, vn, un);
		}
		else {
			if (v_seamed) return _genParametricCylinder(fun, un, vn);
			else return _genParametricTorus(fun, un, vn);
		}
	}

	// "flower" shaped closed torus with noise inside a cube
	State cersis(float r0, float r1, int n, int un, int vn, float noise) {
		auto fun = [=](float u, float v) {
			u *= 2.0f*PI, v *= 2.0f*PI;
			vec3 p = vec3(cos(u)*(r0+r1*cos(v)), sin(u)*(r0+r1*cos(v)), r1*sin(v));
			p += 0.5f * asin(sin(n*atan(p.y, p.x))) * vec3(cos(u), sin(u), 0);
			p.z *= 0.04f*(p.x*p.x+p.y*p.y) + 0.8f;
			p += noise * (2.0f*vec3(random(), random(), random())-1.0f);
			return p;
		};
		return _genParametricTorus(fun, un, vn);
	}

	State states[] = {
		unitCube(),
		plane(vec2(1.5f, 1.0f), 15, 10, 0.2f),
		cylinder(1.2f, 0.8f, 1.0f, 30, 10, true, 0.2f),
		cylinder(1.2f, 0.8f, 1.0f, 30, 10, false, 0.2f),
		torus(1.0f, 0.5f, 10, 30, false, false, 0.3f),
		torus(1.0f, 0.5f, 10, 30, false, true, 0.3f),
		cersis(1.0f, 0.3f, 3, 60, 10, 0.5f),
		cersis(1.0f, 0.3f, 5, 60, 10, 0.05f),
		cersis(1.0f, 0.3f, 5, 300, 50, 0.02f),
		cersis(1.0f, 0.3f, 5, 200, 100, 0.02f),
		torus(1.0f, 0.5f, 60, 200, false, true, 0.3f),
		torus(1.0f, 0.5f, 60, 200, false, false, 0.3f),
	};

}