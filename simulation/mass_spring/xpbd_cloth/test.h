// Gradient check

float checkGrad(
	int ndim, std::function<float(float*)> fun,
	float *x0, const float* calculated_grad, float eps = 0.001f
) {
	float maxerr = 0.0f;
	for (int i = 0; i < ndim; i++) {
		float xi = x0[i];
		x0[i] = xi + eps;
		float c1 = fun(x0);
		x0[i] = xi - eps;
		float c0 = fun(x0);
		x0[i] = xi;
		float ddx = (c1 - c0) / (2.0f*eps);
		float err = abs(calculated_grad[i]) < 1.0f ?
			ddx - calculated_grad[i] : ddx / calculated_grad[i] - 1.0f;
		printf("%.4f ", err);
		maxerr = max(maxerr, abs(err));
	}
	printf("\n");
	return maxerr;
}

float checkGrad(
	void (State::*constraint)(const TriangleConstraint *pc,
		float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]),
	State &state, const TriangleConstraint *pc,
	float eps = 0.001f
) {
	auto vertices = &state.vertices;
	// get points
	vec3 x0[3];
	for (int i = 0; i < 3; i++)
		x0[i] = (*vertices)[pc->vi[i]].x;
	// function to differentiate
	float ks, kd, c; int xi[3]; vec3 dcdx[3];
	auto fun = [&](float* x) -> float {
		for (int i = 0; i < 3; i++)
			(*vertices)[pc->vi[i]].x = ((vec3*)x)[i];
		(state.*constraint)(pc, &ks, &kd, &c, xi, dcdx);
		return c;
	};
	// calculated gradient
	fun((float*)&x0[0]);
	vec3 dcdx0[3] = { dcdx[0], dcdx[1], dcdx[2] };
	// check gradient
	float err = checkGrad(3*3, fun, (float*)&x0[0], (float*)&dcdx0[0], eps);
	// reset gradient
	fun((float*)&x0[0]);
	return err;
}

float checkGrad(
	void (State::*constraint)(const EdgeConstraint *pc,
		float *ks, float *kd, float *c, int xi[3], vec3 dcdx[3]),
	State &state, const EdgeConstraint *pc,
	float eps = 0.001f
) {
	auto vertices = &state.vertices;
	// get points
	vec3 x0[3];
	for (int i = 0; i < 2; i++)
		x0[i] = (*vertices)[pc->ai[i]].x;
	// function to differentiate
	float ks, kd, c; int xi[2]; vec3 dcdx[2];
	auto fun = [&](float* x) -> float {
		for (int i = 0; i < 2; i++)
			(*vertices)[pc->ai[i]].x = ((vec3*)x)[i];
		(state.*constraint)(pc, &ks, &kd, &c, xi, dcdx);
		return c;
	};
	// calculated gradient
	fun((float*)&x0[0]);
	vec3 dcdx0[2] = { dcdx[0], dcdx[1] };
	// check gradient
	float err = checkGrad(3*2, fun, (float*)&x0[0], (float*)&dcdx0[0], eps);
	// reset gradient
	fun((float*)&x0[0]);
	return err;
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
#if 1
				if ((ui + vi) % 2 == 1) {
					faces.push_back(ivec3(i00, i01, i11));
					faces.push_back(ivec3(i00, i11, i10));
				}
				else {
					faces.push_back(ivec3(i10, i11, i01));
					faces.push_back(ivec3(i10, i01, i00));
				}
#else
				faces.push_back(ivec3(i00, i01, i11));
				faces.push_back(ivec3(i00, i11, i10));
#endif
			}
		State state(vertices, faces, 1.0);
		state.vertices[0].inv_m = 0.0;
		state.vertices[vn].inv_m = 0.0;
		state.vertices[un*(vn+1)].inv_m = 0.0;
		state.vertices[un*(vn+1)+vn].inv_m = 0.0;
		return state;
	}

#if 0
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

#endif
	// plane with noise along z
	State plane(vec2 xyr, int xn, int yn) {
		auto fun = [=](float u, float v) {
			vec2 xy = xyr * (2.0f*vec2(u, v)-1.0f);
			float z = 1.0;
			return vec3(xy, z);
		};
		return _genParametricPatch(fun, xn, yn);
	}

	State states[] = {
		plane(vec2(1.5f, 0.5f), 3, 1),
		plane(vec2(1.5f, 0.25f), 6, 1),
		plane(vec2(1.5f, 0.1f), 15, 1),
		plane(vec2(1.5f, 0.5f), 6, 2),
		plane(vec2(2.0f, 2.0f), 3, 3),
		plane(vec2(2.0f, 2.0f), 4, 4),
		plane(vec2(2.0f, 2.0f), 8, 8),
		plane(vec2(2.0f, 2.0f), 16, 16),
		plane(vec2(1.5f, 1.0f), 15, 10),
	};

}