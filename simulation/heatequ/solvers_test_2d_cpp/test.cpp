// test the time complexity of the algorithm

#include <stdio.h>
#include <vector>
#include <chrono>

#include "state.h"
namespace Integrators {
#include "integrators.h"
}


void testTimeComplexity() {

	printf("[");

	for (int size = 16; size<=800; size = int(size*1.1)) {

		State state(vec2(-1.2, -1.0), vec2(1.2, 1.0), size, size, 0.1f,
			[](vec2 xy) { return length(xy-0.3f)<0.5f ? 1.0f : 0.0f; },
			[](vec2 xy, float t) { return 0.0f; }
		);

		Integrators::ImplicitEuler integrator(&state);

		auto t0 = std::chrono::high_resolution_clock::now();
		integrator.update(0.1f);
		auto t1 = std::chrono::high_resolution_clock::now();
		double dt = std::chrono::duration<double>(t1-t0).count();
		printf("(%d,%f),", size*size, dt);

	}

	printf("\b]\n");

	// https://www.desmos.com/calculator/llhd2fojry
	// should be O(N^1.5)
}
