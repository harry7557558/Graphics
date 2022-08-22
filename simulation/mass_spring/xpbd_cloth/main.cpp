// Template from https://www.opengl-tutorial.org/beginners-tutorials/

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#define PI 3.1415927f
#define iResolution ivec2(600, 600)
#define FPS 400.0

#include "viewport.h"
#include "state.h"
#include "solver.h"
#include "test.h"

GLFWwindow* window;
Viewport* viewport;
bool mouseDown = false;
vec2 mousePos(-1, -1);


int main(int argc, char* argv[]) {

	// Initialise GLFW
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW.\n");
		getchar(); return -1;
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);  // not resizable
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(iResolution.x, iResolution.y, "XPBD Cloth Test", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		getchar(); glfwTerminate(); return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW.\n");
		getchar(); glfwTerminate(); return -1;
	}

	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

	GLuint VertexArrayID = 0;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// GUI
	viewport = new Viewport(vec2(iResolution), 0.5f, vec3(0), -0.33f*PI, -0.17f*PI);
	Slider sliderPv = Slider(vec2(10, 10), vec2(160, 25), 0.0f, 1.0f, 1.0f, 1.0f);
	Slider sliderStep = Slider(vec2(10, 30), vec2(160, 45), 0.0f, 1.0f, 0.0f, 0.5f);

	// states
	State state_original = BuiltInStates::states[1];
	State state = state_original;
	std::vector<BaseSolver*> solvers({
		new EulerCromer(state),
		new XPBDNonIterative(state),
		new XPBDSolver(state),
		});

	// mouse action(s)
	glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
		viewport->mouseScroll(exp(0.04f*(float)yoffset));
	});

	// main loop
	do {

		// mouse drag
		dvec2 newMousePos;
		glfwGetCursorPos(window, &newMousePos.x, &newMousePos.y);
		vec2 mouseDelta = mousePos == vec2(-1) ? vec2(0.0) : vec2(newMousePos) - mousePos;
		mousePos = newMousePos;
		int mouseState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
		if (!mouseDown && mouseState == GLFW_PRESS) {
			sliderPv.mouseDown(mousePos);
			sliderStep.mouseDown(mousePos);
			mouseDown = true;
		}
		else if (mouseDown && mouseState == GLFW_RELEASE) {
			sliderPv.mouseUp();
			sliderStep.mouseUp();
			mouseDown = false;
		}
		if (mouseDown) {
			if (!sliderPv.mouseMove(mousePos) && !sliderStep.mouseMove(mousePos))
				viewport->mouseMove(mouseDelta);
		}

		// draw
		viewport->initDraw();
		for (int i = 0; i < (int)solvers.size(); i++)
			solvers[i]->state.draw(viewport, vec3(1 - i, 1.0, i));
		viewport->finishDraw();
		sliderPv.draw(viewport);
		sliderStep.draw(viewport);

		glfwSwapBuffers(window);
		glfwPollEvents();

		// simulation
		float dt = 1.0f / float(FPS);
		for (int i = 0; i < (int)solvers.size(); i++)
			solvers[i]->update(dt);
		checkGrad(&State::getStretchConstraint, solvers[1]->state,
			&solvers[1]->state.triangles[rand() % solvers[1]->state.triangles.size()],
			0.001f);
		printf("E = ");
		for (int i = 0; i < (int)solvers.size(); i++) {
			float e = solvers[i]->state.getEnergy();
			printf("%f%s", e, i + 1 == solvers.size() ? "\n" : ", ");
		}
		//exit(0);

		// pause
		std::this_thread::sleep_for(std::chrono::milliseconds(int(1000.0f*dt)));
	} while (glfwWindowShouldClose(window) == 0);

	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	delete viewport;

	return 0;
}
