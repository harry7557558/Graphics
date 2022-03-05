// Template from https://www.opengl-tutorial.org/beginners-tutorials/

#include <stdio.h>
#include <stdlib.h>
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
#define FPS 60.0

#include "viewport.h"
#include "state.h"




// viewport-related global variables
GLFWwindow* window;
Viewport* viewport;
float iRz = 0.3f*PI;
float iRx = 0.2f*PI;
float iCameraDist = 5.0f;
bool mouseDown = false;
vec2 mousePos(-1, -1);



// Built-in states
namespace BuiltInStates {

	State circleDam(vec2 r, int nx, int ny, float k, vec2 circO, float circR) {
		return State(-r, r, nx, ny, k,
			[=](vec2 xy) { return length(xy-circO)<circR ? 1.0f : 0.0f; },
			[=](vec2 xy, float t) { return 0.0f; }
		);
	}

	State heaterCooler(vec2 r, int nx, int ny, float k, float hr, float pw) {
		return State(-r, r, nx, ny, k,
			[=](vec2 xy) { return 0.0f; },
			[=](vec2 xy, float t) { return length(xy)>hr ? pw*sign(xy.x+xy.y) : 0.0f; }
		);
	}

	State states[] = {
		circleDam(vec2(1.5, 1.0), 16, 12, 0.1f, vec2(-0.5, -0.3), 0.5f),
		circleDam(vec2(1.5, 1.0), 32, 24, 0.1f, vec2(-0.5, -0.3), 0.5f),
		circleDam(vec2(1.5, 1.0), 48, 36, 0.1f, vec2(-0.5, -0.3), 0.5f),
		circleDam(vec2(1.5, 1.0), 120, 90, 0.1f, vec2(-0.5, -0.3), 0.5f),
		heaterCooler(vec2(1.0, 0.8), 20, 20, 0.2f, 0.4f, 0.4f),
		heaterCooler(vec2(1.0, 0.8), 20, 20, 2.0f, 0.4f, 4.0f),
		heaterCooler(vec2(1.0, 0.8), 40, 40, 20.0f, 0.4f, 40.0f),
		heaterCooler(vec2(1.0, 0.8), 40, 40, 1000.0f, 0.4f, 2000.0f),
		circleDam(vec2(1.28, 0.8), 160, 100, 0.1f, vec2(-0.5, -0.3), 0.6f),
		circleDam(vec2(1.28, 0.8), 320, 200, 0.1f, vec2(-0.5, -0.3), 0.6f),
		circleDam(vec2(1.28, 0.8), 640, 400, 0.1f, vec2(-0.5, -0.3), 0.6f),
		heaterCooler(vec2(1.0, 0.8), 500, 400, 2.0f, 0.4f, 4.0f),
	};

}


int main(int argc, char* argv[]) {
	//testTimeComplexity(); exit(0);

	// states
	State state = BuiltInStates::states[3];

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
	window = glfwCreateWindow(iResolution.x, iResolution.y, "Implicit Mesh Smoothing Test", NULL, NULL);
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
	Slider sliderPv = Slider(vec2(10, 10), vec2(120, 25), 0.0f, 1.0f, 1.0f, 1.0f);
	Slider sliderStep = Slider(vec2(10, 30), vec2(120, 45), 0.0f, 1.0f, 0.0f, 0.5f);

	// mouse scroll
	glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
		viewport->mouseScroll(exp(0.04f*(float)yoffset));
	});

	// main loop
	do {

		// mouse drag
		dvec2 newMousePos;
		glfwGetCursorPos(window, &newMousePos.x, &newMousePos.y);
		vec2 mouseDelta = mousePos == vec2(-1) ? dvec2(0.0) : vec2(newMousePos) - mousePos;
		mousePos = newMousePos;
		int mouseState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
		if (mouseState == GLFW_PRESS) {
			sliderPv.mouseDown(mousePos);
			sliderStep.mouseDown(mousePos);
			mouseDown = true;
		}
		else if (mouseState == GLFW_RELEASE) {
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
		state.draw(viewport, vec3(1.0));
		viewport->finishDraw();
		sliderPv.draw(viewport);
		sliderStep.draw(viewport);

		glfwSwapBuffers(window);
		glfwPollEvents();

		// simulation
		float dt = 1.0f / float(FPS);
		/*for (int i = 0; i < (int)solvers.size(); i++)
			solvers[i].update(dt, 1);*/

			// pause
		std::this_thread::sleep_for(std::chrono::milliseconds(int(1000.0f*dt)));
	} while (glfwWindowShouldClose(window) == 0);

	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
	delete viewport;

	return 0;
}
