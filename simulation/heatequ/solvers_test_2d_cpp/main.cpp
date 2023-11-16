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
#include <glm/gtx/color_space.hpp>
using namespace glm;


// forward declaration(s)
GLuint createShaderProgram(const char* vs_source, const char* fs_source);
mat4 calcTransformMatrix();


// constants
#define PI 3.1415927f
#define iResolution ivec2(800, 600)
#define FPS 20.0


// viewport-related global variables
GLFWwindow* window;
float iRz = 0.3f*PI;
float iRx = 0.2f*PI;
float iCameraDist = 5.0f;
bool mouseDown = false;
vec2 mousePos(-1, -1);


#include "state.h"
namespace Integrators {
#include "integrators.h"
}
#include "test.cpp"


// PDE solver
class Solver {
	const State *state;
	Integrators::Integrator *integrator;
	vec3 color;

public:

	Solver(Integrators::Integrator *integrator, float hue) {
		this->state = integrator->getState();
		this->integrator = integrator;
		this->color = glm::rgbColor(vec3(360.0f*hue, 0.5f, 1.0f));
	}

	void update(float deltaT, int stepCount) {
		auto t0 = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < stepCount; i++)
			integrator->update(deltaT / stepCount);
		auto t1 = std::chrono::high_resolution_clock::now();
		double time_elapsed = std::chrono::duration<double>(t1-t0).count();
		if (1) printf("%s %.1lfms\n", &(integrator->getName())[0], 1000.0*time_elapsed);
	}

	void draw(GLuint programID, mat4 transformMatrix,
		GLuint vertexbuffer, GLuint colorbuffer, GLuint indicebuffer) {
		state->draw(programID, transformMatrix, this->color,
			vertexbuffer, colorbuffer, indicebuffer);
	}
};


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

	// simulation states
	State state = BuiltInStates::states[9];
	int nsv = 4, nsi = 0;
	std::vector<Solver> solvers({
		//Solver(new Integrators::Euler(new State(state)), float(nsi++)/nsv),
		//Solver(new Integrators::Midpoint(new State(state)), float(nsi++)/nsv),
		//Solver(new Integrators::RungeKutta(new State(state)), float(nsi++)/nsv),
		Solver(new Integrators::ImplicitEuler(new State(state)), float(nsi++)/nsv),
		});

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
	window = glfwCreateWindow(iResolution.x, iResolution.y, "2D Heat Equation Solver Test", NULL, NULL);
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

	// Depth test
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	// Create and compile our GLSL program from the shaders
	GLuint programID = createShaderProgram(
		// vertex shader source
		R"""(#version 330 core
layout(location=0) in vec3 vertexPosition;
layout(location=1) in vec3 vertexColor;
out vec3 fragmentColor;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragmentColor = vertexColor;
}
)""",
// fragment shader source
R"""(#version 330 core
in vec3 fragmentColor;
out vec3 color;
void main() {
    color = fragmentColor;
}
)"""
);

	GLuint VertexArrayID = 0;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// create buffers
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	GLuint colorbuffer;
	glGenBuffers(1, &colorbuffer);
	GLuint indicebuffer;
	glGenBuffers(1, &indicebuffer);

	do {
		// draw
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glUseProgram(programID);

		mat4 transformMatrix = calcTransformMatrix();
		for (int i = 0; i < (int)solvers.size(); i++)
			solvers[i].draw(programID, transformMatrix,
				vertexbuffer, colorbuffer, indicebuffer);

		glfwSwapBuffers(window);
		glfwPollEvents();

		// simulation
		float dt = 1.0f / float(FPS);
		for (int i = 0; i < (int)solvers.size(); i++)
			solvers[i].update(dt, 1);

		// pause
		std::this_thread::sleep_for(std::chrono::milliseconds(int(1000.0f*dt)));
	} while (glfwWindowShouldClose(window) == 0);

	// Cleanup VBO and shader
	glDeleteBuffers(1, &vertexbuffer);
	glDeleteBuffers(1, &colorbuffer);
	glDeleteBuffers(1, &indicebuffer);
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();

	return 0;
}


// Compile shaders and create a shader program
GLuint createShaderProgram(const char* vs_source, const char* fs_source) {
	// Create the shaders
	GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
	GLint Result = GL_FALSE;
	int InfoLogLength;
	std::string errorMessage;

	// Vertex Shader
	glShaderSource(VertexShaderID, 1, &vs_source, NULL);
	glCompileShader(VertexShaderID);
	glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		errorMessage.resize(InfoLogLength + 1);
		glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &errorMessage[0]);
		printf("Vertex shader compile error.\n%s\n", &errorMessage[0]);
	}

	// Fragment Shader
	glShaderSource(FragmentShaderID, 1, &fs_source, NULL);
	glCompileShader(FragmentShaderID);
	glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		errorMessage.resize(InfoLogLength + 1);
		glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &errorMessage[0]);
		printf("Fragment shader compile error.\n%s\n", &errorMessage[0]);
	}

	// Link the program
	GLuint ProgramID = glCreateProgram();
	glAttachShader(ProgramID, VertexShaderID);
	glAttachShader(ProgramID, FragmentShaderID);
	glLinkProgram(ProgramID);
	glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0) {
		errorMessage.resize(InfoLogLength + 1);
		glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &errorMessage[0]);
		printf("Program linking error.\n%s\n", &errorMessage[0]);
	}

	glDetachShader(ProgramID, VertexShaderID);
	glDetachShader(ProgramID, FragmentShaderID);
	glDeleteShader(VertexShaderID);
	glDeleteShader(FragmentShaderID);
	return ProgramID;
}


// Handle viewport interaction + Return transformation matrix
mat4 calcTransformMatrix() {

	// get mouse pos and delta
	dvec2 newMousePos;
	glfwGetCursorPos(window, &newMousePos.x, &newMousePos.y);
	vec2 mouseDelta = mousePos == vec2(-1) ? dvec2(0.0) : vec2(newMousePos) - mousePos;
	mousePos = newMousePos;

	// mouse drag
	int mouseState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (mouseState == GLFW_PRESS) mouseDown = true;
	else if (mouseState == GLFW_RELEASE) mouseDown = false;
	if (mouseDown) {
		iRz = iRz + 0.01f*mouseDelta.x;
		iRx = clamp(iRx + 0.01f*mouseDelta.y, -1.57f, 1.57f);
	}

	// mouse scroll
	glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
		iCameraDist *= exp(-0.04f*(float)yoffset);
	});

	// matrix
	mat4 projection = glm::perspective(0.25f*PI, 4.0f / 3.0f, 0.1f, 100.0f);
	mat4 view = glm::lookAt(
		vec3(cos(iRx)*sin(iRz), cos(iRx)*cos(iRz), sin(iRx)) * iCameraDist, // camera position
		vec3(0, 0, 0), // look at
		vec3(0, 0, 1)  // head up
	);
	return projection * view;
}
