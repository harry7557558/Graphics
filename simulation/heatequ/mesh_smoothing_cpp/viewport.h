#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include <vector>
#include <functional>


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


class Viewport {

	GLuint shaderProgram;
	GLuint vertexbuffer, normalbuffer, indicebuffer;
	mat4 transformMatrix;

public:

	vec2 resolution;
	float scale;
	vec3 center;
	float rx, rz;

	Viewport(vec2 resolution, float scale, vec3 center, float rx, float rz) {
		this->resolution = resolution;
		this->scale = scale;
		this->center = center;
		this->rx = rx, this->rz = rz;

		this->shaderProgram = createShaderProgram(
			// vertex shader
			R"""(#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
out vec3 fragNormal;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragNormal = (transformMatrix*vec4(vertexNormal,0)).xyz;
})""",
// fragment shader
R"""(#version 330
in vec3 fragNormal;
out vec3 fragColor;
uniform vec3 baseColor;
void main() {
    vec3 n = normalize(fragNormal);
    float amb = 1.0;
    float dif = abs(n.z);
    float spc = pow(abs(n.z), 40.0);
    fragColor = (0.2*amb+0.7*dif+0.1*spc)*baseColor;
})"""
);
		glGenBuffers(1, &this->vertexbuffer);
		glGenBuffers(1, &this->normalbuffer);
		glGenBuffers(1, &this->indicebuffer);
	}

	~Viewport() {
		// Cleanup VBO and shader
		glDeleteBuffers(1, &vertexbuffer);
		glDeleteBuffers(1, &normalbuffer);
		glDeleteBuffers(1, &indicebuffer);
		glDeleteProgram(this->shaderProgram);
	}

	void mouseMove(vec2 mouse_delta) {
		this->rx += 0.015f*mouse_delta.y;
		this->rz += 0.015f*mouse_delta.x;
	}
	void mouseScroll(float zoom) {
		if (zoom > 0.0)
			this->scale *= zoom;
	}

	// Call this when start drawing the 3D scene
	void initDraw() {
		glUseProgram(this->shaderProgram);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.04f, 0.04f, 0.04f, 1.0f);

		transformMatrix = glm::perspective(0.25f*PI, resolution.x/resolution.y, 0.1f, 100.0f) * transformMatrix;
		transformMatrix = glm::translate(transformMatrix, vec3(0.0, 0.0, -3.0/scale));
		transformMatrix = glm::rotate(transformMatrix, rx, vec3(1, 0, 0));
		transformMatrix = glm::rotate(transformMatrix, rz, vec3(0, 0, 1));
		transformMatrix = glm::translate(transformMatrix, -center);
	}
	// Call this when finished drawing the 3D scene and start drawing sliders
	void finishDraw() {
		glDisable(GL_DEPTH_TEST);
		this->transformMatrix = mat4(1.0);
	}

	void drawVBO(std::vector<vec3> vertices, std::vector<vec3> normals, std::vector<ivec3> indices, vec3 color) {
		assert(sizeof(vec3) == 12);
		assert(sizeof(ivec3) == 12);
		assert(sizeof(mat4) == 64);

		// vertices
		GLuint vertexPositionLocation = glGetAttribLocation(shaderProgram, "vertexPosition");
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3)*vertices.size(), &vertices[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(vertexPositionLocation);
		glVertexAttribPointer(
			vertexPositionLocation,  // attribute location
			3,  // size
			GL_FLOAT,  // type
			GL_FALSE,  // normalized?
			0,  // stride
			(void*)0  // array buffer offset
		);

		// colors
		GLuint vertexNormalLocation = glGetAttribLocation(shaderProgram, "vertexNormal");
		glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3)*normals.size(), &normals[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(vertexNormalLocation);
		glVertexAttribPointer(
			vertexNormalLocation,  // attribute location
			3,  // size
			GL_FLOAT,  // type
			GL_FALSE,  // normalized?
			0,  // stride
			(void*)0  // array buffer offset
		);

		// indices
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicebuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ivec3)*indices.size(), &indices[0], GL_STATIC_DRAW);

		// set uniform(s)
		GLuint matrixLocation = glGetUniformLocation(shaderProgram, "transformMatrix");
		glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, &transformMatrix[0][0]);
		GLuint colorLocation = glGetUniformLocation(shaderProgram, "baseColor");
		glUniform3fv(colorLocation, 1, (float*)&color);

		// draw
		glDrawElements(GL_TRIANGLES,
			3*(int)indices.size(),
			GL_UNSIGNED_INT,
			(void*)0);

		// clean-up
		glDisableVertexAttribArray(vertexPositionLocation);
		glDisableVertexAttribArray(vertexNormalLocation);
	}

	/*Draw an axes-aligned rectangle on screen
		p0, p1: in screen coordinate, top left (0, 0)
		color: RGB values between 0 and 1 */
	void drawRect(vec2 p0, vec2 p1, vec3 color) {
		p0 = 2.0f * vec2(p0.x/resolution.x, 1.0f-p0.y/resolution.y) - 1.0f;
		p1 = 2.0f * vec2(p1.x/resolution.x, 1.0f-p1.y/resolution.y) - 1.0f;
		this->drawVBO(
			std::vector<vec3>({ vec3(p0.x,p0.y,.5f), vec3(p0.x,p1.y,.5f), vec3(p1.x,p1.y,.5f), vec3(p1.x,p0.y,.5f) }),
			std::vector<vec3>({ vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), }),
			std::vector<ivec3>({ ivec3(0,1,2), ivec3(0,2,3) }),
			color
		);
	}
};


class Slider {
public:
	vec2 p0, p1;
	float v0, v1, vstep;
	float v;
	std::function<void()> callback;
	bool has_capture;

	Slider(
		vec2 p0, vec2 p1, float v0, float v1, float vstep, float v,
		std::function<void()> callback = nullptr
	) : p0(p0), p1(p1), v0(v0), v1(v1), vstep(vstep) {
		this->v = v;
		this->callback = callback;
		this->has_capture = false;
	}
	void setCallback(std::function<void()> callback) {
		this->callback = callback;
	}

	float getValue() const {
		return this->v;
	}

	void draw(Viewport *viewport) {
		vec2 dp = p1 - p0;
		float t = (v-v0)/(v1-v0);
		viewport->drawRect(p0, p1, vec3(0.25));
		viewport->drawRect(p0, vec2(p0.x+dp.x*t, p1.y), vec3(0.75));
	}

	// Call this when mouse down and pass mouse position
	void mouseDown(vec2 pos) {
		if (pos.x >= p0.x && pos.x < p1.x && pos.y >= p0.y && pos.y <= p1.y)
			this->has_capture = true;
		else this->has_capture = false;
	}
	// Call this when mouse up (optional)
	void mouseUp() {
		this->has_capture = false;
	}
	/* Call this when mouse click/drag to update value
	Returns True if mouse has effect, False otherwise */
	bool mouseMove(vec2 pos) {
		if (this->has_capture) {
			float dv = (v1-v0)*(pos.x-p0.x)/(p1.x-p0.x);
			if (vstep != 0.0)
				dv = round(dv/vstep)*vstep;
			v = clamp(v0+dv, v0, v1);
			if (callback != nullptr)
				callback();
			return true;
		}
		return false;
	}

};