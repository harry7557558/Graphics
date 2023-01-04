#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#ifndef PIf
#define PIf 3.1415927f
#endif

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <functional>
#include <thread>

#include "elements.h"
#include "solver.h"


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
    GLuint vertexbuffer, normalbuffer, indicebuffer, colorbuffer;
    glm::mat4 transformMatrix;

public:

    glm::vec2 resolution;
    float scale;
    glm::vec3 center;
    float rx, rz;

    Viewport(glm::vec2 resolution, float scale, glm::vec3 center, float rx, float rz) {
        this->resolution = resolution;
        this->scale = scale;
        this->center = center;
        this->rx = rx, this->rz = rz;

        this->shaderProgram = createShaderProgram(
            // vertex shader
            R"""(#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec3 vertexColor;
out vec3 fragNormal;
out vec3 interpolatedColor;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragNormal = (transformMatrix*vec4(vertexNormal,0)).xyz;
    interpolatedColor = vertexColor;
})""",
// fragment shader
R"""(#version 330
in vec3 fragNormal;
in vec3 interpolatedColor;
out vec3 fragColor;
void main() {
    vec3 n = normalize(fragNormal);
    float amb = 1.0;
    float dif = abs(n.z);
    float spc = pow(abs(n.z), 40.0);
    fragColor = (0.2*amb+0.7*dif+0.1*spc)*interpolatedColor;
})"""
);
        glGenBuffers(1, &this->vertexbuffer);
        glGenBuffers(1, &this->normalbuffer);
        glGenBuffers(1, &this->indicebuffer);
        glGenBuffers(1, &this->colorbuffer);
    }

    ~Viewport() {
        // Cleanup VBO and shader
        glDeleteBuffers(1, &vertexbuffer);
        glDeleteBuffers(1, &normalbuffer);
        glDeleteBuffers(1, &indicebuffer);
        glDeleteBuffers(1, &colorbuffer);
        glDeleteProgram(this->shaderProgram);
    }

    void mouseMove(glm::vec2 mouse_delta) {
        this->rx += 0.015f * mouse_delta.y;
        this->rz += 0.015f * mouse_delta.x;
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

        transformMatrix = glm::perspective(0.25f * PIf, resolution.x / resolution.y, 0.1f / scale, 100.0f / scale) * transformMatrix;
        transformMatrix = glm::translate(transformMatrix, glm::vec3(0.0, 0.0, -3.0 / scale));
        transformMatrix = glm::rotate(transformMatrix, rx, glm::vec3(1, 0, 0));
        transformMatrix = glm::rotate(transformMatrix, rz, glm::vec3(0, 0, 1));
        transformMatrix = glm::translate(transformMatrix, -center);
    }
    // Call this when finished drawing the 3D scene and start drawing 2D stuff
    void finishDraw() {
        glDisable(GL_DEPTH_TEST);
        this->transformMatrix = glm::mat4(1.0);
    }

    void drawVBO(
        std::vector<glm::vec3> vertices,
        std::vector<glm::vec3> normals,
        std::vector<glm::ivec3> indices,
        std::vector<glm::vec3> colors
    ) {
        assert(sizeof(glm::vec3) == 12);
        assert(sizeof(glm::ivec3) == 12);
        assert(sizeof(glm::mat4) == 64);

        // vertices
        GLuint vertexPositionLocation = glGetAttribLocation(shaderProgram, "vertexPosition");
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexPositionLocation);
        glVertexAttribPointer(
            vertexPositionLocation,  // attribute location
            3,  // size
            GL_FLOAT,  // type
            GL_FALSE,  // normalized?
            0,  // stride
            (void*)0  // array buffer offset
        );

        // normals
        GLuint vertexNormalLocation = glGetAttribLocation(shaderProgram, "vertexNormal");
        glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(), &normals[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexNormalLocation);
        glVertexAttribPointer(
            vertexNormalLocation,  // attribute location
            3,  // size
            GL_FLOAT,  // type
            GL_FALSE,  // normalized?
            0,  // stride
            (void*)0  // array buffer offset
        );

        // values/colors
        GLuint vertexColorLocation = glGetAttribLocation(shaderProgram, "vertexColor");
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * colors.size(), &colors[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexColorLocation);
        glVertexAttribPointer(
            vertexColorLocation,  // attribute location
            3,  // size
            GL_FLOAT,  // type
            GL_FALSE,  // normalized?
            0,  // stride
            (void*)0  // array buffer offset
        );

        // indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicebuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::ivec3) * indices.size(), &indices[0], GL_STATIC_DRAW);

        // set uniform(s)
        GLuint matrixLocation = glGetUniformLocation(shaderProgram, "transformMatrix");
        glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, &transformMatrix[0][0]);

        // draw
        glDrawElements(GL_TRIANGLES,
            3 * (int)indices.size(),
            GL_UNSIGNED_INT,
            (void*)0);

        // clean-up
        glDisableVertexAttribArray(vertexPositionLocation);
        glDisableVertexAttribArray(vertexNormalLocation);
    }

    /*Draw an axes-aligned rectangle on screen
        p0, p1: in screen coordinate, top left (0, 0)
        color: RGB values between 0 and 1 */
    void drawRect(glm::vec2 p0, glm::vec2 p1, glm::vec3 color) {
        using glm::vec2, glm::vec3, glm::ivec3;
        p0 = 2.0f * vec2(p0.x / resolution.x, 1.0f - p0.y / resolution.y) - 1.0f;
        p1 = 2.0f * vec2(p1.x / resolution.x, 1.0f - p1.y / resolution.y) - 1.0f;
        this->drawVBO(
            std::vector<vec3>({ vec3(p0.x,p0.y,.5f), vec3(p0.x,p1.y,.5f), vec3(p1.x,p1.y,.5f), vec3(p1.x,p0.y,.5f) }),
            std::vector<vec3>({ vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), }),
            std::vector<ivec3>({ ivec3(0,1,2), ivec3(0,2,3) }),
            std::vector<vec3>({ color, color, color, color })
        );
    }
};



namespace RenderParams {
    const glm::ivec2 iResolution(600, 600);
    const double fps = 60.0;
    GLFWwindow* window;
    Viewport* viewport;
    bool mouseDown = false;
    glm::vec2 mousePos(-1, -1);
}

namespace ColorFunctions {
    using glm::vec3;
    vec3 RainbowC(float t) {
        return vec3(132.23, .39, -142.83) + vec3(-245.97, -1.4, 270.69) * t + vec3(755.63, 1.32, 891.31) * cos(vec3(.3275, 2.39, .3053) * t + vec3(-1.7461, -1.84, 1.4092));
    }
    vec3 TemperatureMapC(float t) {
        return vec3(.37, .89, 1.18) + vec3(.71, -2.12, -.94) * t + vec3(.26, 1.56, .2) * cos(vec3(5.2, 2.48, 8.03) * t + vec3(-2.51, -1.96, 2.87));
    }

}

void mainRender(DiscretizedStructure structure) {
    using glm::vec2, glm::vec3;

    // structure
    std::vector<vec3> vertices(structure.N, vec3(0));
    std::vector<vec3> normals(structure.N, vec3(0));
    std::vector<glm::ivec3> indices;
    // faces/normals
    auto ivec3Cmp = [](glm::ivec3 a, glm::ivec3 b) {
        return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
    };
    std::set<glm::ivec3, decltype(ivec3Cmp)> unique_indices(ivec3Cmp);
    vec3 minv(1e10f), maxv(-1e10f);
    for (int i = 0; i < structure.N; i++) {
        auto v = structure.X[i] + structure.U[i];
        vertices[i] = vec3(v.x, v.y, v.z);
        minv = glm::min(minv, vertices[i]);
        maxv = glm::max(maxv, vertices[i]);
    }
    ivec3 ts[MAX_SOLID_ELEMENT_TN];
    for (int i = 0; i < structure.M; i++) {
        int vn = structure.SE[i]->getNumTriangles();
        structure.SE[i]->getTriangles(ts);
        for (ivec3 t0 : ts) {
            glm::ivec3 t(t0.x, t0.y, t0.z);
            indices.push_back(t);
            unique_indices.insert(t);
            vec3 n = glm::cross(
                vertices[t.y] - vertices[t.x],
                vertices[t.z] - vertices[t.x]);
            normals[t.x] += n, normals[t.y] += n, normals[t.z] += n;
        }
    }
    indices = std::vector<glm::ivec3>(unique_indices.begin(), unique_indices.end());
    // stresses
    std::vector<float> values = structure.getTauXZ();
    std::vector<float> valuesSorted(structure.N, 0.0f);
    for (int i = 0; i < structure.N; i++)
        valuesSorted[i] = abs(values[i]);
    std::sort(valuesSorted.begin(), valuesSorted.end());
    float valueRange = valuesSorted[int(0.95 * structure.N)];
    printf("95th percentile value: %f\n", valueRange);
    std::vector<vec3> colors(structure.N, vec3(0));
    for (int i = 0; i < structure.N; i++) {
        float t = 0.5 + 0.5 * values[i] / valueRange;
        // float t = 0.5 + 0.5 * values[i] / 3.3;
        vec3 c = ColorFunctions::RainbowC(glm::clamp(t, 0.0f, 1.0f));
        colors[i] = glm::clamp(c, 0.0f, 1.0f);
    }

    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW.\n");
        return;
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);  // not resizable
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    RenderParams::window = glfwCreateWindow(
        RenderParams::iResolution.x, RenderParams::iResolution.y, "FEM Visualizer", NULL, NULL);
    if (RenderParams::window == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate(); return;
    }
    glfwMakeContextCurrent(RenderParams::window);

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW.\n");
        glfwTerminate(); return;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(RenderParams::window, GLFW_STICKY_KEYS, GL_TRUE);

    GLuint VertexArrayID = 0;
    glGenVertexArrays(1, &VertexArrayID);
    glBindVertexArray(VertexArrayID);

    // GUI
    RenderParams::viewport = new Viewport(
        vec2(RenderParams::iResolution),
        2.0 / max(max(maxv.x - minv.x, maxv.y - minv.y), maxv.z - minv.z),
        0.5f * (minv + maxv), -0.33f * PIf, -0.17f * PIf);

    // mouse action(s)
    glfwSetScrollCallback(RenderParams::window,
        [](GLFWwindow* window, double xoffset, double yoffset) {
            RenderParams::viewport->mouseScroll(exp(0.04f * (float)yoffset));
        });

    // main loop
    do {

        // mouse drag
        using RenderParams::mousePos, RenderParams::mouseDown;
        using RenderParams::viewport;
        glm::dvec2 newMousePos;
        glfwGetCursorPos(RenderParams::window, &newMousePos.x, &newMousePos.y);
        vec2 mouseDelta = mousePos == vec2(-1) ? vec2(0.0) : vec2(newMousePos) - mousePos;
        mousePos = newMousePos;
        int mouseState = glfwGetMouseButton(RenderParams::window, GLFW_MOUSE_BUTTON_LEFT);
        if (!mouseDown && mouseState == GLFW_PRESS) {
            mouseDown = true;
        }
        else if (mouseDown && mouseState == GLFW_RELEASE) {
            mouseDown = false;
        }
        if (mouseDown) {
            viewport->mouseMove(mouseDelta);
        }

        // draw
        viewport->initDraw();
        viewport->drawVBO(vertices, normals, indices, colors);
        viewport->finishDraw();

        glfwSwapBuffers(RenderParams::window);
        glfwPollEvents();

        float dt = 1.0f / float(RenderParams::fps);
        std::this_thread::sleep_for(std::chrono::milliseconds(int(1000.0f * dt)));
    } while (glfwWindowShouldClose(RenderParams::window) == 0);

    glDeleteVertexArrays(1, &VertexArrayID);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();
    delete RenderParams::viewport;
}
