#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glm/gtc/matrix_transform.hpp"

#ifndef PIf
#define PIf 3.1415927f
#endif

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <functional>
#include <algorithm>
#include <thread>

#include "elements.h"
#include "solver.h"

#if SUPPRESS_ASSERT
#undef assert
#define assert(x) 0
#endif


const char VERTEX_SHADER_SOURCE[] = R"""(#version 300 es
precision highp float;
in vec3 vertexPosition;
in vec3 vertexNormal;
in vec4 vertexColor;
in vec2 vertexTexcoord;
out vec3 fragNormal;
out vec2 fragTexcoord;
out vec4 fragColor;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragNormal = -(transformMatrix*vec4(vertexNormal,0)).xyz;
    // fragNormal = vertexNormal;
    fragColor = vertexColor;
    fragTexcoord = vertexTexcoord;
})""";

const char FRAGMENT_SHADER_SOURCE[] = R"""(#version 300 es
precision highp float;
in vec3 fragNormal;
in vec2 fragTexcoord;
in vec4 fragColor;
uniform sampler2D fragTexture;
out vec4 glFragColor;
void main() {
    vec3 col = fragColor.xyz;
    if (dot(col, vec3(1)) < 0.0)
        col = texture(fragTexture, fragTexcoord).xyz;
    if (fragNormal == vec3(0)) {
        glFragColor = vec4(col, 1.0);
        return;
    }
    col = pow(col, vec3(2.0));
    vec3 n = normalize(fragNormal);
    float amb = 1.0+0.3*max(-n.z,0.);
    float dif = dot(n,normalize(vec3(-0.5,-0.5,1)));
    float spc = pow(max(n.z,0.), 10.0);
    col *= 0.6*amb+0.4*max(dif,0.)+0.1*max(-dif,0.)+0.2*spc;
    col += 0.025+0.025*dif;
    glFragColor = vec4(pow(0.8*col, vec3(1.0/2.2)), 1.0);
})""";


// compile shaders into a program
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
    if (InfoLogLength > 1) {
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


GLuint createSampleTexture() {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    int color = 0x00ffffff;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &color);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    return tex;
}


class Viewport;

namespace RenderParams {
    glm::ivec2 iResolution(600, 500);
    const double fps = 60.0;
    GLFWwindow* window;
    GLuint vertexArrayID = 0;
    Viewport* viewport;
    bool mouseDown = false;
    glm::vec2 mousePos(-1, -1);
}


class Viewport {

    GLuint shaderProgram;
    GLuint vertexbuffer, normalbuffer, indicebuffer, colorbuffer, texcoordbuffer;
    glm::mat4 transformMatrix;

public:

    float scale;
    glm::vec3 center;
    float rx, rz;
    bool renderNeeded;

    Viewport(float scale, glm::vec3 center, float rx, float rz)
        : renderNeeded(true) {
        this->scale = scale;
        this->center = center;
        this->rx = rx, this->rz = rz;
        this->shaderProgram = createShaderProgram(
            VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
        glGenBuffers(1, &this->vertexbuffer);
        glGenBuffers(1, &this->normalbuffer);
        glGenBuffers(1, &this->indicebuffer);
        glGenBuffers(1, &this->colorbuffer);
        glGenBuffers(1, &this->texcoordbuffer);
    }

    ~Viewport() {
        // Cleanup VBO and shader
        glDeleteBuffers(1, &vertexbuffer);
        glDeleteBuffers(1, &normalbuffer);
        glDeleteBuffers(1, &indicebuffer);
        glDeleteBuffers(1, &colorbuffer);
        glDeleteBuffers(1, &texcoordbuffer);
        glDeleteProgram(this->shaderProgram);
    }

    // interactions
    void mouseClick() {
        this->renderNeeded = true;
    }
    void mouseMove(glm::vec2 mouse_delta) {
        this->rx -= 0.015f * mouse_delta.y;
        this->rz += 0.015f * mouse_delta.x;
        this->renderNeeded = true;
    }
    void mouseScroll(float yoffset) {
        this->scale *= exp(0.04f * yoffset);
        this->renderNeeded = true;
    }

    // Clear the screen and setup projection matrix
    void initDraw3D() {
        glUseProgram(this->shaderProgram);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // glDisable(GL_DEPTH_TEST);
        // glEnable(GL_BLEND);
        // glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);

        glm::vec2 res = RenderParams::iResolution;
        transformMatrix = glm::perspective(0.2f * PIf, res.x / res.y, 0.1f / scale, 100.0f / scale);
        transformMatrix = glm::translate(transformMatrix, glm::vec3(0.0, 0.0, -3.0 / scale));
        transformMatrix = glm::rotate(transformMatrix, rx, glm::vec3(1, 0, 0));
        transformMatrix = glm::rotate(transformMatrix, rz, glm::vec3(0, 1, 0));
        transformMatrix = glm::translate(transformMatrix, -center);
    }
    // Setup transformation matrix for 2D drawing
    void initDraw2D() {
        glDisable(GL_DEPTH_TEST);
        glm::vec2 res = RenderParams::iResolution;
        transformMatrix = glm::mat4(
            2.0f / res.x, 0, 0, 0,
            0, 2.0f / res.y, 0, 0,
            0, 0, -1, 0,
            -1, -1, 0, 1
        );
    }

    void drawVBO(
        std::vector<glm::vec3> vertices,
        std::vector<glm::vec3> normals,
        std::vector<glm::ivec3> indices,
        std::vector<glm::vec4> colors,
        std::vector<glm::vec2> texcoords,
        GLuint texture
    ) {
        assert(sizeof(glm::vec3) == 12);
        assert(sizeof(glm::ivec3) == 12);
        assert(sizeof(glm::vec4) == 16);
        assert(sizeof(glm::mat4) == 64);
        glUseProgram(this->shaderProgram);

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

        // vertex colors
        GLuint vertexColorLocation = glGetAttribLocation(shaderProgram, "vertexColor");
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * colors.size(), &colors[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexColorLocation);
        glVertexAttribPointer(
            vertexColorLocation,  // attribute location
            4,  // size
            GL_FLOAT,  // type
            GL_FALSE,  // normalized?
            0,  // stride
            (void*)0  // array buffer offset
        );

        // texcoords
        GLuint vertexTexcoordLocation = glGetAttribLocation(shaderProgram, "vertexTexcoord");
        glBindBuffer(GL_ARRAY_BUFFER, texcoordbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * texcoords.size(), &texcoords[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexTexcoordLocation);
        glVertexAttribPointer(
            vertexTexcoordLocation,
            2,  // size
            GL_FLOAT,  // type
            GL_FALSE,  // normalized?
            0,  // stride
            (void*)0  // array buffer offset
        );

        // indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicebuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::ivec3) * indices.size(), &indices[0], GL_STATIC_DRAW);

        // set uniform(s)
        GLuint location = glGetUniformLocation(shaderProgram, "transformMatrix");
        glUniformMatrix4fv(location, 1, GL_FALSE, &transformMatrix[0][0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glUniform1i(glGetUniformLocation(shaderProgram, "fragTexture"), 0);

        // draw
        glDrawElements(GL_TRIANGLES,
            3 * (int)indices.size(),
            GL_UNSIGNED_INT,
            (void*)0);

        // clean-up
        glDisableVertexAttribArray(vertexPositionLocation);
        glDisableVertexAttribArray(vertexNormalLocation);
        glDisableVertexAttribArray(vertexColorLocation);
        glDisableVertexAttribArray(vertexTexcoordLocation);
    }

    void drawLinesVBO(
        std::vector<glm::vec3> vertices,
        std::vector<glm::vec3> normals,
        std::vector<glm::ivec2> indices,
        std::vector<glm::vec4> colors
    ) {
        glUseProgram(this->shaderProgram);

        // vertices
        GLuint vertexPositionLocation = glGetAttribLocation(shaderProgram, "vertexPosition");
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexPositionLocation);
        glVertexAttribPointer(vertexPositionLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        // normals
        GLuint vertexNormalLocation = glGetAttribLocation(shaderProgram, "vertexNormal");
        glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * normals.size(), &normals[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexNormalLocation);
        glVertexAttribPointer(vertexNormalLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        // values/colors
        GLuint vertexColorLocation = glGetAttribLocation(shaderProgram, "vertexColor");
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * colors.size(), &colors[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexColorLocation);
        glVertexAttribPointer(vertexColorLocation, 4, GL_FLOAT, GL_FALSE, 0, nullptr);

        // indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicebuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::ivec2) * indices.size(), &indices[0], GL_STATIC_DRAW);

        // set uniform(s)
        GLuint location = glGetUniformLocation(shaderProgram, "transformMatrix");
        glm::mat4 m = transformMatrix; m[3][2] -= 4e-5f;
        glUniformMatrix4fv(location, 1, GL_FALSE, &m[0][0]);

        // draw
        glDrawElements(GL_LINES, 2 * (int)indices.size(), GL_UNSIGNED_INT, nullptr);

        // clean-up
        glDisableVertexAttribArray(vertexPositionLocation);
        glDisableVertexAttribArray(vertexNormalLocation);
        glDisableVertexAttribArray(vertexColorLocation);
    }

};




struct RenderModel {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> colors;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::ivec3> indicesF;
    std::vector<glm::ivec2> indicesE;
    GLuint texture;
} renderModel;


bool initWindow() {
    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW.\n");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Open a window and create its OpenGL context
    RenderParams::window = glfwCreateWindow(
        RenderParams::iResolution.x, RenderParams::iResolution.y, "GLFW Window", NULL, NULL);
    if (RenderParams::window == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate(); return false;
    }
    glfwMakeContextCurrent(RenderParams::window);

#ifndef __EMSCRIPTEN__
    if (glewInit() != GLEW_OK) {
        glfwTerminate();
        return false;
    }
#endif

    glGenVertexArrays(1, &RenderParams::vertexArrayID);
    glBindVertexArray(RenderParams::vertexArrayID);

    RenderParams::viewport = new Viewport(
        0.8f, glm::vec3(0.0f), 0.02f*PI, -0.05f*PI);
    return true;
}


std::function<void()> loop;
void main_loop() { loop(); }

void mainGUI(void (*callback)(void)) {

    renderModel.texture = createSampleTexture();

    // mouse action(s)
    glfwSetScrollCallback(RenderParams::window,
        [](GLFWwindow* window, double xoffset, double yoffset) {
            RenderParams::viewport->mouseScroll((float)yoffset);
        });

    // main loop
    loop = [&] {
        callback();

        // mouse drag
        using RenderParams::mousePos, RenderParams::mouseDown;
        using RenderParams::viewport;
        glm::dvec2 newMousePos;
        glfwGetCursorPos(RenderParams::window, &newMousePos.x, &newMousePos.y);
        newMousePos.y = RenderParams::iResolution.y - 1 - newMousePos.y;
        vec2 mouseDelta = mousePos == vec2(-1) ? vec2(0.0) : vec2(newMousePos) - mousePos;
        mousePos = newMousePos;
        int mouseState = glfwGetMouseButton(RenderParams::window, GLFW_MOUSE_BUTTON_LEFT);
        if (!mouseDown && mouseState == GLFW_PRESS) {
            mouseDown = true;
            viewport->mouseClick();
        }
        else if (mouseDown && mouseState == GLFW_RELEASE) {
            mouseDown = false;
            viewport->renderNeeded = true;
        }
        if (mouseDown) {
            viewport->mouseMove(mouseDelta);
        }
        if (!viewport->renderNeeded) return;

        glViewport(0, 0, RenderParams::iResolution.x, RenderParams::iResolution.y);

        // draw
        viewport->initDraw3D();
        if (!renderModel.vertices.empty()) {
            glm::vec4 colorsF(0.9, 0.9, 0.9, 1);
            glm::vec4 colorsE(0.6, 0.6, 0.6, 1);
            viewport->drawVBO(
                renderModel.vertices, renderModel.normals, renderModel.indicesF,
                !renderModel.colors.empty() ? renderModel.colors :
                    std::vector<glm::vec4>(renderModel.vertices.size(),
                        renderModel.texcoords.empty() ? colorsF : -colorsF),
                !renderModel.texcoords.empty() ? renderModel.texcoords :
                    std::vector<glm::vec2>(renderModel.vertices.size(), vec2(0.0f)),
                renderModel.texture);
            if (!renderModel.indicesE.empty()) viewport->drawLinesVBO(
                renderModel.vertices, renderModel.normals, renderModel.indicesE,
                std::vector<glm::vec4>(renderModel.vertices.size(), colorsE));
        }
        // axes
        viewport->drawLinesVBO(
            { vec3(0), vec3(1.5, 0, 0),
                vec3(0), vec3(0, 1.5, 0),
                vec3(0), vec3(0, 0, 1.5) },
            std::vector<vec3>(6, vec3(0)),
            { ivec2(0, 1), ivec2(2, 3), ivec2(4, 5) },
            { vec4(1, 0, 0, 1), vec4(1, 0, 0, 1),
                vec4(0, 0.5, 0, 1), vec4(0, 0.5, 0, 1),
                vec4(0, 0, 1, 1), vec4(0, 0, 1, 1) }
        );

        glfwSwapBuffers(RenderParams::window);
        glfwPollEvents();

        if (!mouseDown)
            viewport->renderNeeded = false;
    };

    emscripten_set_main_loop(main_loop, 0, true);
    main_loop();
    return;



    // Close OpenGL window and terminate GLFW
    glDeleteVertexArrays(1, &RenderParams::vertexArrayID);
    glfwTerminate();
    delete RenderParams::viewport;
}
