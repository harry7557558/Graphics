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
in float vertexValue;
in vec4 vertexColor;
out vec3 fragNormal;
out vec4 interpolatedColor;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragNormal = -(transformMatrix*vec4(vertexNormal,0)).xyz;
    // fragNormal = vertexNormal;
    interpolatedColor = vertexColor;
})""";

const char FRAGMENT_SHADER_SOURCE[] = R"""(#version 300 es
precision highp float;
in vec3 fragNormal;
in vec4 interpolatedColor;
out vec3 fragColor;  // negative rgb for value<0-1> colormap
uniform float maxValue;  // colormap, positive for smooth and negative for steps
uniform float colorRemapK;  // negative for slider, positive for object
uniform float brightness;  // multiply color by this number
float log10(float x) { return log(x)/log(10.); }
float remap(float t, float k) { return pow(t,k) / (pow(t,k) + pow(1.-t,k)); }
void main() {
    vec3 n = fragNormal==vec3(0) ? vec3(0): normalize(fragNormal);
    float amb = 1.0+0.3*max(-n.z,0.);
    float dif = max(n.z,0.);
    float spc = pow(max(n.z,0.), 40.0);
    vec3 col = interpolatedColor.xyz;
    if (interpolatedColor.x < 0.0) {
        float maxv = abs(maxValue);
        float t = clamp(interpolatedColor.w, 0., 1.);
        float tm = colorRemapK<0. ? remap(t, -1./colorRemapK) : t;
        float v = mix(-maxv, maxv, tm);
        if (maxValue < 0.0 && v != 0.0) {
            float s = sign(v); v = abs(v);
            float k = ceil(log10(v)) - 1.;
            float u = log10(v) - k;
            float w = u < log10(2.) ? 1.5 :
                u < log10(5.) ? 3.5 : 7.5;
            if (pow(10.,k)*w > maxv)
                w = 0.5*(0.5*w + maxv*pow(0.1,k));
            v = s * pow(10.,k) * w;
        }
        tm = clamp(0.5+0.5*v/maxv, 0., 1.);
        t = remap(tm, abs(colorRemapK));
        col = vec3(132.23,.39,-142.83)+vec3(-245.97,-1.4,270.69)*t+vec3(755.63,1.32,891.31)*cos(vec3(.3275,2.39,.3053)*t+vec3(-1.7461,-1.84,1.4092));
        if (isnan(dot(col,col))) col = vec3(0,1,0);
    }
    fragColor = brightness*(0.2*amb+0.7*dif+0.1*spc)*col;
})""";

const char VERTEX_SHADER_SOURCE_FONT[] = R"""(#version 300 es
precision highp float;
in vec3 vertexPosition;
in vec2 vertexUv;
out vec2 fragmentUv;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragmentUv = vertexUv;
})""";

const char FRAGMENT_SHADER_SOURCE_FONT[] = R"""(#version 300 es
precision highp float;
in vec2 fragmentUv;
out vec3 fragColor;
uniform int i;
uniform vec3 fontColor;
void main() {
    ivec4 c;
if (i==0) c=ivec4(0,1178739736,1113739850,6180); // 0
if (i==1) c=ivec4(0,136845320,134744072,15880); // 1
if (i==2) c=ivec4(0,37896764,1075843084,32320); // 2
if (i==3) c=ivec4(0,37896764,1107427868,15426); // 3
if (i==4) c=ivec4(0,605293572,75383876,1028); // 4
if (i==5) c=ivec4(0,1077952638,33686140,15426); // 5
if (i==6) c=ivec4(0,1077944348,1111638652,15426); // 6
if (i==7) c=ivec4(0,67240574,134743044,2056); // 7
if (i==8) c=ivec4(0,1111638588,1111638588,15426); // 8
if (i==9) c=ivec4(0,1111638588,33686078,14340); // 9
if (i==10) c=ivec4(0,0,0,6168); // .
if (i==11) c=ivec4(0,1111228416,1077968450,15426); // e
if (i==12) c=ivec4(0,134742016,134774536,8); // +
if (i==13) c=ivec4(0,0,15360,0); // -
if (i==14) c=ivec4(0,1212088320,1145324612,14404); // σ
if (i==15) c=ivec4(0,276692992,269488144,3088); // τ
if (i==16) c=ivec4(0,1111621632,1111638594,14918); // u
if (i==17) c=ivec4(0,1077952638,1077952636,16448); // F
if (i==18) c=ivec4(0,1111621632,605558820,16962); // x
if (i==19) c=ivec4(0,1111621632,641876546,1006764570); // y
if (i==20) c=ivec4(0,41811968,537921540,32320); // z
if (i==21) c=ivec4(0,1111621632,606348354,6168); // v
if (i==22) c=ivec4(0,1232470016,1229539657,18761); // m
if (i==23) c=ivec4(0,2081427472,269488144,3088); // t
if (i==24) c=ivec4(0,1111228416,1077952576,15426); // c
    ivec2 uv = ivec2(vec2(8,16)*fragmentUv);
    if (uv.x >= 0 && uv.x < 8 && uv.y >= 0 && uv.y < 16) {
        int k = 8 * (15-uv.y) + (7-uv.x);
        int b = k/32==0 ? c.x : k/32==1 ? c.y : k/32==2 ? c.z : c.w;
        b = (b >> (k%32)) & 1;
        if (b==0) discard;
        fragColor = fontColor;
    }
    else discard;
}
)""";

namespace FontCharacters {
    enum FontCharacter {
        _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, dot, e, plus, minus,
        sigma, tau, u, F, x, y, z, v, m, t, c,
        other
    };
}
typedef std::vector<FontCharacters::FontCharacter> FontString;


// round(3.1415926, 3) == 3.14
// round(-0.0012345, 2) == -0.0012
float roundSigfig(float v, float sigfig) {
    if (v == 0.0f) return v;
    if (v < 0.0f) return -roundSigfig(-v, sigfig);
    float k = pow(10.0f, sigfig - ceil(log10(v)));
    return round(v * k) / k;
}
// round(12.3456, 3, 2.0) == 12.35
// round(23.4567, 3, 2.0) == 23.5
float roundSigfigHalf(float v, float sigfig, float cutoff) {
    if (v < 0.0f) return -roundSigfigHalf(-v, sigfig, cutoff);
    sigfig += fmod(log10(v) + 100.f, 1.0f) < log10(cutoff) ? 1.0f : 0.0f;
    return roundSigfig(v, sigfig);
}
// one sigfig, last digit must be in {1,2,5}
float round125(float v) {
    if (v == 0.0f) return v;
    if (v < 0.0f) return -round125(-v);
    float k = ceil(log10(v)) - 1.0f;
    float u = log10(v) - k;
    float w = u < log10(1.5f) ? 1.0f :
        u < log10(3.5f) ? 2.0f :
        u < log10(7.5f) ? 5.0f : 10.0f;
    return pow(10.0f, k) * w;
}



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



class Viewport;

namespace RenderParams {
    const glm::ivec2 iResolution(600, 500);
    const double fps = 60.0;
    GLFWwindow* window;
    GLuint vertexArrayID = 0;
    Viewport* viewport;
    bool mouseDown = false;
    glm::vec2 mousePos(-1, -1);
}


// toggles, buttons, sliders, etc.
struct WindowInput {
    // interaction
    bool captured = false;
    virtual bool isHover() = 0;
    virtual void mouseClick() {}
    virtual void mouseMove(glm::vec2 mouse_delta) {}
    virtual void mouseScroll(float yoffset) {}
    // rendering
    virtual void render() = 0;
};


class Viewport {

    GLuint shaderProgram, shaderProgramFont;
    GLuint vertexbuffer, normalbuffer, indicebuffer, colorbuffer;
    glm::mat4 transformMatrix;

    std::vector<WindowInput*> windowInputs;

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
        this->shaderProgramFont = createShaderProgram(
            VERTEX_SHADER_SOURCE_FONT, FRAGMENT_SHADER_SOURCE_FONT);
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
        glDeleteProgram(this->shaderProgramFont);
    }

    // add a button/toggle/slider
    void addWindowInput(WindowInput* wi) {
        this->windowInputs.push_back(wi);
    }

    // interactions
    void mouseClick() {
        for (WindowInput* mi : windowInputs) {
            if (mi->isHover()) {
                mi->mouseClick();
                mi->captured = true;
            }
            else mi->captured = false;
        }
        this->renderNeeded = true;
    }
    void mouseMove(glm::vec2 mouse_delta) {
        for (WindowInput* mi : windowInputs) {
            if (mi->captured) {
                if (RenderParams::mouseDown) {
                    mi->mouseMove(mouse_delta);
                    return;
                }
                else mi->captured = false;
            }
        }
        this->rx -= 0.015f * mouse_delta.y;
        this->rz += 0.015f * mouse_delta.x;
        this->renderNeeded = true;
    }
    void mouseScroll(float yoffset) {
        for (WindowInput* mi : windowInputs)
            if (mi->isHover()) {
                mi->mouseScroll(yoffset);
                return;
            }
        this->scale *= exp(0.04f * yoffset);
        this->renderNeeded = true;
    }

    // Clear the screen and setup projection matrix
    void initDraw3D() {
        glUseProgram(this->shaderProgram);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.04f, 0.04f, 0.04f, 1.0);

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);

        glm::vec2 res = RenderParams::iResolution;
        transformMatrix = glm::perspective(0.25f * PIf, res.x / res.y, 0.1f / scale, 100.0f / scale);
        transformMatrix = glm::translate(transformMatrix, glm::vec3(0.0, 0.0, -3.0 / scale));
        transformMatrix = glm::rotate(transformMatrix, rx, glm::vec3(1, 0, 0));
        transformMatrix = glm::rotate(transformMatrix, rz, glm::vec3(0, 0, 1));
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
    // Draw window input accessories
    void drawWindowInputs() {
        for (WindowInput* wi : windowInputs)
            wi->render();
    }

    /*VBO drawing
        Precombute vertices, normals (not necessarily normalized), indices, and colors
        Colors: RGB must be non-negative; Pass vec4(-1,-1,-1,value<0-1>) for "heatmap"
    */
    void drawVBO(
        std::vector<glm::vec3> vertices,
        std::vector<glm::vec3> normals,
        std::vector<glm::ivec3> indices,
        std::vector<glm::vec4> colors,
        float maxValue = 1.0f, float colorRemapK = 1.0f, float brightness = 1.0f
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

        // values/colors
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

        // indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicebuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::ivec3) * indices.size(), &indices[0], GL_STATIC_DRAW);

        // set uniform(s)
        GLuint location = glGetUniformLocation(shaderProgram, "transformMatrix");
        glUniformMatrix4fv(location, 1, GL_FALSE, &transformMatrix[0][0]);
        location = glGetUniformLocation(shaderProgram, "maxValue");
        glUniform1f(location, maxValue);
        location = glGetUniformLocation(shaderProgram, "colorRemapK");
        glUniform1f(location, colorRemapK);
        location = glGetUniformLocation(shaderProgram, "brightness");
        glUniform1f(location, brightness);

        // draw
        glDrawElements(GL_TRIANGLES,
            3 * (int)indices.size(),
            GL_UNSIGNED_INT,
            (void*)0);

        // clean-up
        glDisableVertexAttribArray(vertexPositionLocation);
        glDisableVertexAttribArray(vertexNormalLocation);
        glDisableVertexAttribArray(vertexColorLocation);
    }

    void drawLinesVBO(
        std::vector<glm::vec3> vertices,
        std::vector<glm::vec3> normals,
        std::vector<glm::ivec2> indices,
        std::vector<glm::vec4> colors,
        float maxValue = 1.0f, float colorRemapK = 1.0f, float brightness = 1.0f
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
        location = glGetUniformLocation(shaderProgram, "maxValue");
        glUniform1f(location, maxValue);
        location = glGetUniformLocation(shaderProgram, "colorRemapK");
        glUniform1f(location, colorRemapK);
        location = glGetUniformLocation(shaderProgram, "brightness");
        glUniform1f(location, brightness);

        // draw
        glDrawElements(GL_LINES, 2 * (int)indices.size(), GL_UNSIGNED_INT, nullptr);

        // clean-up
        glDisableVertexAttribArray(vertexPositionLocation);
        glDisableVertexAttribArray(vertexNormalLocation);
        glDisableVertexAttribArray(vertexColorLocation);
    }

    /*Draw an axes-aligned rectangle on the screen
        p0, p1: in screen coordinates, bottom left (0, 0)
        colors: 00, 10, 11, 01
    */
    void drawRect(glm::vec2 p0, glm::vec2 p1,
        std::vector<glm::vec4> colors, float maxValue = 1.0f, float colorRemapK = 1.0f
    ) {
        using glm::vec3, glm::ivec3;
        this->drawVBO(
            std::vector<vec3>({ vec3(p0.x,p0.y,.5f), vec3(p1.x,p0.y,.5f), vec3(p1.x,p1.y,.5f), vec3(p0.x,p1.y,.5f) }),
            std::vector<vec3>({ vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), vec3(0,0,1), }),
            std::vector<ivec3>({ ivec3(0,1,2), ivec3(0,2,3) }),
            colors, maxValue, colorRemapK
        );
    }

    /* Draw a character on the screen
        p0: in screen coordinates, bottom left (0, 0)
    */
    void drawCharacter(FontCharacters::FontCharacter chr,
        glm::vec2 p0, float width, glm::vec3 color)
    {
        using glm::vec2, glm::vec3, glm::ivec3;
        vec2 p1 = p0 + vec2(1, 2) * width;
        vec3 vertices[4] = { vec3(p0.x,p0.y,.5f), vec3(p1.x,p0.y,.5f), vec3(p1.x,p1.y,.5f), vec3(p0.x,p1.y,.5f) };
        vec2 uvs[4] = { vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1) };
        ivec3 indices[2] = { ivec3(0, 1, 2), ivec3(0, 2, 3) };

        glUseProgram(this->shaderProgramFont);

        // vertices
        GLuint vertexPositionLocation = glGetAttribLocation(shaderProgramFont, "vertexPosition");
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * 4, &vertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexPositionLocation);
        glVertexAttribPointer(vertexPositionLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

        // values/colors
        GLuint vertexUvLocation = glGetAttribLocation(shaderProgramFont, "vertexUv");
        glBindBuffer(GL_ARRAY_BUFFER, colorbuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * 4, &uvs[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(vertexUvLocation);
        glVertexAttribPointer(vertexUvLocation, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        // indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicebuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(glm::ivec3) * 4, &indices[0], GL_STATIC_DRAW);

        // set uniforms
        GLuint location = glGetUniformLocation(shaderProgramFont, "transformMatrix");
        glUniformMatrix4fv(location, 1, GL_FALSE, &transformMatrix[0][0]);
        location = glGetUniformLocation(shaderProgramFont, "i");
        glUniform1i(location, chr);
        location = glGetUniformLocation(shaderProgramFont, "fontColor");
        glUniform3f(location, color.x, color.y, color.z);

        // draw
        glDrawElements(GL_TRIANGLES, 3 * 2, GL_UNSIGNED_INT, nullptr);

        // clean-up
        glDisableVertexAttribArray(vertexPositionLocation);
        glDisableVertexAttribArray(vertexUvLocation);
    }

    /* Draw a string, numbers only */
    void drawNumberString(std::string str, glm::vec2 p0,
        float charWidth, float charSpace, glm::vec3 color) {
        int n = (int)str.size();
        p0.x += 0.5 * charSpace;
        for (int i = 0; i < n; i++) {
            char c = str[i];
            FontCharacters::FontCharacter fc =
                c >= '0' && c <= '9' ? (FontCharacters::FontCharacter)(c - '0') :
                c == '.' ? FontCharacters::dot :
                c == 'e' ? FontCharacters::e :
                c == '+' ? FontCharacters::plus :
                c == '-' ? FontCharacters::minus :
                FontCharacters::other;
            drawCharacter(fc, p0, charWidth, color);
            p0.x += charWidth + charSpace;
        }
    }
};


struct PrecomputedValues {
    std::vector<float> values;
    float maxValue;  // maximum absolute value
    float minValue;  // isAllPositive||isAllNegative ? 0.0 : -maxValue
    float medianValue;  // a measure of the median absolute value
    bool isAllPositive, isAllNegative;
    std::vector<glm::vec4> colors;

    float getHash() {
        using glm::vec2, glm::fract;
        int n = (int)values.size();
        int increment = std::max(n / 32, 1);
        float h = 0.0f;
        for (int i = increment / 3; i < n; i += increment) {
            h += fract(sin(dot(
                vec2(i, values[i]) / vec2(n + 1, maxValue + 1.0f),
                vec2(12.9898, 78.233))) * 43758.5453);
        }
        return h;
    }
};


struct ToggleCheckbox: WindowInput {
    bool toggled;
    glm::vec2 pos;  // x0, y0, from bottom left
    float size;  // square button
    FontString text;
    ToggleCheckbox(glm::vec2 pos, float size,
        FontString text = FontString()
    ): pos(pos), size(size), text(text), toggled(false) {};

    // interaction
    bool isHover() {
        glm::vec2 dx = RenderParams::mousePos - this->pos;
        return dx.x >= 0.0 && dx.y >= 0.0 && dx.x <= size && dx.y <= size;
    }
    void mouseClick() {
        if (isHover()) this->toggled ^= true;
    }

    // rendering
    void render() {
        using glm::vec2, glm::vec3, glm::vec4;
        using RenderParams::viewport;
        viewport->drawRect(
            pos, pos + size,
            std::vector<vec4>(4, vec4(toggled ? 0.8 : 0.2))
        );
        vec3 c = toggled ? vec3(0.2) : vec3(1.0);
        if (text.size() == 1) {
            viewport->drawCharacter(
                text[0], pos + vec2(0.25, 0.1) * size, 0.5 * size, c);
        }
        if (text.size() == 2) {
            viewport->drawCharacter(
                text[0], pos + vec2(0.08, 0.2) * size, 0.45 * size, c);
            viewport->drawCharacter(
                text[1], pos + vec2(0.55, 0.1) * size, 0.35 * size, c);
        }
        if (text.size() == 3) {
            viewport->drawCharacter(
                text[0], pos + vec2(0.08, 0.2) * size, 0.4 * size, c);
            viewport->drawCharacter(
                text[1], pos + vec2(0.55, 0.52) * size, 0.32 * size, c);
            viewport->drawCharacter(
                text[2], pos + vec2(0.55, 0.0) * size, 0.32 * size, c);
        }
    }
};


// a slider that controls color remapping to fit extreme values
// at the bottom of the window
struct ColorRemapSlider: WindowInput {

    PrecomputedValues* precomputed;

    float hashv;  // check of precomputed is changed
    float vmax, vmid;  // precomputed->maxValue/medianValue
    float v0, v1;  // display min, display max
    float tt;  // slider location

    // remapping:
    // remap v2t(vmid) to tt

    const float height = 30.0;
    const float sidePadding = 30.0;

    void resetV() {
        hashv = precomputed->getHash();
        vmax = precomputed->maxValue;
        vmid = precomputed->medianValue;
        v0 = -vmax, v1 = vmax;
        tt = tt > 0.5f ? 0.65f : 0.35f;
        float vmid1 = round125(vmid);
        tt = tt > 0.5f ? v2rt(vmid1) : 1.0f - v2rt(vmid1);
        vmid = vmid1;
    }
    void clampTt() {
        if (tt < 0.5f) tt = clamp(tt, 0.001f, 0.48f);
        else tt = clamp(tt, 0.52f, 0.999f);
    }

    ColorRemapSlider(PrecomputedValues* precomputed)
        : precomputed(precomputed), tt(0.35f) {
        resetV();
    }

    // parameter to screen coordinate
    float t2x(float t) {
        return glm::mix(
            sidePadding,
            (float)RenderParams::iResolution.x - sidePadding,
            t
        );
    }
    // screen coordinate to parameter
    float x2t(float x) {
        return (x - sidePadding) /
            ((float)RenderParams::iResolution.x - 2.0f * sidePadding);
    }

    // get the nonlinear remapping parameter k
    float getK() {
        float tm = 0.5f + 0.5f * vmid / vmax;
        float tu = tt > 0.5f ? tt : 1.0f - tt;
        float k = log((1 - tu) / tu) / log((1 - tm) / tm);
        return k;
    }
    // value to remapped t
    float v2rt(float v) {
        float t = 0.5 + 0.5 * v / vmax;
        float k = getK();
        return pow(t, k) / (pow(t, k) + pow(1.0f - t, k));
    }

    // interactions
    bool isHover() {
        if (precomputed->getHash() != hashv) resetV();
        glm::vec2 res = RenderParams::iResolution;
        glm::vec2 mouse = RenderParams::mousePos;
        return abs(mouse.x - 0.5f * res.x) < (0.5f * res.x - sidePadding)
            && mouse.y <= height;
    }
    void mouseClick() {
        tt = x2t(RenderParams::mousePos.x);
        clampTt();
    }
    void mouseMove(glm::vec2 mouse_delta) {
        if (!RenderParams::mouseDown) return;
        tt = x2t(RenderParams::mousePos.x);
        clampTt();
    }
    void mouseScroll(float yoffset) {
        tt += (tt > 0.5f ? -1 : 1) * 0.005 * yoffset;
        clampTt();
    }

    // rendering
    void render() {
        using glm::vec2, glm::vec3, glm::vec4;
        glm::vec2 res = RenderParams::iResolution;
        vec4 c0 = vec4(-1, -1, -1, 0.0f);
        vec4 c1 = vec4(-1, -1, -1, 1.0f);
        RenderParams::viewport->drawRect(
            vec2(t2x(0), 0), vec2(t2x(1), height),
            std::vector<vec4>({ c0, c1, c1, c0 }),
            tt > 0.5f ? vmax : -vmax, -getK()
        );
        RenderParams::viewport->drawRect(
            vec2(t2x(tt) - 2, 0),
            vec2(t2x(tt) + 2, height),
            std::vector<vec4>(4, vec4(0.9, 0.9, 0.9, 1))
        );
        // draw numbers
        auto drawNumber = [=](float num, float t) {
            char ctemp[1024];
            if (abs(num) >= 99.5f) sprintf(ctemp, "%d", (int)round(num));
            else sprintf(ctemp, "%.2g", num);
            std::string s = ctemp;
            if (s.size() > 2 && s[0] == '0' && s[1] == '.')
                s = s.substr(1, s.size() - 1);
            if (s.size() > 3 && s[0] == '-' && s[1] == '0' && s[2] == '.')
                s = "-" + s.substr(2, s.size() - 2);
            RenderParams::viewport->drawNumberString(
                s, vec2(t2x(t) - 9 * 0.5 * s.size(), height + 2),
                8, 1, vec3(1));
        };
        float tu = tt > 0.5f ? tt : 1.0f - tt;
        // key numbers
        drawNumber(vmid, tu);
        drawNumber(-vmid, 1.0f - tu);
        drawNumber(vmax, 1.0f);
        drawNumber(-vmax, 0.0f);
        drawNumber(0.0f, 0.5f);
        // larger scales
        float v = vmid, tprev = tu;
        for (int i = 0; i < 32; i++) {
            v = round125(2.0f * v);
            float t = v2rt(v);
            if (t > 0.95f || t < 0.05f) break;
            if (abs(t - tprev) < 0.05f) continue;
            drawNumber(v, t);
            drawNumber(-v, 1.0f - t);
            tprev = t;
        }
        // smaller scales
        v = vmid, tprev = tu;
        for (int i = 0; i < 32; i++) {
            v = round125(0.5f * v);
            float t = v2rt(v);
            if (abs(t - 0.5f) < 0.05f) break;
            if (abs(t - tprev) < 0.05f) continue;
            drawNumber(v, t);
            drawNumber(-v, 1.0f - t);
            tprev = t;
        }
    }

};



bool initWindow() {
    // Initialise GLFW
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW.\n");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    // Open a window and create its OpenGL context
    RenderParams::window = glfwCreateWindow(
        RenderParams::iResolution.x, RenderParams::iResolution.y, "FEM Visualizer", NULL, NULL);
    if (RenderParams::window == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate(); return false;
    }
    glfwMakeContextCurrent(RenderParams::window);

    // Ensure we can capture the escape key being pressed below
    // glfwSetInputMode(RenderParams::window, GLFW_STICKY_KEYS, GL_TRUE);

    glGenVertexArrays(1, &RenderParams::vertexArrayID);
    glBindVertexArray(RenderParams::vertexArrayID);
    return true;
}


std::function<void()> loop;
void main_loop() { loop(); }

void mainGUI(
    DiscretizedModel<float, float> model,
    bool flatShading = false
) {
    using glm::vec2, glm::vec3, glm::vec4;

    // model
    std::vector<vec3> vertices(model.N, vec3(0));
    std::vector<vec3> normals(model.N, vec3(0));
    PrecomputedValues precomputed;
    vec3 minv(1e10f), maxv(-1e10f);
    for (int i = 0; i < model.N; i++) {
        vec3 v = vec3(model.X[i], model.U[i]);
        vertices[i] = vec3(v.x, v.y, v.z);
        minv = glm::min(minv, vertices[i]);
        maxv = glm::max(maxv, vertices[i]);
    }

    // faces
    auto ivec3Cmp = [](glm::ivec3 a, glm::ivec3 b) {
        // std::sort(&a.x, &a.x + 3);
        // std::sort(&b.x, &b.x + 3);
        return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
    };
    std::map<glm::ivec3, int, decltype(ivec3Cmp)> uniqueIndicesF(ivec3Cmp);  // count
    ivec3 ts[MAX_AREA_ELEMENT_TN];
    for (int i = 0; i < model.M; i++) {
        int tn = model.SE[i]->getNumTriangles();
        model.SE[i]->getTriangles(ts);
        for (int ti = 0; ti < tn; ti++) {
            int* t0 = &ts[ti].x;
            assert(t0[0] != t0[1] && t0[0] != t0[2]);
            int i = t0[0] < t0[1] && t0[0] < t0[2] ? 0 :
                t0[1] < t0[2] && t0[1] < t0[0] ? 1 : 2;
            glm::ivec3 t(t0[i], t0[(i + 1) % 3], t0[(i + 2) % 3]);
            uniqueIndicesF[t] += 1;
            t = glm::ivec3(t.x, t.z, t.y);
            assert(uniqueIndicesF.find(t) == uniqueIndicesF.end());
        }
    }
    std::vector<glm::ivec3> indicesF;
    for (auto p : uniqueIndicesF) //if (p.second == 1)
        indicesF.push_back(p.first);

    // normals
    for (auto fc : uniqueIndicesF) {
        assert(fc.second == 1);
        glm::ivec3 f = fc.first;
        vec3 n = glm::cross(
            vertices[f.y] - vertices[f.x],
            vertices[f.z] - vertices[f.x]);
        n = glm::dot(n, n) == 0. ? n : normalize(n);
        n = vec3(0, 0, 1);
        normals[f.x] += n, normals[f.y] += n, normals[f.z] += n;
    }

    // edges
    auto ivec2Cmp = [](glm::ivec2 a, glm::ivec2 b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    };
    std::set<glm::ivec2, decltype(ivec2Cmp)> uniqueIndicesE(ivec2Cmp);
    ivec2 es[MAX_AREA_ELEMENT_EN];
    for (int i = 0; i < model.M; i++) {
        int en = model.SE[i]->getNumEdges();
        model.SE[i]->getEdges(es);
        for (int ei = 0; ei < en; ei++) {
            glm::ivec2 e(es[ei].x, es[ei].y);
            if (e.x > e.y) std::swap(e.x, e.y);
            uniqueIndicesE.insert(e);
        }
    }
    std::vector<glm::ivec2> indicesE =
        std::vector<glm::ivec2>(uniqueIndicesE.begin(), uniqueIndicesE.end());


    // create window
    if (!initWindow()) return;

    // GUI
    RenderParams::viewport = new Viewport(
        2.5f / fmax(fmax(maxv.x - minv.x, maxv.y - minv.y), maxv.z - minv.z),
        0.5f * (minv + maxv), -0.33f * PIf, -0.17f * PIf);
    vec2 res = RenderParams::iResolution;
    ColorRemapSlider slider(&precomputed);
    RenderParams::viewport->addWindowInput(&slider);

    // flat shading
    if (flatShading) {
        std::vector<glm::ivec3> indicesF1;
        for (auto f : indicesF) {
            vec3 n = cross(
                vertices[f[1]] - vertices[f[0]],
                vertices[f[2]] - vertices[f[0]]
            );
            int vn = (int)vertices.size();
            for (int _ = 0; _ < 3; _++) {
                vertices.push_back(vertices[f[_]]);
                normals.push_back(n);
            }
            indicesF1.push_back(glm::ivec3(vn, vn + 1, vn + 2));
        }
        indicesF = indicesF1;
        // for debugging mesh generation, not so good
        precomputed.colors = std::vector<glm::vec4>(
            vertices.size(), glm::vec4(0.9));
    }

    // mouse action(s)
    glfwSetScrollCallback(RenderParams::window,
        [](GLFWwindow* window, double xoffset, double yoffset) {
            RenderParams::viewport->mouseScroll((float)yoffset);
        });

    // main loop
    loop = [&] {
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
        viewport->drawVBO(vertices, normals, indicesF,
            precomputed.colors,
            (slider.tt > 0.5f ? 1 : -1) * precomputed.maxValue,
            slider.getK(), 1.0f
        );
        viewport->drawLinesVBO(vertices, normals, indicesE,
            precomputed.colors,
            (slider.tt > 0.5f ? 1 : -1) * precomputed.maxValue,
            slider.getK(), 0.8f
        );
        // viewport->initDraw2D();
        // viewport->drawWindowInputs();

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
