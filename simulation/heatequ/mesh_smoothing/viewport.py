import pygame
from pygame import Vector2, Vector3
from OpenGL.GL import *
from OpenGL.GLU import *
import ctypes
from typing import Callable


class Viewport():

    def __init__(self, resolution: Vector2, scale: float, center: Vector3, rx: float, rz: float):
        """Angles are in degrees"""
        self._resolution = resolution
        self._scale = scale
        self._center = center
        self._rx = rx
        self._rz = rz

        self.shader_program = self._create_shader_program(
            """#version 330
in vec3 vertexPosition;
in vec3 vertexNormal;
out vec3 fragNormal;
uniform mat4 transformMatrix;
void main() {
    gl_Position = transformMatrix * vec4(vertexPosition, 1);
    fragNormal = (transformMatrix*vec4(vertexNormal,0)).xyz;
}""",
            """#version 330
in vec3 fragNormal;
out vec3 fragColor;
uniform vec3 baseColor;
void main() {
    vec3 n = normalize(fragNormal);
    float amb = 1.0;
    float dif = abs(n.z);
    float spc = pow(abs(n.z), 40.0);
    fragColor = (0.2*amb+0.7*dif+0.1*spc)*baseColor;
}""")
        self.vertexbuffer = glGenBuffers(1)
        self.normalbuffer = glGenBuffers(1)
        self.indicebuffer = glGenBuffers(1)

    def mouse_move(self, mouse_delta: Vector2) -> None:
        if type(mouse_delta) != Vector2:
            mouse_delta = Vector2(*mouse_delta)
        self._rx += 1.0*mouse_delta.y
        self._rz += 1.0*mouse_delta.x

    def mouse_scroll(self, mouse_pos: Vector2, zoom: float) -> None:
        if type(mouse_pos) != Vector2:
            mouse_pos = Vector2(*mouse_pos)
        if not zoom > 0.0:
            raise ValueError("Mouse zooming must be positive.")
        self._scale *= zoom

    def init_draw(self) -> None:
        """Call this when start drawing the 3D scene"""
        glUseProgram(self.shader_program)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # setup matrix
        glLoadIdentity()
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45.0, self._resolution.x/self._resolution.y, 1e-1, 1e+2)
        glTranslated(0.0, 0.0, -3.0/self._scale)
        glRotated(self._rx, 1, 0, 0)
        glRotated(self._rz, 0, 0, 1)
        glTranslated(*(-self._center))

    def finish_draw(self) -> None:
        """Call this when finished drawing the 3D scene and start drawing sliders"""
        glDisable(GL_DEPTH_TEST)
        glLoadIdentity()

    def draw_vbo(self, vertices: list[float], normals: list[float], indices: list[int], color: Vector3) -> None:

        vertexPositionLocation = glGetAttribLocation(
            self.shader_program, "vertexPosition")
        glBindBuffer(GL_ARRAY_BUFFER, self.vertexbuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertices),
                     (ctypes.c_float*len(vertices))(*vertices), GL_STATIC_DRAW)
        glEnableVertexAttribArray(vertexPositionLocation)
        glVertexAttribPointer(vertexPositionLocation, 3,
                              GL_FLOAT, GL_FALSE, 0, None)

        vertexNormalLocation = glGetAttribLocation(
            self.shader_program, "vertexNormal")
        glBindBuffer(GL_ARRAY_BUFFER, self.normalbuffer)
        glBufferData(GL_ARRAY_BUFFER, 4*len(normals),
                     (ctypes.c_float*len(normals))(*normals), GL_STATIC_DRAW)
        glEnableVertexAttribArray(vertexNormalLocation)
        glVertexAttribPointer(vertexNormalLocation, 3,
                              GL_FLOAT, GL_FALSE, 0, None)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.indicebuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(indices),
                     (ctypes.c_uint32*len(indices))(*indices), GL_STATIC_DRAW)

        matrixLocation = glGetUniformLocation(
            self.shader_program, "transformMatrix")
        matrix = glGetFloatv(GL_PROJECTION_MATRIX)
        glUniformMatrix4fv(matrixLocation, 1, GL_FALSE, matrix)

        colorLocation = glGetUniformLocation(self.shader_program, "baseColor")
        glUniform3fv(colorLocation, 1, (ctypes.c_float*3)(*color))

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glDisableVertexAttribArray(vertexPositionLocation)
        glDisableVertexAttribArray(vertexNormalLocation)

    def draw_rect(self, p0: Vector2, p1: Vector2, color: Vector3) -> None:
        """Draw an axes-aligned rectangle on screen
           p0, p1: in screen coordinate, top left (0, 0)
           color: RGB values between 0 and 1"""
        p0 = 2.0 * Vector2(p0.x/self._resolution.x,
                           1.0 - p0.y/self._resolution.y) - Vector2(1.0)
        p1 = 2.0 * Vector2(p1.x/self._resolution.x,
                           1.0 - p1.y/self._resolution.y) - Vector2(1.0)
        vertices = [p0.x, p0.y, 0.5,
                    p0.x, p1.y, 0.5,
                    p1.x, p1.y, 0.5,
                    p1.x, p0.y, 0.5]
        normals = [0, 0, 1] * 4
        indices = [0, 1, 2,
                   0, 2, 3]
        self.draw_vbo(vertices, normals, indices, color)

    def _create_shader_program(self, vs_source: str, fs_source: str):
        VertexShaderID = glCreateShader(GL_VERTEX_SHADER)
        FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER)
        Result = GL_FALSE

        glShaderSource(VertexShaderID, vs_source)
        glCompileShader(VertexShaderID)
        error = glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS)
        if error != GL_TRUE:
            info = glGetShaderInfoLog(VertexShaderID)
            print("Vertex shader compile error.", info, sep='\n')

        glShaderSource(FragmentShaderID, fs_source)
        glCompileShader(FragmentShaderID)
        error = glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS)
        if error != GL_TRUE:
            info = glGetShaderInfoLog(FragmentShaderID)
            print("Fragment shader compile error.", info, sep='\n')

        ProgramID = glCreateProgram()
        glAttachShader(ProgramID, VertexShaderID)
        glAttachShader(ProgramID, FragmentShaderID)
        glLinkProgram(ProgramID)
        error = glGetProgramiv(ProgramID, GL_LINK_STATUS)
        if error != GL_TRUE:
            info = glGetShaderInfoLog(ProgramID)
            print("Program linking error.", info, sep='\n')

        glDetachShader(ProgramID, VertexShaderID)
        glDetachShader(ProgramID, FragmentShaderID)
        glDeleteShader(VertexShaderID)
        glDeleteShader(FragmentShaderID)
        return ProgramID


class Slider:

    def __init__(self, p0: Vector2, p1: Vector2,
                 v0: float, v1: float, vstep: float, v: float,
                 callback: Callable = None):
        self.p0 = Vector2(p0)
        self.p1 = Vector2(p1)
        self.v0 = float(v0)
        self.v1 = float(v1)
        self.vstep = float(vstep)
        self.v = float(v)
        self.callback = callback
        self.has_capture = False

    def set_callback(self, callback: Callable):
        self.callback = callback

    def get_value(self) -> float:
        return self.v

    def draw(self, viewport: Viewport):
        p0, p1 = self.p0, self.p1
        dp = p1 - p0
        t = (self.v - self.v0) / (self.v1 - self.v0)
        viewport.draw_rect(p0, p1, Vector3(0.25))
        viewport.draw_rect(p0, Vector2(p0.x+dp.x*t, p1.y), Vector3(0.75))

    def mouse_down(self, pos: Vector2) -> bool:
        """Call this when mouse down and pass mouse position"""
        pos = Vector2(pos)
        if self.p0.x <= pos.x < self.p1.x and self.p0.y <= pos.y < self.p1.y:
            self.has_capture = True
        else:
            self.has_capture = False

    def mouse_up(self) -> bool:
        """Call this when mouse up (optional)"""
        self.has_capture = False

    def mouse_move(self, pos: Vector2) -> bool:
        """Call this when mouse click/drag to update value
           Returns True if mouse has effect, False otherwise"""
        pos = Vector2(pos)
        if self.has_capture:
            t = (pos.x - self.p0.x) / (self.p1.x - self.p0.x)
            dv = (self.v1 - self.v0) * t
            if self.vstep != 0.0:
                dv = round(dv/self.vstep)*self.vstep
            self.v = min(max(self.v0+dv, self.v0), self.v1)
            if self.callback != None:
                self.callback()
            return True
        return False
