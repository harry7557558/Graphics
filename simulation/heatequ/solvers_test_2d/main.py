import pygame
from pygame import Vector2, Vector3
import numpy as np
import math
from copy import deepcopy

from viewport_3d import Viewport3D
from OpenGL.GL import *
from OpenGL.GLU import *
from state import State
import states_builtin

import integrators
import colorsys

import time


class Solver:

    def __init__(self, initial_state: State, integrator: integrators.Base, color_hue: float):
        self.state = deepcopy(initial_state)
        self.integrator = integrator(self.state)
        self.curve_color = self._hsl_to_rgb(color_hue, 1.0, 0.65)
        self.text_color = self._hsl_to_rgb(color_hue, 1.0, 0.8)

    @staticmethod
    def _hsl_to_rgb(h, s, l) -> tuple[int, int, int]:
        rgb = colorsys.hls_to_rgb(h, l, s)  # between 0 and 1
        return (rgb[0], rgb[1], rgb[2])

    def update(self, delta_t: float, eval_count: int) -> None:
        if eval_count % self.integrator.EVAL_COUNT != 0:
            raise ValueError("Eval count are not integer multiples")
        step_count = eval_count//self.integrator.EVAL_COUNT
        t0 = time.perf_counter()
        for i in range(step_count):
            self.integrator.update(delta_t/step_count)
        dt = 1000.0 * (time.perf_counter() - t0)
        # Implicit Euler is faster than explicit Euler?!
        print(self.integrator.__class__.__name__, "{:.1f}ms".format(dt))

    def draw(self, surface, viewport, font, text_pos: tuple[int, int]) -> None:
        self.state.draw(surface, viewport, self.curve_color)
        self.state.draw_info(surface, font, text_pos, self.text_color)


RESOLUTION = (512, 512)
FPS = 30


def main():
    pygame.init()
    screen = pygame.display.set_mode(
        RESOLUTION, pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("2D Diffusion Equation Integrators Test")

    font = pygame.font.SysFont('Consolas', 16)

    #state = states_builtin.circle_dam(16, 12, 0.1, Vector2(-0.3, 0.3), 0.5)
    #state = states_builtin.circle_dam(8, 6, 2.0, Vector2(-0.3, 0.3), 0.5)
    #state = states_builtin.circle_dam(40, 40, 2.0, Vector2(-0.3, 0.3), 0.5)
    #state = states_builtin.heater_cooler(12, 12, 0.2, 0.5, 4.0)
    state = states_builtin.heater_cooler(20, 20, 2.0, 0.5, 40.0)
    #state = states_builtin.heater_cooler(40, 40, 10000.0, 0.5, 200000.0)

    viewport = Viewport3D(Vector2(RESOLUTION), 0.5,
                          Vector3(0, 0, 0), -60, -30)

    integrators_ = [
        #integrators.Euler,
        #integrators.Midpoint,
        #integrators.RungeKutta,
        integrators.ImplicitEuler
    ]
    solvers = []
    for i in range(len(integrators_)):
        solvers.append(Solver(state, integrators_[i], i/len(integrators_)))
    state.visualize_matrix("D:\\sparse-matrix.png")

    time_start = time.perf_counter()

    running = True
    while running:
        # get info
        mouse_down = pygame.mouse.get_pressed()[0]
        mouse_pos = pygame.mouse.get_pos()
        mouse_delta = pygame.mouse.get_rel()
        wheel_delta = 0.0

        # handle events + get info
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            # mouse scroll
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    wheel_delta += 180.0
                elif event.button == 5:
                    wheel_delta -= 180.0

        # update viewport
        if mouse_down:
            viewport.mouse_move(mouse_delta)
        if wheel_delta != 0.0:
            viewport.mouse_scroll(mouse_pos, math.exp(0.001*wheel_delta))

        # simulation
        for solver in solvers:
            solver.update(1.0/FPS, 4)

        # draw
        viewport.draw(screen)
        for i in range(len(solvers)):
            solvers[i].draw(screen, viewport, font, [5, 5+18*(i+1)])

        # display actual time
        text = font.render(
            "{:.2f}s".format(time.perf_counter()-time_start),
            True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topleft = [5, 5]
        screen.blit(text, text_rect)

        # update
        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    # __import__("cProfile").run("main()")
    main()
