import pygame
from pygame import Vector2, Vector3
import numpy as np
import math
from copy import deepcopy

from viewport import Viewport, Slider
from OpenGL.GL import *
from OpenGL.GLU import *
from state import State
import states_builtin

import integrators

import time


RESOLUTION = (512, 512)
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode(
        RESOLUTION, pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Mesh Smoothing Test")

    # state
    # state = states_builtin.unit_cube()
    # state = states_builtin.plane(Vector2(1.5, 1.0), 15, 10, 0.2)
    state = states_builtin.cylinder(1.2, 1.0, 1.0, 20, 10, True, 0.2)
    # state = states_builtin.cylinder(1.2, 1.0, 1.0, 20, 10, False, 0.2)
    # state = states_builtin.cylinder(1.2, 1.0, 1.0, 100, 50, False, 0.1)
    # state = states_builtin.sphere_uv(1.2, 1.1, 1.0, 20, 10, True, 0.5)
    # state = states_builtin.sphere_uv(1.2, 1.1, 1.0, 20, 10, False, 0.5)
    # state = states_builtin.sphere_uv(1.2, 1.1, 1.0, 100, 50, False, 0.2)
    # state = states_builtin.torus(1.0, 0.5, 10, 30, False, True, 0.3)
    # state = states_builtin.torus(1.0, 0.5, 30, 100, False, True, 0.2)

    # GUI
    viewport = Viewport(Vector2(RESOLUTION), 0.5,
                        Vector3(0, 0, 0), -60, -30)
    slider_pv = Slider(Vector2(10, 10), Vector2(120, 25), 0.0, 1.0, 1.0, 1.0)
    slider_step = Slider(Vector2(10, 30), Vector2(120, 45), 0.0, 1.0, 0.0, 0.5)

    # smoothing
    state_ieuler = deepcopy(state)

    def recompute_smoothing():
        nonlocal state_ieuler
        state_ieuler = deepcopy(state)
        t0 = time.perf_counter()

        step_size = slider_step.get_value()
        h = step_size / max(1.0-step_size, 1e-100)
        if slider_pv.get_value() > 0.5:
            h = 1.0 * h
            integrators.ieuler(state_ieuler, h)
        else:
            h = 1.0 * h
            integrators.euler(state_ieuler, h)

        t1 = time.perf_counter()
        print("Implicit Euler {:.1f}ms".format(1000.0*(t1-t0)))

    slider_pv.set_callback(recompute_smoothing)
    slider_step.set_callback(recompute_smoothing)
    recompute_smoothing()

    running = True
    while running:
        # get info
        mouse_down = pygame.mouse.get_pressed()[0]
        mouse_pos = pygame.mouse.get_pos()
        mouse_delta = pygame.mouse.get_rel()
        wheel_delta = 0.0

        # handle events + get info
        events = pygame.event.get()
        if len(events) == 0:
            pygame.time.Clock().tick(FPS)
            continue
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # click
                if event.button == 1:
                    slider_pv.mouse_down(mouse_pos)
                    slider_step.mouse_down(mouse_pos)
                # scroll
                if event.button == 4:
                    wheel_delta += 180.0
                elif event.button == 5:
                    wheel_delta -= 180.0
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # this is optional
                    slider_pv.mouse_up()
                    slider_step.mouse_up()

        # update viewport
        if mouse_down:
            if not slider_pv.mouse_move(mouse_pos) and not slider_step.mouse_move(mouse_pos):
                viewport.mouse_move(mouse_delta)
        if wheel_delta != 0.0:
            viewport.mouse_scroll(mouse_pos, math.exp(0.001*wheel_delta))

        # draw
        viewport.init_draw()
        state_ieuler.draw(viewport, Vector3(0.5, 0.5, 0.6))
        viewport.finish_draw()
        slider_pv.draw(viewport)
        slider_step.draw(viewport)

        # update
        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    # __import__("cProfile").run("main()")
    main()
