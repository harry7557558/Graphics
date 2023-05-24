import pygame
from pygame import Vector2
import numpy as np
import math
from copy import deepcopy

from viewport import Viewport, Slider
from state import State
import states_builtin

import integrators

import time


RESOLUTION = (512, 512)
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("2D Polygon Smoothing Test")

    # state = states_builtin.noisy_circle(200, 1.0, 0.5)
    # state = states_builtin.noisy_line(200, 1.0, 0.5)
    # state = states_builtin.noisy_line(200, 1.0, 0.5, 0.4, False)
    # state = states_builtin.noisy_circle(200, 1.0, 0.05, 0.5)
    # state = states_builtin.noisy_circle(200, 1.0, 0.05, 2.0)
    # state = states_builtin.noisy_flower(200, 5, 0.5, 0.5, 0.05)
    state = states_builtin.noisy_flower(500, 5, 0.5, 0.5, 0.05)
    # state = states_builtin.noisy_flower(1000, 5, 0.5, 0.5, 0.02)
    # state = states_builtin.noisy_flower(2000, 5, 0.5, 0.5, 0.01)

    # gui
    viewport = Viewport(Vector2(RESOLUTION), 1.5, Vector2(0, 0))
    font = pygame.font.SysFont("Consolas", 16)
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
            h = 1000.0 * h**2
            integrators.update_implicit_euler_pv(state_ieuler, h)
        else:
            h = 50.0 * h
            integrators.update_implicit_euler(state_ieuler, h)

        t1 = time.perf_counter()
        print("Implicit Euler {:.1f}ms".format(1000.0*(t1-t0)))

    slider_pv.set_callback(recompute_smoothing)
    slider_step.set_callback(recompute_smoothing)
    recompute_smoothing()

    # main loop
    running = True
    while running:
        # get info
        mouse_down = pygame.mouse.get_pressed()[0]
        mouse_pos = pygame.mouse.get_pos()
        mouse_delta = pygame.mouse.get_rel()
        wheel_delta = 0.0
        mouse_pos_world = viewport.screen_to_world(mouse_pos)

        # handle events + get info
        events = pygame.event.get()
        if len(events) == 0:
            continue
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            # mouse scroll
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
        viewport.draw(screen)
        state.draw(screen, viewport, (128, 128, 128))
        state_ieuler.draw(screen, viewport, (0, 128, 255))
        slider_pv.draw(screen)
        slider_step.draw(screen)

        # draw mouse position
        mouse_text = font.render(
            "({:.3f},{:.3f})".format(*mouse_pos_world), True, (255, 255, 255))
        text_rect = mouse_text.get_rect()
        text_rect.topright = [RESOLUTION[0]-5, 5]
        screen.blit(mouse_text, text_rect)

        # update
        pygame.display.flip()
        pygame.time.Clock().tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    # __import__("cProfile").run("main()")
    main()
