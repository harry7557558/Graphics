import pygame
from pygame import Vector2
import numpy as np
import math
from copy import deepcopy

from viewport import Viewport
from state import State
import states_builtin

import integrators
import colorsys

import time


RESOLUTION = (512, 512)
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("2D Polygon Smoothing Test")

    #state = states_builtin.noisy_circle(200, 1.0, 0.5)
    state = states_builtin.noisy_line(200, 1.0, 0.5)

    # smoothing
    state_euler = deepcopy(state)
    state_ieuler = deepcopy(state)

    step_size = 1000.0

    t0 = time.perf_counter()
    for i in range(int(2.0*step_size+1.0)):
        h = step_size/int(2.0*step_size+1.0)
        integrators.update_euler(state_euler, h)
        #integrators.update_euler(state_euler, -h)
    t1 = time.perf_counter()
    print("Euler {:.1f}ms".format(1000.0*(t1-t0)))

    t0 = time.perf_counter()
    for i in range(5):
        h = step_size/5
        integrators.update_implicit_euler(state_ieuler, h)
    t1 = time.perf_counter()
    print("Implicit Euler {:.1f}ms".format(1000.0*(t1-t0)))

    viewport = Viewport(Vector2(RESOLUTION), 1.5, Vector2(0, 0))
    font = pygame.font.SysFont("Consolas", 16)

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

        # draw
        viewport.draw(screen)
        state.draw(screen, viewport, (128, 128, 128))
        state_euler.draw(screen, viewport, (255, 128, 0))
        state_ieuler.draw(screen, viewport, (0, 128, 255))

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
