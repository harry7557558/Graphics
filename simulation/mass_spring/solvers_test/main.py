import pygame
from pygame import Vector2
import numpy as np
import math
from copy import deepcopy

from viewport import Viewport
from state import Mass, Spring, State
import states_builtin

import integrators
import xpbd
import colorsys


class Solver:

    def __init__(self, initial_state: State, integrator: integrators.Base, color_hue: float):
        self.state = deepcopy(initial_state)
        self.integrator = integrator(self.state)
        self.mass_color = self._hsl_to_rgb(color_hue, 1.0, 0.7)
        self.spring_color = self._hsl_to_rgb(color_hue, 1.0, 0.6)
        self.text_color = self._hsl_to_rgb(color_hue, 1.0, 0.8)

    @staticmethod
    def _hsl_to_rgb(h, s, l) -> tuple[int, int, int]:
        rgb = colorsys.hls_to_rgb(h, l, s)
        rgb = (255.99 * np.array(rgb)).astype(int)
        return (rgb[0], rgb[1], rgb[2])

    def update(self, delta_t: float, eval_count: int) -> None:
        if eval_count % self.integrator.EVAL_COUNT != 0:
            raise ValueError("Eval count are not integer multiples")
        step_count = eval_count//self.integrator.EVAL_COUNT
        for i in range(step_count):
            self.integrator.update(delta_t/step_count)

    def draw(self, surface, viewport, font, text_pos: tuple[int, int]) -> None:
        self.state.draw(surface, viewport, self.mass_color, self.spring_color)
        name = self.integrator.__class__.__name__
        is_upper = [ord('A') <= ord(c) <= ord('Z') for c in name]
        if is_upper.count(True) == 2:
            u1 = 1 + is_upper[1:].index(True)
            name = name[:2] + name[u1:u1+2]
        name = name[:4]
        self.state.draw_info(surface, font, name, text_pos, self.text_color)


RESOLUTION = (512, 512)
FPS = 60


def main():
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("Mass-Spring Integrators Test")

    font = pygame.font.SysFont('Consolas', 16)

    # state = states_builtin.square_barred()
    state = states_builtin.sheet_hang(1., 4.0, 0.7, 8, 4, 3.0, 1.5, False, False)
    # state = states_builtin.sheet_hang(1., 2.0, 0.1, 12, 6, 3.0, 1.5, True, False)
    # state = states_builtin.sheet_hang(1., 1000.0, 20.0, 12, 3, 4.0, 1.0, False, False)

    viewport = Viewport(Vector2(RESOLUTION), 2.5, Vector2(0, 1))

    integrators_ = [
        integrators.Euler,
        integrators.EulerCromer,
        # integrators.Midpoint,
        # integrators.RungeKutta,
        integrators.Verlet,
        xpbd.XPBD
    ]
    solvers = []
    for i in range(len(integrators_)):
        solvers.append(Solver(state, integrators_[i], i/len(integrators_)))

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

        # simulation
        for solver in solvers:
            solver.update(1.0/FPS, 4)

        # draw
        viewport.draw(screen)
        for i in range(len(solvers)):
            solvers[i].draw(screen, viewport, font, [5, 5+18*i])

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
    main()
