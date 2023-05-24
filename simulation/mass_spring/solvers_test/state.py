import pygame
from pygame import Vector2
from viewport import Viewport
from typing import Callable


class Mass:

    def __init__(self, inv_m: float = 0.0, drag: float = 0.0, x: Vector2 = Vector2(0.0),
                 v: Vector2 = Vector2(0.0), a: Vector2 = Vector2(0.0)):
        self.inv_m = inv_m  # reciprocal of mass in kg^-1
        self.drag = drag  # add -drag*v to acceleration
        self.x = x  # position in m
        self.v = v  # velocity in m/s
        self.a = a  # acceleration in m/s²


class Spring:

    def __init__(self, masses: tuple[int, int], l0: float, ks: float, kd: float):
        self.masses = (masses[0], masses[1])  # pointers
        self.l0 = l0  # rest length in m
        self.ks = ks  # spring constant in N/m
        self.kd = kd  # damping constant


class State:

    def __init__(self, masses: list[Mass], springs: list[Spring],
                 acc_field: Callable[[float], list[Vector2, float]] = None,
                 time: float = 0.0,
                 g: Vector2 = Vector2(0.0, -9.8)):
        self.masses = masses
        self.springs = springs
        self.g = g  # acceleration due to gravity in m/s²
        self.time = time  # time in s
        if acc_field == None:
            self.acc_field = lambda x, t: Vector2(0.0)
        else:
            self.acc_field = acc_field

    def get_spring_force(self, spring: Spring):
        """Returns a tuple of two vectors,
            the spring force for each mass."""
        m1 = self.masses[spring.masses[0]]
        m2 = self.masses[spring.masses[1]]
        dx = m2.x - m1.x
        dv = m2.v - m1.v
        l = dx.length()
        dx_n = dx / l if l != 0.0 else dx
        fs = spring.ks * (l-spring.l0)  # spring
        fd = spring.kd * dx_n.dot(dv)  # damping
        f = (fs + fd) * dx_n
        return (f, -f)

    def get_spring_constraint(self, spring: Spring):
        """For XPBD. Returns a tuple of:
            - stiffness constant ks (α^-1)
            - damping constant kd (β)
            - value of the constraint C(x1, x2)
            - ∇C(x1)
            - ∇C(x2)
        """
        m1 = self.masses[spring.masses[0]]
        m2 = self.masses[spring.masses[1]]
        dx = m2.x - m1.x
        l = dx.length()
        dl = l - spring.l0
        dx_n = dx / l if l != 0.0 else dx
        return (spring.ks, spring.kd, dl, -dx_n, dx_n)

    def calc_acceleration(self, ext_only=False) -> None:
        # clear accelerations
        for mass in self.masses:
            mass.a = Vector2(0.0) if mass.inv_m == 0.0 else Vector2(self.g)
        # force field
        for mass in self.masses:
            if mass.inv_m != 0.0:
                mass.a += self.acc_field(mass.x, self.time)
        if ext_only:
            return
        # viscous drag
        for mass in self.masses:
            mass.a -= mass.drag * mass.v
        # spring forces
        for spring in self.springs:
            m1 = self.masses[spring.masses[0]]
            m2 = self.masses[spring.masses[1]]
            f1, f2 = self.get_spring_force(spring)
            m1.a += m1.inv_m * f1
            m2.a += m2.inv_m * f2

    def draw(self, surface: pygame.Surface, viewport: Viewport,
             mass_color=(192, 192, 192), spring_color=(128, 128, 128)):
        for spring in self.springs:
            p1 = self.masses[spring.masses[0]].x
            p2 = self.masses[spring.masses[1]].x
            q1 = viewport.world_to_screen(p1)
            q2 = viewport.world_to_screen(p2)
            if max(q1.length(), q2.length()) < 15360:
                pygame.draw.aaline(surface, spring_color, q1, q2)
        for mass in self.masses:
            q = viewport.world_to_screen(mass.x)+Vector2(0.5)
            pygame.draw.circle(surface, mass_color, q, 2)

    def calc_system_mass(self) -> float:
        """ sum of masses in kg """
        m = 0.0
        for mass in self.masses:
            if mass.inv_m != 0.0:
                m += 1.0 / mass.inv_m
        return m

    def calc_system_momentum(self) -> Vector2:
        """ sum of momentum of all masses in kg*m/s """
        p = Vector2(0.0)
        for mass in self.masses:
            if mass.inv_m != 0.0:
                p += mass.v / mass.inv_m
        return p

    def calc_system_energy(self) -> float:
        """ kinatic + gravitational potential + elastic potential, in Joules """
        kinetic = 0.0
        gravitational = 0.0
        for mass in self.masses:
            if mass.inv_m != 0.0:
                m = 1.0 / mass.inv_m
                kinetic += 0.5 * m * mass.v.length_squared()
                gravitational += m * -mass.x.dot(self.g)
        elastic = 0.0
        for spring in self.springs:
            p1 = self.masses[spring.masses[0]].x
            p2 = self.masses[spring.masses[1]].x
            dx = (p2 - p1).length() - spring.l0
            elastic += 0.5 * spring.ks * dx * dx
        return kinetic + gravitational + elastic

    def draw_info(self, surface: pygame.Surface, font: pygame.font.Font,
                  name: str, topleft: tuple[int, int], color=(255, 255, 255)) -> None:
        time = self.time
        velocity = self.calc_system_momentum() / self.calc_system_mass()
        energy = self.calc_system_energy()
        text = font.render(
            "{} {:.2f}s ({:.2f},{:.2f})m/s {:.2f}J".format(
                name, time, *velocity, energy),
            True, color)
        text_rect = text.get_rect()
        text_rect.topleft = topleft
        surface.blit(text, text_rect)
