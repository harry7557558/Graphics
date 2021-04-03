# debug implicit integration for mass-spring simulation in 2d

from math import *
import numpy as np
import pygame


class vec2():
    """ 2d vector class """

    def __init__(self, x: float = 0.0, y: float = 0.0): self.x, self.y = x, y
    def __str__(self): return f"vec2({self.x}, {self.y})"
    def convert(self): return (self.x, self.y)
    def __neg__(self): return vec2(-self.x, -self.y)
    def __add__(self, other): return vec2(self.x+other.x, self.y+other.y)
    def __sub__(self, other): return vec2(self.x-other.x, self.y-other.y)

    def __mul__(self, other):
        if type(other) == float:
            return vec2(self.x*other, self.y*other)
        return vec2(self.x*other.x, self.y*other.y)

    def __truediv__(self, other):
        if type(other) == float:
            return vec2(self.x/other, self.y/other)
        return vec2(self.x/other.x, self.y/other.y)


def dot(u, v): return u.x*v.x+u.y*v.y
def length(v): return sqrt(v.x*v.x+v.y*v.y)
def cossin(x): return vec2(cos(x), sin(x))


class mass():
    def __init__(self, pos: vec2, m: float = 1.0, vd: float = 0.0, v: vec2 = vec2(0.0)):
        self.inv_m = 1.0/m  # reciprocal of mass
        self.vd = vd        # air resistance coefficient, f=-vd*v
        self.p = pos        # position
        self.v = v          # velocity
        self.f = vec2(0.0)  # net force acting on it


class spring:
    def __init__(self, m1: int, m2: int, l0: float = 0.0, ks: float = 1.0, kd: float = 0.0):
        self.l0 = l0                # rest length
        self.m1, self.m2 = m1, m2   # index of the two connected masses
        self.ks, self.kd = ks, kd   # spring constant and damping constant
        # |F|=ks*(length(Δx)-r)+kd*dot(Δv,normalize(Δx))


#
# ================================================================================================
#

class scene():
    def __init__(self, mass_list=[], spring_list=[]):
        self.mass_list = mass_list
        self.spring_list = spring_list
        self.time = 0

    def calc_force(self):
        mass_list = self.mass_list
        spring_list = self.spring_list
        mn = len(mass_list)
        sn = len(spring_list)
        for i in range(mn):
            self.mass_list[i].f = vec2()

        # gravity
        for i in range(mn):
            mass_list[i].f = vec2(0, -1) / max(mass_list[i].inv_m, 1e-12)

        # viscous drag
        for i in range(mn):
            mass_list[i].f -= mass_list[i].v * mass_list[i].vd

        # ground contact
        for i in range(mn):
            fn = 1000.*max(-mass_list[i].p.y, 0)
            mass_list[i].f += vec2(0, fn)

        # spring forces
        for i in range(sn):
            s = spring_list[i]
            m1, m2 = mass_list[s.m1], mass_list[s.m2]
            dp = m2.p - m1.p
            dv = m2.v - m1.v
            dpl = length(dp)
            edp = dp / dpl
            fm = s.ks*(dpl-s.l0) + s.kd*dot(dv, edp)
            f = edp * fm
            mass_list[s.m1].f += f
            mass_list[s.m2].f -= f

    def update_expeuler(self, dt):
        """ explicit Euler integration """
        self.calc_force()

        mass_list = self.mass_list

        for i in range(len(self.mass_list)):
            mass_list[i].v += mass_list[i].f * mass_list[i].inv_m * dt
            mass_list[i].p += mass_list[i].v * dt


#
# ================================================================================================
#

class PresetScenes():
    """ a list of preset object scenes """

    def box_x(self):
        masses = [mass(vec2(0, 1), .8, .1), mass(vec2(1, 1), 1., .1),
                  mass(vec2(1, 2), 1., .1), mass(vec2(0, 2), 1., .1)]
        rt2 = sqrt(2.) - 0.1
        springs = [spring(0, 1, 1.0), spring(1, 2, 1.0), spring(2, 3, 1.0),
                   spring(3, 0, 1.0), spring(0, 2, rt2), spring(1, 3, rt2)]
        for i in range(len(springs)):
            springs[i].ks = 100.
            springs[i].kd = 2.
        return scene(masses, springs)

    def box_xx(self, xD, yD, lx, ly, ang):
        dx, dy = lx/xD, ly/yD
        dl = hypot(dx, dy)
        # masses
        masses = []
        for y in range(0, yD+1):
            for x in range(0, xD+1):
                p = vec2(x, y)*vec2(dx, dy) - vec2(lx, ly)*0.5
                p = vec2(cos(ang)*p.x-sin(ang)*p.y,
                         sin(ang)*p.x+cos(ang)*p.y) + vec2(0, 2)
                masses.append(mass(p, 1.0, 0.1))
        # horizontal springs
        springs = []
        ks, kd = 100.0, 5.0
        for y in range(0, yD+1):
            for x in range(0, xD):
                springs.append(
                    spring(y*(xD+1)+x, y*(xD+1)+x+1, dx, ks/dx, kd/dx))
        # vertical springs
        for x in range(0, xD+1):
            for y in range(0, yD):
                springs.append(
                    spring(y*(xD+1)+x, (y+1)*(xD+1)+x, dy, ks/dy, kd/dy))
        # cross springs
        for x in range(xD):
            for y in range(yD):
                springs.append(
                    spring(y*(xD+1)+x, (y+1)*(xD+1)+x+1, dl, ks/dl, kd/dl))
                springs.append(spring(y*(xD+1)+x+1, (y+1) *
                                      (xD+1)+x, dl, ks/dl, kd/dl))
        return scene(masses, springs)

    def polygon_s(self, N, r1, r2):
        # masses
        masses = []
        for i in range(N):
            masses.append(mass(cossin(i*2*pi/N)*r1,
                               1.0+0.1*sin(1234.56*i+78.9), 0.1))
        for i in range(N):
            masses.append(mass(cossin(i*2*pi/N)*r2,
                               1.0+0.1*sin(3456.78*i+90.1), 0.1))
        for i in range(2*N):
            masses[i].p += vec2(0, 2)
        # springs
        springs = []
        ks, kd = 500.0, 5.0
        for i in range(N):
            # longitude springs
            springs.append(spring(i, (i+1) % N, 2.0*r1*sin(pi/N), ks, kd))
            springs.append(spring(i+N, (i+1) % N+N, 2.0*r2*sin(pi/N), ks, kd))
            # latitude springs
            springs.append(spring(i, i+N, abs(r2-r1), ks, kd))
            # cross springs
            springs.append(spring(i, (i+1) % N+N, nan, ks, kd))
            springs.append(spring(i+N, (i+1) % N, nan, ks, kd))
        # recalculate spring length
        for i in range(5*N):
            springs[i].l0 = length(
                masses[springs[i].m2].p - masses[springs[i].m1].p)
        return scene(masses, springs)

    def sheet_hang(self, xD, yD, lx, ly, has_cross=False, all_hang=False):
        """ this one is pretty stiff when @all_hang is set to False """
        dx, dy = lx/xD, ly/yD
        dl = hypot(dx, dy)
        # masses
        masses = []
        for y in range(0, yD+1):
            for x in range(0, xD+1):
                m = 10.0/((xD+1)*(yD+1))
                masses.append(mass(vec2(x, y)*vec2(lx/xD, ly/yD) -
                                   vec2(0.5*lx, ly)+vec2(0, 2.5), m, 0.1))
        # hang
        masses[yD*(xD+1)].inv_m = 0.0
        masses[yD*(xD+1)+xD].inv_m = 0.0
        if all_hang:
            for i in range(0, xD+1):
                masses[yD*(xD+1)+i].inv_m = 0.0
        # horizontal springs
        springs = []
        ks, kd = 10.0, 5.0
        for y in range(0, yD+1):
            for x in range(0, xD):
                springs.append(
                    spring(y*(xD+1)+x, y*(xD+1)+x+1, dx, ks/dx, kd/dx))
        # vertical springs
        for x in range(0, xD+1):
            for y in range(0, yD):
                springs.append(
                    spring(y*(xD+1)+x, (y+1)*(xD+1)+x, dy, ks/dy, kd/dy))
        # cross springs
        if has_cross:
            for x in range(xD):
                for y in range(yD):
                    springs.append(
                        spring(y*(xD+1)+x, (y+1)*(xD+1)+x+1, dl, ks/dl, kd/dl))
                    springs.append(
                        spring(y*(xD+1)+x+1, (y+1)*(xD+1)+x, dl, ks/dl, kd/dl))
        return scene(masses, springs)


obj = PresetScenes().box_x()
obj = PresetScenes().box_xx(5, 4, 1.0, 0.8, 0.2)
#obj = PresetScenes().box_xx(10, 8, 2.0, 1.6, 1.5)
#obj = PresetScenes().polygon_s(8, 0.5, 0.8)
#obj = PresetScenes().polygon_s(16, 0.5, 1.0)
#obj = PresetScenes().sheet_hang(16, 8, 3.0, 1.5, False, False)
obj_expeuler = obj


#
# ================================================================================================
#

pygame.init()

WIDTH = 600
HEIGHT = 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))

sc = 2.0/sqrt(WIDTH*HEIGHT)
center = vec2(0, 1)
pmin = vec2(-sc*WIDTH, -sc*HEIGHT)+center
pmax = vec2(sc*WIDTH, sc*HEIGHT)+center


def transform(p):
    xy = (p-pmin)/(pmax-pmin)*vec2(WIDTH, HEIGHT)
    return [xy.x, HEIGHT-xy.y]


def draw_object(screen, obj):
    for s in obj.spring_list:
        i, j = s.m1, s.m2
        pygame.draw.line(screen, (255, 255, 255),
                         transform(obj.mass_list[i].p), transform(obj.mass_list[j].p), 1)
    for m in obj.mass_list:
        pygame.draw.circle(screen, (255, 0, 0), transform(m.p), 2)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    # update objects
    N = int(0.01 / 0.01)
    dt = 0.01 / N
    for i in range(N):
        obj_expeuler.update_expeuler(dt)

    screen.fill((0, 0, 0))

    # draw axis
    pygame.draw.line(screen, (20, 20, 20),
                     transform(vec2(-10, 0)), transform(vec2(10, 0)), 1)
    pygame.draw.line(screen, (20, 20, 20),
                     transform(vec2(0, -10)), transform(vec2(0, 10)), 1)
    pygame.draw.line(screen, (40, 40, 40),
                     transform(vec2(0, 0)), transform(vec2(1, 0)), 1)
    pygame.draw.line(screen, (40, 40, 40),
                     transform(vec2(0, 0)), transform(vec2(0, 1)), 1)

    # draw objects
    draw_object(screen, obj_expeuler)

    pygame.display.flip()
    pygame.time.Clock().tick(100)

pygame.quit()
