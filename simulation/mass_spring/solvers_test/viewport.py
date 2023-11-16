import pygame
from pygame import Vector2
import math


class Viewport:

    def __init__(self, resolution: Vector2, radius: float, center: Vector2):
        """Constructor
        Additional info:
            Create a viewport represented by the minimum and maximum coordinates of the points
        Args:
            resolution: the resolution of the screen, (width, height)
            radius: the maximum radius of a circle centered at (0, 0) that can be placed inside the viewport
            center: the center of the viewport in world coordinates"""
        sc = radius / min(resolution.x, resolution.y)
        self._resolution = resolution
        self._pmin = center - sc * resolution
        self._pmax = center + sc * resolution

    def fit_aabb(self, pmin: Vector2, pmax: Vector2, scale: float) -> None:
        """Update the viewport to fit the axes-aligned bounding box of an object
        Args:
            pmin: the minimum coordinates of the AABB of the object
            pmax: the maximum coordinates of the AABB of the object
            scale: the object appears to be scaled by this factor in the viewport
        """
        if not math.isfinite(pmin.dot(pmax)):
            return
        center = 0.5 * (pmin + pmax)
        radius = 0.5 * (pmax - pmin)
        msc = max(radius.x/self._resolution.x,
                  radius.y/self._resolution.y) / scale
        self._pmin = center - msc * self._resolution
        self._pmax = center + msc * self._resolution

    def world_to_screen(self, xy: Vector2) -> Vector2:
        """Convert the coordinates of a point from world space to screen space
        Args:
            xy: a point in world space
        Returns:
            the point with coordinates converted to screen space
        """
        xy = Vector2(xy)
        xy.x = self._resolution.x * \
            ((xy.x - self._pmin.x) / (self._pmax.x - self._pmin.x))
        xy.y = self._resolution.y * \
            ((xy.y - self._pmin.y) / (self._pmax.y - self._pmin.y))
        return Vector2(xy.x, -xy.y) + Vector2(0, self._resolution.y)

    def screen_to_world(self, xy: Vector2) -> Vector2:
        """Convert the coordinates of a point from screen space to world space
        Args:
            xy: a point in screen space
        Returns:
            the point with coordinates converted to world space
        """
        xy = Vector2(xy)
        xy = Vector2(xy.x, -xy.y) + Vector2(0, self._resolution.y)
        xy.x = self._pmin.x + (self._pmax.x - self._pmin.x) * \
            (xy.x / self._resolution.x)
        xy.y = self._pmin.y + (self._pmax.y - self._pmin.y) * \
            (xy.y / self._resolution.y)
        return xy

    def mouse_move(self, mouse_delta: Vector2) -> None:
        """Update the viewport when the mouse is moved"""
        if type(mouse_delta) != Vector2:
            mouse_delta = Vector2(*mouse_delta)
        p0 = self.screen_to_world((0, 0))
        p1 = self.screen_to_world(mouse_delta)
        dp = p1 - p0
        self._pmin -= dp
        self._pmax -= dp

    def mouse_scroll(self, mouse_pos: Vector2, zoom: float) -> None:
        """Update the viewport when the mouse wheel is scrolled
        Args:
            mouse_pos: the position of the mouse in world coordinate
            zoom: positive quantity, greater than 1 when zoom in, less than 1 when zoom out
        """
        if type(mouse_pos) != Vector2:
            mouse_pos = Vector2(*mouse_pos)
        if not zoom > 0.0:
            raise ValueError("Mouse zooming must be positive.")
        center = self.screen_to_world(mouse_pos)
        self._pmin = center + (self._pmin - center) / zoom
        self._pmax = center + (self._pmax - center) / zoom

    def draw(self, surface: pygame.Surface) -> None:
        """Draw axes and grid on a Pygame surface"""
        # background
        pygame.draw.rect(surface, (12, 12, 12), surface.get_rect())
        # grid
        GRID_COLOR = (32, 32, 32)
        view_delta = max(self._pmax.x - self._pmin.x,
                         self._pmax.y - self._pmin.y)
        grid_step = 10 ** round(math.log10(0.15*view_delta))
        p0 = self.screen_to_world((0, 0))
        p1 = self.screen_to_world((surface.get_width(), surface.get_height()))
        for x in range(math.floor(p0.x/grid_step), math.ceil(p1.x/grid_step)):
            x = self.world_to_screen((grid_step*x, 0)).x
            pygame.draw.line(surface, GRID_COLOR,
                             (x, 0), (x, surface.get_height()))
        for y in range(math.ceil(p0.y/grid_step), math.floor(p1.y/grid_step), -1):
            y = self.world_to_screen((0, grid_step*y)).y
            pygame.draw.line(surface, GRID_COLOR,
                             (0, y), (surface.get_width(), y))
        # axes
        AXES_COLOR = (64, 64, 64)
        origin = self.world_to_screen((0, 0))
        pygame.draw.line(surface, AXES_COLOR,
                         (0, origin.y), (surface.get_width(), origin.y))
        pygame.draw.line(surface, AXES_COLOR,
                         (origin.x, 0), (origin.x, surface.get_height()))
