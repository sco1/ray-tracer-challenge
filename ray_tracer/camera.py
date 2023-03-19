import math
from dataclasses import dataclass, field
from itertools import product

from ray_tracer import NUMERIC_T
from ray_tracer.canvas import Canvas
from ray_tracer.rayple import point
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix
from ray_tracer.world import World


@dataclass(slots=True)
class Camera:  # noqa: D101
    h_size: int
    v_size: int
    fov: NUMERIC_T  # radians
    transform: Matrix = field(default_factory=Matrix.identity)

    pixel_size: NUMERIC_T = field(init=False)
    _half_width: NUMERIC_T = field(init=False)
    _half_height: NUMERIC_T = field(init=False)

    def __post_init__(self) -> None:
        """
        Determine pixel scaling from the provided camera parameters.

        NOTE: FOV is assumed to be provided in radians.
        """
        # To simplify calculations, it is assumed that the camera's canvas is exactly one unit in
        # front of the camera
        half_view = math.tan(self.fov / 2)
        aspect_ratio = self.h_size / self.v_size
        if aspect_ratio >= 1:
            self._half_width = half_view
            self._half_height = half_view / aspect_ratio
        else:
            self._half_width = half_view * aspect_ratio
            self._half_height = half_view

        self.pixel_size = (self._half_width * 2) / self.h_size

    def ray_for_pixel(self, x: int, y: int) -> Ray:
        """Compute a ray from the camera to the center of the pixel at the given XY coordinates."""
        x_offset = (x + 0.5) * self.pixel_size
        y_offset = (y + 0.5) * self.pixel_size

        # Determine the untransformed coordinates of the pixel in world space
        world_x = self._half_width - x_offset
        world_y = self._half_height - y_offset

        # Then transform the canvas point & origin in order to compute the ray's direction
        # Since we're assuming that the canvas is exactly one unit in front of the camera, we can
        # say that z = -1
        inv_trans = self.transform.inv()
        pixel = inv_trans * point(world_x, world_y, -1)
        origin = inv_trans * point(0, 0, 0)
        direction = (pixel - origin).normalize()

        return Ray(origin, direction)

    def render(self, world: World) -> Canvas:
        """Render the camera's current fiew of the world."""
        img = Canvas(self.h_size, self.v_size)
        for y, x in product(range(self.v_size - 1), range(self.h_size - 1)):
            r = self.ray_for_pixel(x, y)
            c = world.color_at(r)
            img.write_pixel(x, y, c)

        return img
