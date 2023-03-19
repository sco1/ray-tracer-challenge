from math import pi, sqrt

import pytest

from ray_tracer import NUMERIC_T
from ray_tracer.camera import Camera
from ray_tracer.rayple import color, point, vector
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix, rot_y, translation, view_transform
from ray_tracer.world import World

PIXEL_SIZE_CASES = (
    ((200, 125, pi / 2), 0.01),
    ((125, 200, pi / 2), 0.01),
)


@pytest.mark.parametrize(("camera_params", "truth_pixel_size"), PIXEL_SIZE_CASES)
def test_pixel_size_computation(
    camera_params: tuple[int, int, NUMERIC_T], truth_pixel_size: NUMERIC_T
) -> None:
    c = Camera(*camera_params)
    assert c.pixel_size == pytest.approx(truth_pixel_size)


RAY_FOR_PIXEL_CASES = (
    (Matrix.identity(), 100, 50, Ray(point(0, 0, 0), vector(0, 0, -1))),
    (Matrix.identity(), 0, 0, Ray(point(0, 0, 0), vector(0.66519, 0.33259, -0.66851))),
    (
        (rot_y(pi / 4) * translation(0, -2, 5)),
        100,
        50,
        Ray(point(0, 2, -5), vector(sqrt(2) / 2, 0, -(sqrt(2) / 2))),
    ),
)


@pytest.mark.parametrize(("transform", "x", "y", "truth_ray"), RAY_FOR_PIXEL_CASES)
def test_ray_for_pixel(transform: Matrix, x: int, y: int, truth_ray: Ray) -> None:
    c = Camera(201, 101, pi / 2, transform=transform)
    r = c.ray_for_pixel(x, y)

    assert r == truth_ray


def test_render() -> None:
    w = World.default_world()
    trans = view_transform(point(0, 0, -5), point(0, 0, 0), vector(0, 1, 0))
    c = Camera(11, 11, pi / 2, transform=trans)

    img = c.render(w)
    assert img.pixel_at(5, 5) == color(0.38066, 0.47583, 0.2855)
