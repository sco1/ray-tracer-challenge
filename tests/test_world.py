import pytest

from ray_tracer.intersections import Intersection, prepare_computations
from ray_tracer.lights import PointLight
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, color, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere
from ray_tracer.transforms import scaling
from ray_tracer.world import DEFAULT_LIGHT, World


def test_default_world_intersection() -> None:
    w = World.default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))

    intersects = w.intersect_world(r)
    assert len(intersects) == 4
    assert [intersect.t for intersect in intersects] == [4, 4.5, 5.5, 6]


def test_shade_intersection() -> None:
    w = World.default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    i = Intersection(4, w.objects[0])

    comps = prepare_computations(i, r)
    c = w._shade_hit(comps)
    assert c == color(0.38066, 0.47583, 0.2855)


def test_shade_inside_intersection() -> None:
    w = World.default_world()
    w.light = PointLight(point(0, 0.25, 0), color(1, 1, 1))

    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    i = Intersection(0.5, w.objects[1])

    comps = prepare_computations(i, r)
    c = w._shade_hit(comps)
    assert c == color(0.90498, 0.90498, 0.90498)


RAY_COLOR_CASES = (
    (Ray(point(0, 0, -5), vector(0, 1, 0)), color(0, 0, 0)),  # Ray miss
    (Ray(point(0, 0, -5), vector(0, 0, 1)), color(0.38066, 0.47583, 0.2855)),  # Ray hit
)


@pytest.mark.parametrize(("r", "truth_color"), RAY_COLOR_CASES)
def test_color_at(r: Ray, truth_color: Rayple) -> None:
    w = World.default_world()
    assert w.color_at(r) == truth_color


def test_color_at_intersection_behind_ray() -> None:
    s1 = Sphere(material=Material(ambient=1))  # Outer
    s2 = Sphere(material=Material(ambient=1), transform=scaling(0.5, 0.5, 0.5))  # Inner
    w = World(light=DEFAULT_LIGHT, objects=[s1, s2])

    r = Ray(point(0, 0, 0.75), vector(0, 0, -1))
    assert w.color_at(r) == w.objects[1].material.color
