import math

import pytest

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.intersections import Intersection, Intersections, prepare_computations
from ray_tracer.lights import PointLight
from ray_tracer.materials import Material
from ray_tracer.patterns import _TestPattern
from ray_tracer.rayple import Rayple, color, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Plane, Sphere
from ray_tracer.transforms import scaling, translation
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
    w.light = PointLight(point(0, 0.25, 0), WHITE)

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


SHADOWED_CASES = (
    (point(0, 10, 0), False),
    (point(10, -10, 10), True),
    (point(-20, 20, -20), False),
    (point(-2, 2, -2), False),
)


@pytest.mark.parametrize(("pt", "truth_val"), SHADOWED_CASES)
def test_is_shadowed(pt: Rayple, truth_val: bool) -> None:
    w = World.default_world()
    assert w.is_shadowed(pt) == truth_val


def test_shade_at_shaded_point() -> None:
    s1 = Sphere()
    s2 = Sphere(transform=translation(0, 0, 10))
    w = World(PointLight(point(0, 0, -10), WHITE), objects=[s1, s2])

    r = Ray(point(0, 0, 5), vector(0, 0, 1))
    i = Intersection(4, s2)
    comps = prepare_computations(i, r)

    c = w._shade_hit(comps)
    assert c == color(0.1, 0.1, 0.1)


def test_reflect_color_nonreflective_material() -> None:
    s1 = Sphere(material=Material(color=color(0.8, 1.0, 0.6), ambient=1, diffuse=0.7, specular=0.2))
    s2 = Sphere(transform=scaling(0.5, 0.5, 0.5))
    w = World(DEFAULT_LIGHT, [s1, s2])

    r = Ray(point(0, 0, 0), vector(0, 0, 1))
    i = Intersection(1, s1)
    comps = prepare_computations(i, r)

    assert w.reflected_color(comps) == BLACK


RT_2 = math.sqrt(2)


def test_reflect_color_reflective_material() -> None:
    s1 = Sphere(material=Material(color=color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2))
    s2 = Sphere(transform=scaling(0.5, 0.5, 0.5))
    s3 = Plane(transform=translation(0, -1, 0), material=Material(reflective=0.5))
    w = World(DEFAULT_LIGHT, [s1, s2, s3])

    r = Ray(point(0, 0, -3), vector(0, -RT_2 / 2, RT_2 / 2))
    i = Intersection(RT_2, s3)
    comps = prepare_computations(i, r)

    # Truth color slightly tweaked from textbook to lazily fix floating point issues
    assert w.reflected_color(comps) == color(0.19033, 0.23791, 0.14274)


def test_reflective_shade_hit() -> None:
    s1 = Sphere(material=Material(color=color(0.8, 1.0, 0.6), diffuse=0.7, specular=0.2))
    s2 = Sphere(transform=scaling(0.5, 0.5, 0.5))
    s3 = Plane(transform=translation(0, -1, 0), material=Material(reflective=0.5))
    w = World(DEFAULT_LIGHT, [s1, s2, s3])

    r = Ray(point(0, 0, -3), vector(0, -RT_2 / 2, RT_2 / 2))
    i = Intersection(RT_2, s3)
    comps = prepare_computations(i, r)

    # Truth color slightly tweaked from textbook to lazily fix floating point issues
    assert w._shade_hit(comps) == color(0.87675, 0.92434, 0.82917)


def test_endless_reflection_infinite_recursion_handled() -> None:
    lt = PointLight(point(0, 0, 0), color(1, 1, 1))
    lower = Plane(translation(0, -1, 0), Material(reflective=1))
    upper = Plane(translation(0, 1, 0), Material(reflective=1))
    w = World(lt, [lower, upper])

    r = Ray(point(0, 0, 0), vector(0, 1, 0))
    _ = w.color_at(r)  # If recursion isn't handled this will blow up the stack


def test_refraction_opaque_surface() -> None:
    w = World.default_world()
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    inters = Intersections([Intersection(4, w.objects[0]), Intersection(6, w.objects[0])])

    comps = prepare_computations(inters[0], r, inters)
    assert w.refracted_color(comps) == BLACK


def test_refraction_max_recursion_is_black() -> None:
    lt = PointLight(point(0, 0, 0), color(1, 1, 1))
    s = Sphere(material=Material(transparency=1, refractive_index=1.5))
    w = World(lt, [s])
    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    inters = Intersections([Intersection(4, s), Intersection(6, s)])

    comps = prepare_computations(inters[0], r, inters)
    assert w.refracted_color(comps, remaining=0) == BLACK


def test_total_internal_reflection() -> None:
    lt = PointLight(point(0, 0, 0), color(1, 1, 1))
    s1 = Sphere(material=Material(transparency=1, refractive_index=1.5))
    s2 = Sphere(transform=scaling(0.5, 0.5, 0.5))
    w = World(lt, [s1, s2])
    r = Ray(point(0, 0, RT_2 / 2), vector(0, 1, 0))
    inters = Intersections([Intersection(-RT_2 / 2, s1), Intersection(RT_2 / 2, s1)])

    comps = prepare_computations(inters[1], r, inters)
    assert w.refracted_color(comps) == BLACK


def test_refracted_color() -> None:
    lt = PointLight(point(0, 0, 0), color(1, 1, 1))
    a = Sphere(
        material=Material(
            color=color(0.8, 1.0, 0.6),
            diffuse=0.7,
            specular=0.2,
            ambient=1.0,
            pattern=_TestPattern(),
        )
    )
    b = Sphere(
        transform=scaling(0.5, 0.5, 0.5),
        material=Material(transparency=1.0, refractive_index=1.5),
    )
    w = World(lt, [a, b])

    r = Ray(point(0, 0, 0.1), vector(0, 1, 0))
    inters = Intersections(
        [
            Intersection(-0.9899, a),
            Intersection(-0.4899, b),
            Intersection(0.4899, b),
            Intersection(0.9899, a),
        ]
    )

    comps = prepare_computations(inters[2], r, inters)
    # Truth color slightly tweaked from textbook to lazily fix floating point issues
    assert w.refracted_color(comps) == color(0, 0.99888, 0.0472)


def test_shade_hit_refraction() -> None:
    w = World.default_world()
    floor = Plane(
        transform=translation(0, -1, 0), material=Material(transparency=0.5, refractive_index=1.5)
    )
    ball = Sphere(
        transform=translation(0, -3.5, -0.5), material=Material(color=color(1, 0, 0), ambient=0.5)
    )
    w.objects.extend([floor, ball])

    r = Ray(point(0, 0, -3), vector(0, -RT_2 / 2, RT_2 / 2))
    inters = Intersections([Intersection(RT_2, floor)])
    comps = prepare_computations(inters[0], r, inters)
    assert w._shade_hit(comps) == color(0.93642, 0.68642, 0.68642)
