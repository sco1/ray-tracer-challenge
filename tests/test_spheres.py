from functools import partial

import pytest

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere


def test_sphere_components() -> None:
    s = Sphere()

    assert s.origin == point(0, 0, 0)
    assert s.radius == 1


def test_sphere_nonpoint_origin_raises() -> None:
    with pytest.raises(ValueError):
        _ = Sphere(vector(1, 0, 0))


def test_sphere_bad_radius_raises() -> None:
    with pytest.raises(ValueError):
        _ = Sphere(radius=-1)

    with pytest.raises(ValueError):
        _ = Sphere(radius=0)


P_INT = partial(Intersection, obj=Sphere())
UNIT_SPHERE_INTERSECT_CASES = (
    (Ray(point(0, 0, -5), vector(0, 0, 1)), Intersections([P_INT(4.0), P_INT(6.0)])),
    (Ray(point(0, 1, -5), vector(0, 0, 1)), Intersections([P_INT(5.0), P_INT(5.0)])),  # tangent
    (Ray(point(0, 2, -5), vector(0, 0, 1)), Intersections([])),  # miss
    (Ray(point(0, 0, 0), vector(0, 0, 1)), Intersections([P_INT(-1.0), P_INT(1.0)])),  # inside
    (Ray(point(0, 0, 5), vector(0, 0, 1)), Intersections([P_INT(-6.0), P_INT(-4.0)])),  # behind
)


@pytest.mark.parametrize(("ray", "truth_intersection"), UNIT_SPHERE_INTERSECT_CASES)
def test_unit_sphere_intersection(ray: Ray, truth_intersection: list[float]) -> None:
    s = Sphere()
    intersections = s.intersect(ray)

    assert intersections == truth_intersection
