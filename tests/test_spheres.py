import math
from functools import partial

import pytest

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere
from ray_tracer.transforms import Matrix, rot_z, scaling, translation

RT_2 = math.sqrt(2)
RT_3 = math.sqrt(3)

# We're just testing intersection points so we want to share a sphere object since those only
# compare equal by object ID
BASE_SPHERE = Sphere()

P_INT = partial(Intersection, obj=BASE_SPHERE)
UNIT_SPHERE_INTERSECT_CASES = (
    (Ray(point(0, 0, -5), vector(0, 0, 1)), Intersections([P_INT(4.0), P_INT(6.0)])),
    (Ray(point(0, 1, -5), vector(0, 0, 1)), Intersections([P_INT(5.0), P_INT(5.0)])),  # tangent
    (Ray(point(0, 2, -5), vector(0, 0, 1)), Intersections([])),  # miss
    (Ray(point(0, 0, 0), vector(0, 0, 1)), Intersections([P_INT(-1.0), P_INT(1.0)])),  # inside
    (Ray(point(0, 0, 5), vector(0, 0, 1)), Intersections([P_INT(-6.0), P_INT(-4.0)])),  # behind
)


@pytest.mark.parametrize(("ray", "truth_intersection"), UNIT_SPHERE_INTERSECT_CASES)
def test_unit_sphere_intersection(ray: Ray, truth_intersection: list[float]) -> None:
    intersections = BASE_SPHERE.intersect(ray)

    assert intersections == truth_intersection


TRANSFORMED_SPHERE_INTERSECT_CASES = (
    (scaling(2, 2, 2), (3.0, 7.0)),
    (translation(5, 0, 0), ()),
)


@pytest.mark.parametrize(("transform", "truth_t"), TRANSFORMED_SPHERE_INTERSECT_CASES)
def test_transformed_sphere_intersection(transform: Matrix, truth_t: tuple[float]) -> None:
    s = Sphere(transform)
    r = Ray(point(0, 0, -5), vector(0, 0, 1))

    intersections = s.intersect(r)
    truth_intersections = Intersections([Intersection(t, s) for t in truth_t])

    assert intersections == truth_intersections


UNIT_SPHERE_NORMAL_CASES = (
    (point(1, 0, 0), vector(1, 0, 0)),
    (point(0, 1, 0), vector(0, 1, 0)),
    (point(0, 0, 1), vector(0, 0, 1)),
    (point(0, 0, 1), vector(0, 0, 1)),
    (point(RT_3 / 3, RT_3 / 3, RT_3 / 3), vector(RT_3 / 3, RT_3 / 3, RT_3 / 3)),
)


@pytest.mark.parametrize(("query", "truth_vector"), UNIT_SPHERE_NORMAL_CASES)
def test_unit_sphere_normal(query: Rayple, truth_vector: Rayple) -> None:
    s = Sphere()

    assert s.normal_at(query) == truth_vector


def test_unit_sphere_normal_is_normalized() -> None:
    s = Sphere()
    n = s.normal_at(point(RT_3 / 3, RT_3 / 3, RT_3 / 3))

    assert n == n.normalize()


def test_normalize_non_point_raises() -> None:
    s = Sphere()
    with pytest.raises(ValueError):
        _ = s.normal_at(vector(1, 0, 0))


def test_normal_translated_sphere() -> None:
    s = Sphere(translation(0, 1, 0))

    query = point(0, 1 + RT_2 / 2, -RT_2 / 2)
    truth_vector = vector(0, RT_2 / 2, -RT_2 / 2)
    assert s.normal_at(query) == truth_vector


def test_normal_transformed_sphere() -> None:
    transform = scaling(1, 0.5, 1) * rot_z(math.pi / 5)
    s = Sphere(transform)

    query = point(0, RT_2 / 2, -RT_2 / 2)
    truth_vector = vector(0, 0.97014, -0.24253)

    assert s.normal_at(query) == truth_vector
