from functools import partial

import pytest

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere
from ray_tracer.transforms import Matrix, scaling, translation

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
