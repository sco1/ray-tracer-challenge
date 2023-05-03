from functools import partial

import pytest

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Plane

DUMMY_INTER = Intersection(1, Plane(), 2, 3)

PLANE_NORMAL_CASES = (
    (point(0, 0, 0), vector(0, 1, 0)),
    (point(10, 0, -10), vector(0, 1, 0)),
    (point(-5, 0, 150), vector(0, 1, 0)),
)


@pytest.mark.parametrize(("pt", "truth_normal"), PLANE_NORMAL_CASES)
def test_plane_normal(pt: Rayple, truth_normal: Rayple) -> None:
    p = Plane()
    assert p.normal_at(pt, DUMMY_INTER) == truth_normal


# We're just testing intersection points so we want to share a plane object since those only
# compare equal by object ID
BASE_PLANE = Plane()
P_INT = partial(Intersection, obj=BASE_PLANE)
PLANE_INTERSECT_CASES = (
    (Ray(point(0, 10, 0), vector(0, 0, 1)), Intersections([])),  # parallel
    (Ray(point(0, 0, 0), vector(0, 0, 1)), Intersections([])),  # coplanar, treated as a miss
    (Ray(point(0, 1, 0), vector(0, -1, 0)), Intersections([P_INT(1.0)])),  # above
    (Ray(point(0, -1, 0), vector(0, 1, 0)), Intersections([P_INT(1.0)])),  # below
)


@pytest.mark.parametrize(("ray", "truth_intersection"), PLANE_INTERSECT_CASES)
def test_plane_intersection(ray: Ray, truth_intersection: list[float]) -> None:
    intersections = BASE_PLANE.intersect(ray)

    assert intersections == truth_intersection
