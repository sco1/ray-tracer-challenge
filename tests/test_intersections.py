from functools import partial

import pytest

from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.shapes import Sphere


def test_intersection_components() -> None:
    t = 3.5
    s = Sphere()
    intersect = Intersection(t, s)

    assert intersect.t == t
    assert intersect.obj == s


def test_intersections_container() -> None:
    s = Sphere()
    intersections = Intersections([Intersection(1, s), Intersection(2, s)])

    assert len(intersections) == 2
    assert intersections[0].t == 1
    assert intersections[1].t == 2


def test_intersections_sorted() -> None:
    s = Sphere()
    intersections = Intersections([Intersection(2, s), Intersection(1, s)])

    assert intersections[0].t == 1


P_INT = partial(Intersection, obj=Sphere())
HIT_TEST_CASES = (
    (Intersections([P_INT(1), P_INT(2)]), P_INT(1)),
    (Intersections([P_INT(-1), P_INT(1)]), P_INT(1)),
    (Intersections([P_INT(-2), P_INT(-1)]), None),
    (Intersections([P_INT(5), P_INT(7), P_INT(-3), P_INT(2)]), P_INT(2)),
)


@pytest.mark.parametrize(("intersections", "truth_hit"), HIT_TEST_CASES)
def test_hit(intersections: Intersections, truth_hit: Intersection) -> None:
    assert intersections.hit == truth_hit
