from functools import partial

import pytest

from ray_tracer.intersections import Comps, Intersection, Intersections, prepare_computations
from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Sphere


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


BASE_SHAPE = Sphere()
COMPUTATIONS_CASES = (
    (
        Ray(point(0, 0, -5), vector(0, 0, 1)),
        Intersection(4, BASE_SHAPE),
        Comps(
            t=4,
            obj=BASE_SHAPE,
            point=point(0, 0, -1),
            eye_v=vector(0, 0, -1),
            normal=vector(0, 0, -1),
            inside=False,
        ),
    ),
    (
        Ray(point(0, 0, 0), vector(0, 0, 1)),
        Intersection(1, BASE_SHAPE),
        Comps(
            t=1,
            obj=BASE_SHAPE,
            point=point(0, 0, 1),
            eye_v=vector(0, 0, -1),
            normal=vector(0, 0, -1),
            inside=True,
        ),
    ),
)


@pytest.mark.parametrize(("r", "inter", "truth_comp"), COMPUTATIONS_CASES)
def test_prepare_computations(r: Ray, inter: Intersection, truth_comp: Comps) -> None:
    comps = prepare_computations(inter, r)

    assert comps == truth_comp
