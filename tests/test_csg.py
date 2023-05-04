import pytest

from ray_tracer.csg import CSG, Operation
from ray_tracer.intersections import Intersection, Intersections
from ray_tracer.rayple import point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Cube, Sphere
from ray_tracer.transforms import translation

ALLOWED_INTERSECTION_CASES = (
    (Operation.UNION, True, True, True, False),
    (Operation.UNION, True, True, False, True),
    (Operation.UNION, True, False, True, False),
    (Operation.UNION, True, False, False, True),
    (Operation.UNION, False, True, True, False),
    (Operation.UNION, False, True, False, False),
    (Operation.UNION, False, False, True, True),
    (Operation.UNION, False, False, False, True),
    (Operation.INTERSECTION, True, True, True, True),
    (Operation.INTERSECTION, True, True, False, False),
    (Operation.INTERSECTION, True, False, True, True),
    (Operation.INTERSECTION, True, False, False, False),
    (Operation.INTERSECTION, False, True, True, True),
    (Operation.INTERSECTION, False, True, False, True),
    (Operation.INTERSECTION, False, False, True, False),
    (Operation.INTERSECTION, False, False, False, False),
    (Operation.DIFFERENCE, True, True, True, False),
    (Operation.DIFFERENCE, True, True, False, True),
    (Operation.DIFFERENCE, True, False, True, False),
    (Operation.DIFFERENCE, True, False, False, True),
    (Operation.DIFFERENCE, False, True, True, True),
    (Operation.DIFFERENCE, False, True, False, True),
    (Operation.DIFFERENCE, False, False, True, False),
    (Operation.DIFFERENCE, False, False, False, False),
)


@pytest.mark.parametrize(
    ("op", "left_hit", "in_left", "in_right", "should_inter"), ALLOWED_INTERSECTION_CASES
)
def test_allowed_intersection(
    op: Operation, left_hit: bool, in_left: bool, in_right: bool, should_inter: bool
) -> None:
    geo = CSG(operation=op, left_shape=Sphere(), right_shape=Cube())

    chk = geo._is_inter_allowed(left_hit=left_hit, in_left=in_left, in_right=in_right)
    assert chk == should_inter


INTERSECTION_FILTER_CASES = (
    (Operation.UNION, 0, 3),
    (Operation.INTERSECTION, 1, 2),
    (Operation.DIFFERENCE, 0, 1),
)


@pytest.mark.parametrize(("op", "x0", "x1"), INTERSECTION_FILTER_CASES)
def test_intersection_filter(op: Operation, x0: int, x1: int) -> None:
    s1 = Sphere()
    s2 = Cube()
    geo = CSG(operation=op, left_shape=s1, right_shape=s2)

    inters = Intersections(
        (Intersection(1, s1), Intersection(2, s2), Intersection(3, s1), Intersection(4, s2))
    )

    filtered = geo._filter_intersections(inters)
    assert len(filtered) == 2
    assert filtered[0] == inters[x0]
    assert filtered[1] == inters[x1]


def test_ray_misses_csg() -> None:
    geo = CSG(operation=Operation.UNION, left_shape=Sphere(), right_shape=Cube())
    r = Ray(point(0, 2, -5), vector(0, 0, 1))

    inters = geo._local_intersect(r)
    assert len(inters) == 0


def test_ray_hits_csg() -> None:
    s1 = Sphere()
    s2 = Sphere(transform=translation(0, 0, 0.5))
    geo = CSG(operation=Operation.UNION, left_shape=s1, right_shape=s2)

    r = Ray(point(0, 0, -5), vector(0, 0, 1))
    inters = geo._local_intersect(r)
    assert len(inters) == 2

    assert inters[0].t == pytest.approx(4)
    assert inters[0].obj == s1

    assert inters[1].t == pytest.approx(6.5)
    assert inters[1].obj == s2
