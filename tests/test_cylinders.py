import pytest

from ray_tracer import NUMERIC_T
from ray_tracer.intersections import Intersection
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Cylinder

DUMMY_INTER = Intersection(1, Cylinder(), 2, 3)

CYLINDER_MISS_CASES = (
    (point(1, 0, 0), vector(0, 1, 0)),
    (point(0, 0, 0), vector(0, 1, 0)),
    (point(0, 0, -5), vector(1, 1, 1)),
)


@pytest.mark.parametrize(("origin", "direction"), CYLINDER_MISS_CASES)
def test_cylinder_miss(origin: Rayple, direction: Rayple) -> None:
    cyl = Cylinder()
    norm_direction = direction.normalize()
    r = Ray(origin, norm_direction)

    inters = cyl._local_intersect(r)
    assert len(inters) == 0


CYLINDER_HIT_CASES = (
    (point(1, 0, -5), vector(0, 0, 1), 5, 5),
    (point(0, 0, -5), vector(0, 0, 1), 4, 6),
    (point(0.5, 0, -5), vector(0.1, 1, 1), 6.80798, 7.08872),
)


@pytest.mark.parametrize(("origin", "direction", "t0", "t1"), CYLINDER_HIT_CASES)
def test_cylinder_hit(origin: Rayple, direction: Rayple, t0: NUMERIC_T, t1: NUMERIC_T) -> None:
    cyl = Cylinder()
    norm_direction = direction.normalize()
    r = Ray(origin, norm_direction)

    inters = cyl._local_intersect(r)
    assert len(inters) == 2
    assert inters[0].t == pytest.approx(t0)
    assert inters[1].t == pytest.approx(t1)


CYLINDER_NORMAL_CASES = (
    (point(1, 0, 0), vector(1, 0, 0)),
    (point(0, 5, -1), vector(0, 0, -1)),
    (point(0, -2, 1), vector(0, 0, 1)),
    (point(-1, 1, 0), vector(-1, 0, 0)),
)


@pytest.mark.parametrize(("origin", "truth_normal"), CYLINDER_NORMAL_CASES)
def test_cylinder_normal(origin: Rayple, truth_normal: Rayple) -> None:
    cyl = Cylinder()
    assert cyl._local_normal_at(origin, DUMMY_INTER) == truth_normal


TRUNCATED_CYL_INTERSECT_CASES = (
    (point(0, 1.5, 0), vector(0.1, 1, 0), 0),
    (point(0, 3, -5), vector(0, 0, 1), 0),
    (point(0, 0, -5), vector(0, 0, 1), 0),
    (point(0, 2, -5), vector(0, 0, 1), 0),
    (point(0, 1, -5), vector(0, 0, 1), 0),
    (point(0, 1.5, -2), vector(0, 0, 1), 2),
)


@pytest.mark.parametrize(("origin", "direction", "truth_n_hits"), TRUNCATED_CYL_INTERSECT_CASES)
def test_truncated_cylinder_intersection(
    origin: Rayple, direction: Rayple, truth_n_hits: int
) -> None:
    cyl = Cylinder(minimum=1, maximum=2)
    norm_direction = direction.normalize()
    r = Ray(origin, norm_direction)

    inters = cyl._local_intersect(r)
    assert len(inters) == truth_n_hits


CAPPED_TRUNCATED_CYL_CASES = (
    (point(0, 3, 0), vector(0, -1, 0)),
    (point(0, 3, -2), vector(0, -1, 2)),
    (point(0, 4, -2), vector(0, -1, 1)),
    (point(0, 0, -2), vector(0, 1, 2)),
    (point(0, -1, -2), vector(0, 1, 1)),
)


@pytest.mark.parametrize(("origin", "direction"), CAPPED_TRUNCATED_CYL_CASES)
def test_capped_truncated_cylinder_intersection(origin: Rayple, direction: Rayple) -> None:
    cyl = Cylinder(minimum=1, maximum=2, closed=True)
    norm_direction = direction.normalize()
    r = Ray(origin, norm_direction)

    inters = cyl._local_intersect(r)
    assert len(inters) == 2


CAPPED_CYL_NORMAL_CASES = (
    (point(0, 1, 0), vector(0, -1, 0)),
    (point(0.5, 1, 0), vector(0, -1, 0)),
    (point(0, 1, 0.5), vector(0, -1, 0)),
    (point(0, 2, 0), vector(0, 1, 0)),
    (point(0.5, 2, 0), vector(0, 1, 0)),
    (point(0, 2, 0.5), vector(0, 1, 0)),
)


@pytest.mark.parametrize(("origin", "truth_normal"), CAPPED_CYL_NORMAL_CASES)
def test_capped_cylinder_normal(origin: Rayple, truth_normal: Rayple) -> None:
    cyl = Cylinder(minimum=1, maximum=2, closed=True)
    assert cyl._local_normal_at(origin, DUMMY_INTER) == truth_normal
