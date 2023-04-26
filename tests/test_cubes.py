import pytest

from ray_tracer import NUMERIC_T
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Cube

CUBE_RAY_INTERSECTION_CASES = (
    (point(5, 0.5, 0), vector(-1, 0, 0), 4, 6),
    (point(-5, 0.5, 0), vector(1, 0, 0), 4, 6),
    (point(0.5, 5, 0), vector(0, -1, 0), 4, 6),
    (point(0.5, -5, 0), vector(0, 1, 0), 4, 6),
    (point(0.5, 0, 5), vector(0, 0, -1), 4, 6),
    (point(0.5, 0, -5), vector(0, 0, 1), 4, 6),
    (point(0, 0.5, 0), vector(0, 0, 1), -1, 1),
)


@pytest.mark.parametrize(("origin", "direction", "t1", "t2"), CUBE_RAY_INTERSECTION_CASES)
def test_cube_ray_intersection(
    origin: Rayple, direction: Rayple, t1: NUMERIC_T, t2: NUMERIC_T
) -> None:
    c = Cube()
    r = Ray(origin, direction)

    inters = c._local_intersect(r)
    assert len(inters) == 2
    assert inters[0].t == pytest.approx(t1)
    assert inters[1].t == pytest.approx(t2)


CUBE_RAY_MISS_CASES = (
    (point(-2, 0, 0), vector(0.2673, 0.5345, 0.8018)),
    (point(0, -2, 0), vector(0.8018, 0.2673, 0.5345)),
    (point(0, 0, -2), vector(0.5345, 0.8018, 0.2673)),
    (point(2, 0, 2), vector(0, 0, -1)),
    (point(0, 2, 2), vector(0, -1, 0)),
    (point(2, 2, 0), vector(-1, 0, 0)),
)


@pytest.mark.parametrize(("origin", "direction"), CUBE_RAY_MISS_CASES)
def test_cube_ray_miss(origin: Rayple, direction: Rayple) -> None:
    c = Cube()
    r = Ray(origin, direction)

    inters = c._local_intersect(r)
    assert len(inters) == 0


CUBE_NORMAL_CASES = (
    (point(1, 0.5, -0.8), vector(1, 0, 0)),
    (point(-1, -0.2, 0.9), vector(-1, 0, 0)),
    (point(-0.4, 1, -0.1), vector(0, 1, 0)),
    (point(0.3, -1, -0.7), vector(0, -1, 0)),
    (point(-0.6, 0.3, 1), vector(0, 0, 1)),
    (point(0.4, 0.4, -1), vector(0, 0, -1)),
    (point(1, 1, 1), vector(1, 0, 0)),
    (point(-1, -1, -1), vector(-1, 0, 0)),
)


@pytest.mark.parametrize(("origin", "truth_normal"), CUBE_NORMAL_CASES)
def test_cube_normal(origin: Rayple, truth_normal: Rayple) -> None:
    c = Cube()
    assert c._local_normal_at(origin) == truth_normal
