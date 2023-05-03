import math

import pytest

from ray_tracer import NUMERIC_T
from ray_tracer.intersections import Intersection
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.shapes import Cone

DUMMY_INTER = Intersection(1, Cone(), 2, 3)

CONE_INTERSECTION_CASES = (
    (point(0, 0, -5), vector(0, 0, 1), 5, 5),
    (point(0, 0, -5), vector(1, 1, 1), 8.66025, 8.66025),
    (point(1, 1, -5), vector(-0.5, -1, 1), 4.55006, 49.44994),
)


@pytest.mark.parametrize(("origin", "direction", "t0", "t1"), CONE_INTERSECTION_CASES)
def test_cone_intersection(origin: Rayple, direction: Rayple, t0: NUMERIC_T, t1: NUMERIC_T) -> None:
    c = Cone()
    norm_direction = direction.normalize()
    r = Ray(origin, norm_direction)

    inters = c._local_intersect(r)
    assert len(inters) == 2
    assert inters[0].t == pytest.approx(t0)
    assert inters[1].t == pytest.approx(t1)


def test_cone_parallel_ray() -> None:
    c = Cone()
    norm_direction = vector(0, 1, 1).normalize()
    r = Ray(point(0, 0, -1), norm_direction)

    inters = c._local_intersect(r)
    assert len(inters) == 1
    assert inters[0].t == pytest.approx(0.3535533)


CAPPED_CONE_INTERSECTION_CASES = (
    (point(0, 0, -5), vector(0, 1, 0), 0),
    (point(0, 0, -0.25), vector(0, 1, 1), 2),
    (point(0, 0, -0.25), vector(0, 1, 0), 4),
)


@pytest.mark.parametrize(("origin", "direction", "truth_n_inters"), CAPPED_CONE_INTERSECTION_CASES)
def test_capped_cone_intersection(origin: Rayple, direction: Rayple, truth_n_inters: int) -> None:
    c = Cone(minimum=-0.5, maximum=0.5, closed=True)
    norm_direction = direction.normalize()
    r = Ray(origin, norm_direction)

    inters = c._local_intersect(r)
    assert len(inters) == truth_n_inters


CAPPED_CONE_NORMAL_CASES = (
    (point(0, 0, 0), vector(0, 0, 0)),
    (point(1, 1, 1), vector(1, -math.sqrt(2), 1)),
    (point(-1, -1, 0), vector(-1, 1, 0)),
)


@pytest.mark.parametrize(("origin", "truth_normal"), CAPPED_CONE_NORMAL_CASES)
def test_cone_normal(origin: Rayple, truth_normal: Rayple) -> None:
    c = Cone()
    assert c._local_normal_at(origin, DUMMY_INTER) == truth_normal
