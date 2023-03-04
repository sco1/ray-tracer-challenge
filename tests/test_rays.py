import pytest

from ray_tracer import NUMERIC_T
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray


def test_ray_components() -> None:
    origin = point(1, 2, 3)
    direction = vector(4, 5, 6)
    r = Ray(origin, direction)

    assert r.origin == origin
    assert r.direction == direction


def test_ray_nonpoint_origin_raises() -> None:
    v = vector(4, 5, 6)

    with pytest.raises(ValueError):
        _ = Ray(v, v)


def test_ray_nonvector_direction_raises() -> None:
    p = point(1, 2, 3)

    with pytest.raises(ValueError):
        _ = Ray(p, p)


POSITION_CASES = (
    (0, point(2, 3, 4)),
    (1, point(3, 3, 4)),
    (-1, point(1, 3, 4)),
    (2.5, point(4.5, 3, 4)),
)


@pytest.mark.parametrize(("t", "truth_position"), POSITION_CASES)
def test_ray_position(t: NUMERIC_T, truth_position: Rayple) -> None:
    origin = point(2, 3, 4)
    direction = vector(1, 0, 0)
    r = Ray(origin, direction)

    assert r.position(t) == truth_position
