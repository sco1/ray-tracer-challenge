import pytest

from ray_tracer import NUMERIC_T
from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.rays import Ray
from ray_tracer.transforms import Matrix, scaling, translation


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


RAY_TRANSFORMATION_CASES = (
    (translation(3, 4, 5), Ray(point(4, 6, 8), vector(0, 1, 0))),
    (scaling(2, 3, 4), Ray(point(2, 6, 12), vector(0, 3, 0))),
)


@pytest.mark.parametrize(("t_matrix", "truth_ray"), RAY_TRANSFORMATION_CASES)
def test_ray_transformation(t_matrix: Matrix, truth_ray: Ray) -> None:
    r = Ray(point(1, 2, 3), vector(0, 1, 0))
    assert r.transform(t_matrix) == truth_ray
