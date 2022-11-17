import pytest

from ray_tracer.base import Rayple, RaypleType, point, vector


def test_rayple_components() -> None:
    rp = Rayple(4.3, -4.2, 3.1, RaypleType.POINT)

    assert rp.x == pytest.approx(4.3)
    assert rp.y == pytest.approx(-4.2)
    assert rp.z == pytest.approx(3.1)
    assert rp.w == RaypleType.POINT


def test_sum_point_vector() -> None:
    p = point(3, -2, 5)
    v = vector(-2, 3, 1)

    assert (p + v) == point(1, 1, 6)


def test_sum_vector_vector() -> None:
    v1 = vector(-2, 3, 1)
    v2 = vector(1, 1, 1)

    assert (v1 + v2) == vector(-1, 4, 2)


def test_sum_point_point_raises() -> None:
    p1 = point(0, 0, 0)
    p2 = point(1, 2, 3)

    with pytest.raises(ValueError):
        p1 + p2

def test_sum_rayple_nonrayple_raises() -> None:
    with pytest.raises(TypeError):
        point(0, 0, 0) + 5
