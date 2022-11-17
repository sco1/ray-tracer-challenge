import math

import pytest

from ray_tracer.base import Rayple, RaypleType, is_point, is_vector, point, vector, dot, cross

NUMERIC_T = int | float


def test_rayple_components() -> None:
    rp = Rayple(4.3, -4.2, 3.1, RaypleType.POINT)

    assert rp.x == pytest.approx(4.3)
    assert rp.y == pytest.approx(-4.2)
    assert rp.z == pytest.approx(3.1)
    assert rp.w == RaypleType.POINT


def test_rayple_helpers() -> None:
    p = point(1, 2, 3)
    v = vector(1, 2, 3)

    assert is_point(p)
    assert not is_point(v)

    assert is_vector(v)
    assert not is_vector(p)


EQ_CASES = (
    (point(1, 2, 3), point(1, 2, 3), True),
    (point(1, 2, 3), point(1.0, 2.0, 3.0), True),
    (vector(1, 2, 3), vector(1, 2, 3), True),
    (vector(1, 2, 3), vector(1.0, 2.0, 3.0), True),
    (point(1, 2, 3), point(-1.0, 2.0, 3.0), False),
    (vector(1, 2, 3), vector(-1.0, 2.0, 3.0), False),
    (point(1, 2, 3), vector(1, 2, 3), False),
    (point(1, 2, 3), 5, False),
    (vector(1, 2, 3), 5, False),
)


@pytest.mark.parametrize(("left", "right", "truth"), EQ_CASES)
def test_rayple_eq(left: Rayple, right: Rayple | int, truth: bool) -> None:
    assert (left == right) == truth


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


def test_diff_point_point() -> None:
    p1 = point(3, 2, 1)
    p2 = point(5, 6, 7)

    assert (p1 - p2) == vector(-2, -4, -6)


def test_diff_point_vector() -> None:
    p = point(3, 2, 1)
    v = vector(5, 6, 7)

    assert (p - v) == point(-2, -4, -6)


def test_diff_vector_vector() -> None:
    v1 = vector(3, 2, 1)
    v2 = vector(5, 6, 7)

    assert (v1 - v2) == vector(-2, -4, -6)


def test_diff_vector_point_raises() -> None:
    p = point(3, 2, 1)
    v = vector(5, 6, 7)

    with pytest.raises(ValueError):
        v - p


def test_diff_rayple_nonrayple_raises() -> None:
    with pytest.raises(TypeError):
        point(0, 0, 0) - 5


def test_negation() -> None:
    p = point(1, -2, 3)
    assert -p == point(-1, 2, -3)


SCALAR_MUL_CASES = (
    (point(1, -2, 3), 3.5, point(3.5, -7, 10.5)),
    (3.5, point(1, -2, 3), point(3.5, -7, 10.5)),
    (point(1, -2, 3), 3, point(3, -6, 9)),
    (3, point(1, -2, 3), point(3, -6, 9)),
)


@pytest.mark.parametrize(("left", "right", "truth"), SCALAR_MUL_CASES)
def test_scalar_multipliciation(
    left: Rayple | NUMERIC_T, right: Rayple | NUMERIC_T, truth: Rayple
) -> None:
    assert (left * right) == truth


def test_nonscalar_multiplication_raises() -> None:
    with pytest.raises(TypeError):
        point(1, 2, 3) * point(1, 2, 3)


def test_scalar_division() -> None:
    p = point(1, -2, 3)
    assert p / 2 == point(0.5, -1, 1.5)


def test_nonscalar_division_raises() -> None:
    p = point(1, -2, 3)
    with pytest.raises(TypeError):
        p / p


def test_scalar_rdivision_raises() -> None:
    p = point(1, -2, 3)
    with pytest.raises(TypeError):
        2 / p  # type: ignore[operator]


SQ_14 = math.sqrt(14)

VECTOR_MAG_CASES = (
    (vector(0, 1, 0), 1),
    (vector(0, 0, 1), 1),
    (vector(1, 2, 3), SQ_14),
    (vector(1, -2, 3), SQ_14),
    (vector(-1, -2, -3), SQ_14),
)


@pytest.mark.parametrize(("vec", "truth"), VECTOR_MAG_CASES)
def test_vector_magnitude(vec: Rayple, truth: NUMERIC_T) -> None:
    assert abs(vec) == pytest.approx(truth)


def test_point_magnitude_raises() -> None:
    with pytest.raises(ValueError):
        abs(point(1, 2, 3))


VECTOR_NORM_CASES = (
    (vector(4, 0, 0), vector(1, 0, 0)),
    (vector(1, 2, 3), vector(1 / SQ_14, 2 / SQ_14, 3 / SQ_14)),
)


@pytest.mark.parametrize(("vec", "truth"), VECTOR_NORM_CASES)
def test_vector_normalize(vec: Rayple, truth: Rayple) -> None:
    assert vec.normalize() == truth


def test_vector_norm_mag() -> None:
    v = vector(1, 2, 3)
    assert abs(v.normalize()) == pytest.approx(1)


def test_point_normalize_raises() -> None:
    with pytest.raises(ValueError):
        point(1, 2, 3).normalize()


def test_dot_product() -> None:
    v1 = vector(1, 2, 3)
    v2 = vector(2, 3, 4)

    assert dot(v1, v2) == 20


def test_dot_product_point_raises() -> None:
    p = point(1, 2, 3)
    with pytest.raises(ValueError):
        dot(p, p)


def test_cross_product() -> None:
    v1 = vector(1, 2, 3)
    v2 = vector(2, 3, 4)

    assert cross(v1, v2) == vector(-1, 2, -1)
    assert cross(v2, v1) == vector(1, -2, 1)


def test_cross_product_point_raises() -> None:
    p = point(1, 2, 3)
    with pytest.raises(ValueError):
        cross(p, p)
