import pytest

from ray_tracer.rayple import point, vector
from ray_tracer.transforms import scaling, translation


def test_matrix_nonrayple_mul_raises() -> None:
    m = translation(0, 0, 0)
    with pytest.raises(TypeError):
        _ = m * 1


def test_translation() -> None:
    p = point(-3, 4, 5)
    truth_shifted = point(2, 1, 7)

    shift = translation(5, -3, 2)
    assert shift * p == truth_shifted


def test_inverse_translation() -> None:
    p = point(-3, 4, 5)
    truth_shifted = point(-8, 7, 3)

    shift = translation(5, -3, 2).inv()
    assert shift * p == truth_shifted


def test_vector_translation_unchanged() -> None:
    v = vector(-3, 4, 5)

    shift = translation(5, -3, 2)
    assert shift * v == v


def test_point_scaling() -> None:
    p = point(-4, 6, 8)
    truth_scaled = point(-8, 18, 32)

    scaled = scaling(2, 3, 4)
    assert scaled * p == truth_scaled


def test_vector_scaling_grow() -> None:
    v = vector(-4, 6, 8)
    truth_scaled = vector(-8, 18, 32)

    scaled = scaling(2, 3, 4)
    assert scaled * v == truth_scaled


def test_vector_scaling_shrink() -> None:
    v = vector(-4, 6, 8)
    truth_scaled = vector(-2, 2, 2)

    scaled = scaling(2, 3, 4).inv()
    assert scaled * v == truth_scaled


def test_reflection() -> None:
    p = point(2, 3, 4)
    truth_reflected = point(-2, 3, 4)

    scaled = scaling(-1, 1, 1)
    assert scaled * p == truth_reflected
