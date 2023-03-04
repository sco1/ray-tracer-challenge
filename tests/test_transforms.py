import math

import pytest

from ray_tracer.rayple import Rayple, point, vector
from ray_tracer.transforms import rot, rot_x, rot_y, rot_z, scaling, shearing, translation

R2O2 = math.sqrt(2) / 2
H_QUART = math.pi / 4
QUART = math.pi / 2


def test_matrix_nonrayple_mul_raises() -> None:
    m = translation(0, 0, 0)
    with pytest.raises(TypeError):
        _ = m * 1  # type: ignore[operator]


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


def test_rot_x() -> None:
    p = point(0, 1, 0)
    truth_half_quarter = point(0, R2O2, R2O2)
    truth_full_quarter = point(0, 0, 1)

    half_quarter = rot_x(H_QUART)
    assert half_quarter * p == truth_half_quarter

    full_quarter = rot_x(QUART)
    assert full_quarter * p == truth_full_quarter


def test_rot_y() -> None:
    p = point(0, 0, 1)
    truth_half_quarter = point(R2O2, 0, R2O2)
    truth_full_quarter = point(1, 0, 0)

    half_quarter = rot_y(H_QUART)
    assert half_quarter * p == truth_half_quarter

    full_quarter = rot_y(QUART)
    assert full_quarter * p == truth_full_quarter


def test_rot_z() -> None:
    p = point(0, 1, 0)
    truth_half_quarter = point(-R2O2, R2O2, 0)
    truth_full_quarter = point(-1, 0, 0)

    half_quarter = rot_z(H_QUART)
    assert half_quarter * p == truth_half_quarter

    full_quarter = rot_z(QUART)
    assert full_quarter * p == truth_full_quarter


def test_rot() -> None:
    p = point(0, 1, 0)
    truth_full_quarter = point(1, 0, 0)

    full_quarter = rot(x=QUART, y=QUART)
    assert full_quarter * p == truth_full_quarter


SHEARING_CASES = (
    ((0, 0, 0, 0, 0, 0), point(2, 3, 4)),
    ((1, 0, 0, 0, 0, 0), point(5, 3, 4)),
    ((0, 1, 0, 0, 0, 0), point(6, 3, 4)),
    ((0, 0, 1, 0, 0, 0), point(2, 5, 4)),
    ((0, 0, 0, 1, 0, 0), point(2, 7, 4)),
    ((0, 0, 0, 0, 1, 0), point(2, 3, 6)),
    ((0, 0, 0, 0, 0, 1), point(2, 3, 7)),
)


@pytest.mark.parametrize(("shear_params", "truth_point"), SHEARING_CASES)
def test_shearing(shear_params: tuple[int, int, int, int, int, int], truth_point: Rayple) -> None:
    p = point(2, 3, 4)

    shear = shearing(*shear_params)
    assert shear * p == truth_point


def test_transform_chaining() -> None:
    p = point(1, 0, 1)
    truth_p = point(15, 0, 7)

    # Non-chained progression
    A = rot_x(QUART)
    B = scaling(5, 5, 5)
    C = translation(10, 5, 7)

    step_p = A * p
    step_p = B * step_p
    step_p = C * step_p
    assert step_p == truth_p

    # Chained transformations
    chained = C * B * A
    assert chained * p == truth_p
