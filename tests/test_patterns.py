import pytest

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.patterns import Checker, Gradient, Ring, Stripe
from ray_tracer.rayple import Rayple, color, point
from ray_tracer.shapes import Sphere
from ray_tracer.transforms import scaling, translation

STRIPED_TEST_CASES = (
    (point(0, 0, 0), WHITE),
    (point(0.9, 0, 0), WHITE),
    (point(1, 0, 0), BLACK),
    (point(1.1, 0, 0), BLACK),
    (point(-0.1, 0, 0), BLACK),
    (point(-1, 0, 0), BLACK),
    (point(-1.1, 0, 0), WHITE),
    # Constant along y
    (point(0, 1, 0), WHITE),
    (point(0, 2, 0), WHITE),
    # Constant along z
    (point(0, 0, 1), WHITE),
    (point(0, 0, 2), WHITE),
)


@pytest.mark.parametrize(("pt", "truth_color"), STRIPED_TEST_CASES)
def test_striped_colors(pt: Rayple, truth_color: Rayple) -> None:
    pattern = Stripe(WHITE, BLACK)
    assert pattern.at_point(pt) == truth_color


def test_striped_pattern_transform() -> None:
    obj = Sphere()
    pattern = Stripe(transform=scaling(2, 2, 2))

    c = pattern.at_object(obj, point(1.5, 0, 0))
    assert c == WHITE


def test_striped_object_transform() -> None:
    obj = Sphere(transform=scaling(2, 2, 2))
    pattern = Stripe()

    c = pattern.at_object(obj, point(1.5, 0, 0))
    assert c == WHITE


def test_striped_pattern_object_transform() -> None:
    obj = Sphere(transform=scaling(2, 2, 2))
    pattern = Stripe(transform=translation(0.5, 0, 0))

    c = pattern.at_object(obj, point(2.5, 0, 0))
    assert c == WHITE


GRADIENT_CASES = (
    (point(0, 0, 0), WHITE),
    (point(0.25, 0, 0), color(0.75, 0.75, 0.75)),
    (point(0.5, 0, 0), color(0.5, 0.5, 0.5)),
    (point(0.75, 0, 0), color(0.25, 0.25, 0.25)),
)


@pytest.mark.parametrize(("pt", "truth_color"), GRADIENT_CASES)
def test_gradient(pt: Rayple, truth_color: Rayple) -> None:
    pattern = Gradient()
    assert pattern.at_point(pt) == truth_color


RING_CASES = (
    (point(0, 0, 0), WHITE),
    (point(1, 0, 0), BLACK),
    (point(0, 0, 1), BLACK),
    (point(0.708, 0, 0.708), BLACK),
)


@pytest.mark.parametrize(("pt", "truth_color"), RING_CASES)
def test_ring(pt: Rayple, truth_color: Rayple) -> None:
    pattern = Ring()
    assert pattern.at_point(pt) == truth_color


CHECKER_CASES = (
    (point(0, 0, 0), WHITE),
    (point(0.99, 0, 0), WHITE),
    (point(0, 0.99, 0), WHITE),
    (point(0, 0, 0.99), WHITE),
    (point(1.01, 0, 0), BLACK),
    (point(0, 1.01, 0), BLACK),
    (point(0, 0, 1.01), BLACK),
)


@pytest.mark.parametrize(("pt", "truth_color"), CHECKER_CASES)
def test_checker(pt: Rayple, truth_color: Rayple) -> None:
    pattern = Checker()
    assert pattern.at_point(pt) == truth_color
