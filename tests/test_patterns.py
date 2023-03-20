import pytest

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.patterns import Striped
from ray_tracer.rayple import Rayple, point
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
    pattern = Striped(WHITE, BLACK)
    assert pattern.at_point(pt) == truth_color


def test_striped_pattern_transform() -> None:
    obj = Sphere()
    pattern = Striped(transform=scaling(2, 2, 2))

    c = pattern.at_object(obj, point(1.5, 0, 0))
    assert c == WHITE


def test_striped_object_transform() -> None:
    obj = Sphere(transform=scaling(2, 2, 2))
    pattern = Striped()

    c = pattern.at_object(obj, point(1.5, 0, 0))
    assert c == WHITE


def test_striped_pattern_object_transform() -> None:
    obj = Sphere(transform=scaling(2, 2, 2))
    pattern = Striped(transform=translation(0.5, 0, 0))

    c = pattern.at_object(obj, point(2.5, 0, 0))
    assert c == WHITE
