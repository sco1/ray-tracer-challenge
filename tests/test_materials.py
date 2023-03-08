import pytest

from ray_tracer.materials import Material
from ray_tracer.rayple import color

BASE_COLOR = color(1, 1, 1)


def test_material_invalid_ambient_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(BASE_COLOR, -1, 1, 1, 1)


def test_material_invalid_diffuse_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(BASE_COLOR, 1, -1, 1, 1)


def test_material_invalid_specular_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(BASE_COLOR, 1, 1, -1, 1)


def test_material_invalid_shininess_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(BASE_COLOR, 1, 1, 1, -1)
