import pytest

from ray_tracer.colors import WHITE
from ray_tracer.materials import Material


def test_material_invalid_ambient_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(color=WHITE, pattern=None, ambient=-1, diffuse=1, specular=1, shininess=1)


def test_material_invalid_diffuse_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(color=WHITE, pattern=None, ambient=1, diffuse=-1, specular=1, shininess=1)


def test_material_invalid_specular_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(color=WHITE, pattern=None, ambient=1, diffuse=1, specular=-1, shininess=1)


def test_material_invalid_shininess_raises() -> None:
    with pytest.raises(ValueError):
        _ = Material(color=WHITE, pattern=None, ambient=1, diffuse=1, specular=1, shininess=-1)
