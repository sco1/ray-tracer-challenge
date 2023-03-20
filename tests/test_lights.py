import math
from functools import partial

import pytest

from ray_tracer.colors import BLACK, WHITE
from ray_tracer.lights import PointLight, lighting
from ray_tracer.materials import Material
from ray_tracer.patterns import Stripe
from ray_tracer.rayple import Rayple, RaypleType, color, point, vector
from ray_tracer.shapes import Sphere

RT2_O2 = math.sqrt(2) / 2


BASE_MATERIAL = Material()
BASE_POSITION = point(0, 0, 0)
BASE_NORM = vector(0, 0, -1)
DUMMY_SHAPE = Sphere()
LIGHTING_P = partial(
    lighting, normal=BASE_NORM, material=BASE_MATERIAL, surf_pos=BASE_POSITION, obj=DUMMY_SHAPE
)
LIGHT_P = partial(PointLight, intensity=WHITE)

ILLUMINATION_TEST_CASES = (
    (vector(0, 0, -1), LIGHT_P(point(0, 0, -10)), color(1.9, 1.9, 1.9)),
    (vector(0, RT2_O2, -RT2_O2), LIGHT_P(point(0, 0, -10)), WHITE),
    (vector(0, 0, -1), LIGHT_P(point(0, 10, -10)), color(0.7364, 0.7364, 0.7364)),
    (vector(0, -RT2_O2, -RT2_O2), LIGHT_P(point(0, 10, -10)), color(1.6364, 1.6364, 1.6364)),
    (vector(0, 0, -1), LIGHT_P(point(0, 0, 10)), color(0.1, 0.1, 0.1)),
)


@pytest.mark.parametrize(("eye_v", "light", "truth_lit"), ILLUMINATION_TEST_CASES)
def test_lighting(eye_v: Rayple, light: PointLight, truth_lit: Rayple) -> None:
    lit = LIGHTING_P(light=light, eye_v=eye_v)

    assert lit.w == RaypleType.COLOR
    assert lit == truth_lit


def test_lighting_nonpoint_surface_raises() -> None:
    with pytest.raises(ValueError):
        _ = lighting(
            material=BASE_MATERIAL,
            obj=DUMMY_SHAPE,
            light=LIGHT_P(point(0, 0, -10)),
            surf_pos=vector(0, 0, 1),  # Should be a point
            eye_v=vector(0, 0, -1),
            normal=BASE_NORM,
        )


def test_lighting_nonvector_eye_raises() -> None:
    with pytest.raises(ValueError):
        _ = lighting(
            material=BASE_MATERIAL,
            obj=DUMMY_SHAPE,
            light=LIGHT_P(point(0, 0, -10)),
            surf_pos=BASE_POSITION,
            eye_v=point(0, 0, 1),  # Should be a vector
            normal=BASE_NORM,
        )


def test_lighting_nonvector_normal_raises() -> None:
    with pytest.raises(ValueError):
        _ = lighting(
            material=BASE_MATERIAL,
            obj=DUMMY_SHAPE,
            light=LIGHT_P(point(0, 0, -10)),
            surf_pos=BASE_POSITION,
            eye_v=vector(0, 0, -1),
            normal=point(0, 0, 1),  # Should be a vector
        )


def test_lighting_in_shadow() -> None:
    eye_v = vector(0, 0, -1)
    light = LIGHT_P(point(0, 0, -10))

    lit = LIGHTING_P(light=light, eye_v=eye_v, in_shadow=True)
    assert lit == color(0.1, 0.1, 0.1)


def test_lighting_with_pattern() -> None:
    m = Material(pattern=Stripe(), ambient=1, diffuse=0, specular=0)
    obj = Sphere()
    eye_v = vector(0, 0, -1)
    normal = vector(0, 0, -1)
    light = PointLight(point(0, 0, -10), WHITE)

    c1 = lighting(
        material=m, obj=obj, light=light, surf_pos=point(0.9, 0, 0), eye_v=eye_v, normal=normal
    )
    assert c1 == WHITE

    c2 = lighting(
        material=m, obj=obj, light=light, surf_pos=point(1.1, 0, 0), eye_v=eye_v, normal=normal
    )
    assert c2 == BLACK
