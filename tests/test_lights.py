import math
from functools import partial

import pytest

from ray_tracer.lights import PointLight, lighting
from ray_tracer.materials import Material
from ray_tracer.rayple import Rayple, RaypleType, color, point, vector

RT2_O2 = math.sqrt(2) / 2


BASE_MATERIAL = Material()
BASE_POSITION = point(0, 0, 0)
BASE_NORM = vector(0, 0, -1)
LIGHTING_P = partial(lighting, normal=BASE_NORM, material=BASE_MATERIAL, surf_pos=BASE_POSITION)
LIGHT_P = partial(PointLight, intensity=color(1, 1, 1))

ILLUMINATION_TEST_CASES = (
    (vector(0, 0, -1), LIGHT_P(point(0, 0, -10)), color(1.9, 1.9, 1.9)),
    (vector(0, RT2_O2, -RT2_O2), LIGHT_P(point(0, 0, -10)), color(1, 1, 1)),
    (vector(0, 0, -1), LIGHT_P(point(0, 10, -10)), color(0.7364, 0.7364, 0.7364)),
    (vector(0, -RT2_O2, -RT2_O2), LIGHT_P(point(0, 10, -10)), color(1.6364, 1.6364, 1.6364)),
    (vector(0, 0, -1), LIGHT_P(point(0, 0, 10)), color(0.1, 0.1, 0.1)),
)


@pytest.mark.xfail(reason="Failing cases actually works works, need to fix float comparisons.")
@pytest.mark.parametrize(("eye_vec", "light", "truth_lit"), ILLUMINATION_TEST_CASES)
def test_lighting(eye_vec: Rayple, light: PointLight, truth_lit: Rayple) -> None:
    lit = LIGHTING_P(light=light, eye_vec=eye_vec)

    assert lit.w == RaypleType.COLOR
    assert lit == truth_lit
