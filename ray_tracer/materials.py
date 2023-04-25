from dataclasses import dataclass
from enum import Enum

from ray_tracer import NUMERIC_T
from ray_tracer.colors import WHITE
from ray_tracer.patterns import Pattern
from ray_tracer.rayple import Rayple


class RefractionIndex(float, Enum):  # noqa: D101
    VACUUM = 1.0
    AIR = 1.00029
    WATER = 1.333
    GLASS = 1.52
    DIAMOND = 2.417


@dataclass(frozen=True, slots=True)
class Material:
    """
    Represent material attributes from the Phong reflection model.

    Reflection attributes are assumed to be positive and non-zero; `ambient`, `diffuse`,
    `reflective`, and `specular` values are typically between `0` and `1`, and `shininess` values
    are typically between `10` and `200`.

    Attribute magnitudes are not enforced beyond ensuring they are positive.
    """

    color: Rayple = WHITE
    pattern: Pattern | None = None
    ambient: NUMERIC_T = 0.1
    diffuse: NUMERIC_T = 0.9
    specular: NUMERIC_T = 0.9
    shininess: NUMERIC_T = 200
    reflective: NUMERIC_T = 0
    transparency: NUMERIC_T = 0
    refractive_index: NUMERIC_T = 1

    def __post_init__(self) -> None:
        if any(
            (
                val < 0
                for val in (
                    self.ambient,
                    self.diffuse,
                    self.specular,
                    self.shininess,
                    self.reflective,
                    self.transparency,
                    self.refractive_index,
                )
            )
        ):
            raise ValueError("Material reflection and refraction attributes must be non-negative.")
