from dataclasses import dataclass

from ray_tracer import NUMERIC_T
from ray_tracer.rayple import Rayple, color


@dataclass(frozen=True, slots=True)
class Material:
    """
    Represent material attributes from the Phong reflection model.

    Reflecton attributes are assumed to be non-zero; `ambient`, `diffuse`, and `specular` values are
    typically between `0` and `1`, and `shininess` values are typically between `10` and `200`.

    Attribute magnitudes are not enforced beyond ensuring they are non-zero.
    """

    color: Rayple = color(1, 1, 1)
    ambient: NUMERIC_T = 0.1
    diffuse: NUMERIC_T = 0.9
    specular: NUMERIC_T = 0.9
    shininess: NUMERIC_T = 200

    def __post_init__(self) -> None:
        if any((val <= 0 for val in (self.ambient, self.diffuse, self.specular, self.shininess))):
            raise ValueError("Reflection attributes must be non-zero.")
