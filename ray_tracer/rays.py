from __future__ import annotations

from dataclasses import dataclass

from ray_tracer import NUMERIC_T
from ray_tracer.rayple import Rayple, RaypleType
from ray_tracer.transforms import Matrix


@dataclass(frozen=True, slots=True)
class Ray:  # noqa: D101
    origin: Rayple
    direction: Rayple

    def __post_init__(self) -> None:
        if self.origin.w != RaypleType.POINT:
            raise ValueError("Ray origin must be a point")

        if self.direction.w != RaypleType.VECTOR:
            raise ValueError("Ray direction must be a vector")

    def position(self, t: NUMERIC_T) -> Rayple:
        """Calculate the ray's position after time `t`, assuming the ray moves one unit per `t`."""
        return self.origin + (self.direction * t)

    def transform(self, t_matrix: Matrix) -> Ray:
        """
        Return a copy of the current `Ray` with the provided tranformation applied.

        The transformation is applied to both the origin and direction of the current ray; note that
        the direction vector is not normalized after the transformation is applied.
        """
        new_origin = t_matrix * self.origin
        new_direction = t_matrix * self.direction

        return Ray(new_origin, new_direction)
