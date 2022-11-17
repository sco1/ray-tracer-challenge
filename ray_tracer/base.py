from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

NUMERIC_T = int | float


class RaypleType(IntEnum):
    VECTOR = 0
    POINT = 1


@dataclass(frozen=True, slots=True)
class Rayple:
    x: NUMERIC_T
    y: NUMERIC_T
    z: NUMERIC_T
    w: RaypleType

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rayple):
            return NotImplemented

        return all(
            (
                math.isclose(self.x, other.x),
                math.isclose(self.y, other.y),
                math.isclose(self.z, other.z),
                self.w == other.w,
            )
        )

    def __add__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        if self.w == RaypleType.POINT and other.w == RaypleType.POINT:
            raise ValueError("Cannot add two points.")

        return Rayple(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
            w=RaypleType(self.w + other.w),
        )


def point(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    return Rayple(x, y, z, RaypleType.POINT)


def vector(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    return Rayple(x, y, z, RaypleType.VECTOR)
