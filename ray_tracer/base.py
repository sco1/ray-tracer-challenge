from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

NUMERIC_T = int | float


class RaypleType(IntEnum):  # noqa: D101
    VECTOR = 0
    POINT = 1


@dataclass(frozen=True, slots=True)
class Rayple:
    """
    The Ray Tracer's generic ordered list of (x,y,z) things.

    The following operations are supported:
        * Vector/Point addition
            * Adding two points is undefined
            * Scalar addition is undefined
        * Vector/Point subtraction
            * Subtracting a point from a vector is undefined
            * Scalar subtraction is undefined
        * Negation
        * Scalar multiplication
        * Scalar division
    """

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

    def __sub__(self, other: object) -> Rayple:
        if not isinstance(other, Rayple):
            return NotImplemented

        if self.w == RaypleType.VECTOR and other.w == RaypleType.POINT:
            raise ValueError("Cannot subtract a point from a vector.")

        return Rayple(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
            w=RaypleType(self.w - other.w),
        )

    def __neg__(self) -> Rayple:
        return Rayple(
            x=-self.x,
            y=-self.y,
            z=-self.z,
            w=self.w,
        )

    def __mul__(self, other: object) -> Rayple:
        if not isinstance(other, (int, float)):
            return NotImplemented

        return Rayple(
            x=self.x * other,
            y=self.y * other,
            z=self.z * other,
            w=self.w,
        )

    def __rmul__(self, other: object) -> Rayple:
        return self * other

    def __truediv__(self, other: object) -> Rayple:
        if not isinstance(other, (int, float)):
            return NotImplemented

        return Rayple(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
            w=self.w,
        )


def point(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    """Shortcut for a point `Rayple` (`w = 1`)."""
    return Rayple(x, y, z, RaypleType.POINT)


def vector(x: NUMERIC_T, y: NUMERIC_T, z: NUMERIC_T) -> Rayple:
    """Shortcut for a vector `Rayple` (`w = 0`)."""
    return Rayple(x, y, z, RaypleType.VECTOR)
